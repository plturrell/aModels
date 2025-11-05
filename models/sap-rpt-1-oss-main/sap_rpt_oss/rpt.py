# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Union

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from sap_rpt_oss.constants import ZMQ_PORT_DEFAULT, ModelSize
from sap_rpt_oss.scripts.start_embedding_server import start_embedding_server
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.torch_model import RPT

warnings.filterwarnings('ignore', message='.*not support non-writable tensors.*')


def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            x[k] = to_device(v, device, dtype=dtype)
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x


class SAP_RPT_OSS_Estimator(BaseEstimator, ABC):
    """SAP_RPT_OSS_Estimator (sap-rpt-1-oss) class.

    Args:
        checkpoint: path to the checkpoint file; must be of size base model size
        bagging: number of bagging iterations; if 1 there is no bagging. If 'auto', then:
            - There is no bagging if the number of samples is less than max_context_size - just use everything
            - Otherwise, the training data is split into chunks size max_context_size rows:
            ceil(len(dataset) / max_context_size) overlapping chunks and each chunk is then used as a bagging
            iteration (capped at MAX_AUTO_BAGS = 16).
        max_context_size: maximum number of samples to use for training
        num_regression_bins: number of bins to use for regression (to convert into classification).
            Unused if regression_type is 'l2'.
        regression_type: regression type that was used in the specified model
            - reg-as-classif - binned regression where bin is associated with the quantile of a given column
            - l2 - direct prediction of the target value with L2 loss during training
        classification_type: classification type that was used in the specified model
            - cross-entropy - class likelihood prediction using cross entropy loss during training
            - clustering - class prediction using similarity between context and query vectors
            - clustering-cosine - class prediction using cosine similarity between context and query vectors 
        drop_constant_columns: flag to indicate to drop constant columns in the input dataframe
        test_chunk_size: Batch size of test rows to use for prediction at once
    """
    classification_or_regression: str
    MAX_AUTO_BAGS = 16
    MAX_NUM_COLUMNS = 500

    def __init__(self,
                 checkpoint: str = '2025-11-04_sap-rpt-one-oss.pt',
                 bagging: Union[Literal['auto'], int] = 8,
                 max_context_size: int = 8192,
                 drop_constant_columns: bool = True,
                 test_chunk_size: int = 1000,
                 device_ids: Optional[list] = None,
                 gpu_orchestrator_url: Optional[str] = None):

        self.model_size = ModelSize.base
        self.checkpoint = checkpoint
        self.regression_type = "l2"
        self.classification_type = "cross-entropy"
        self.test_chunk_size = test_chunk_size
        self._checkpoint_path = hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=checkpoint)
        self.bagging = bagging
        if not isinstance(bagging, int) and bagging != 'auto':
            raise ValueError('bagging must be an integer or "auto"')
        self.max_context_size = max_context_size
        self.num_regression_bins = 16
        self.model = RPT(self.model_size, regression_type=self.regression_type, classification_type=self.classification_type)
        
        # GPU allocation via orchestrator if available
        self.gpu_orchestrator_url = gpu_orchestrator_url or os.getenv("GPU_ORCHESTRATOR_URL")
        self.device_ids = device_ids
        self.allocation_id = None
        
        if self.gpu_orchestrator_url and not device_ids:
            self._allocate_gpus()
        
        # Set device(s) - use first device_id if provided, otherwise default to cuda:0
        if torch.cuda.is_available():
            if self.device_ids and len(self.device_ids) > 0:
                self.device = torch.device(f'cuda:{self.device_ids[0]}')
            else:
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        # Start embedding server(s) - use first GPU if multiple, otherwise use default
        if self.device_ids and len(self.device_ids) > 0:
            start_embedding_server(Tokenizer.sentence_embedding_model_name, device_ids=self.device_ids)
        else:
            start_embedding_server(Tokenizer.sentence_embedding_model_name)
        
        if torch.cuda.is_available():
            # Use first device for capability check
            device_idx = self.device_ids[0] if self.device_ids else 0
            if torch.cuda.get_device_capability(device_idx)[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(Path(self._checkpoint_path), self.device)
        self.seed = 42
        self.drop_constant_columns = drop_constant_columns
        
        # Use first device_id for tokenizer if multiple GPUs are available
        zmq_port = ZMQ_PORT_DEFAULT
        if self.device_ids and len(self.device_ids) > 1:
            # For multiple GPUs, we can use different ports or distribute embedding generation
            # For now, use default port with first GPU
            zmq_port = ZMQ_PORT_DEFAULT
        
        self.tokenizer = Tokenizer(
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            zmq_port=zmq_port,
            random_seed=self.seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=True)
        self.model.to(self.device).eval()
        
        # Wrap model with DataParallel if multiple GPUs are available
        if self.device_ids and len(self.device_ids) > 1 and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
    
    def _allocate_gpus(self):
        """Allocate GPUs from GPU orchestrator."""
        if not self.gpu_orchestrator_url:
            return
        
        try:
            import httpx
            request_data = {
                "service_name": "sap-rpt-oss",
                "workload_type": "embedding",
                "workload_data": {
                    "model_size": "medium",
                    "batch_size": 32
                }
            }
            
            response = httpx.post(
                f"{self.gpu_orchestrator_url}/gpu/allocate",
                json=request_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                allocation = response.json()
                self.allocation_id = allocation.get("id")
                self.device_ids = allocation.get("gpu_ids", [])
                import warnings
                warnings.warn(f"Allocated GPUs {self.device_ids} from orchestrator (allocation ID: {self.allocation_id})")
            else:
                import warnings
                warnings.warn(f"Failed to allocate GPUs: {response.status_code}")
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to allocate GPUs from orchestrator: {e}")
    
    def __del__(self):
        """Cleanup GPU allocation on destruction."""
        if self.allocation_id and self.gpu_orchestrator_url:
            try:
                import httpx
                httpx.post(
                    f"{self.gpu_orchestrator_url}/gpu/release",
                    json={"allocation_id": self.allocation_id},
                    timeout=5.0
                )
            except Exception:
                pass  # Ignore errors during cleanup
    
    def release_gpus(self):
        """Manually release GPU allocation."""
        if self.allocation_id and self.gpu_orchestrator_url:
            try:
                import httpx
                httpx.post(
                    f"{self.gpu_orchestrator_url}/gpu/release",
                    json={"allocation_id": self.allocation_id},
                    timeout=5.0
                )
                self.allocation_id = None
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to release GPU allocation: {e}")

    @abstractmethod
    def task_specific_fit(self):
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the model.

        Args:
            X: The input dataframe.
            y: The target column.
        """
        if len(X) != len(y):
            raise ValueError('X and y must have the same length')
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='TARGET')

        self.X_ = X

        self.bagging_config = self.bagging
        if X.shape[0] < self.max_context_size:
            self.bagging_config = 1

        self.y_ = y
        # Return the classifier
        self.task_specific_fit()
        return self

    @property
    def bagging_number(self):
        check_is_fitted(self)
        if self.bagging_config == 'auto':
            return min(self.MAX_AUTO_BAGS, ceil(len(self.X_) / self.max_context_size))
        else:
            assert isinstance(self.bagging_config, int)
            return self.bagging_config

    def get_tokenized_data(self, X_test, bagging_index):
        X_train = self.X_
        y_train = self.y_

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        y_test = pd.Series([y_train.iloc[0]] * len(X_test), name=self.y_.name, index=X_test.index)

        df_train = pd.concat([X_train, y_train.to_frame()], axis=1)
        df_test = pd.concat([X_test, y_test.to_frame()], axis=1)

        if isinstance(self.bagging_config, int) and self.bagging_config > 1:
            # For bagging, we use replacement
            df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)
        elif len(df_train) > self.max_context_size:
            if isinstance(self.bagging_config, str):
                assert self.bagging_config == 'auto'
                # Split the data into overlapping chunks of size max_context_size
                # Randomize order as well, to have balanced bags
                # bagging_index = 0 --> select 0:max_context_size
                # ... (linearly spaced like np.linspace(0, len(df_train) - max_context_size, self.bagging_number))
                # bagging_index = self.bagging_number - 1 --> select (len(df_train) - max_context_size):len(df_train)
                start = int((len(df_train) - self.max_context_size) / (self.bagging_number - 1) * bagging_index)
                # We need a fixed seed, so across diffent "bagging folds" we select the correct indices
                # (as non-overlapping as possible)
                np.random.seed(self.seed)
                indices = np.random.permutation(df_train.index)
                end = start + self.max_context_size
                df_train = df_train.loc[indices[start:end]]
            else:
                # There is no bagging, but we still have to sample because there are too many points
                df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)

        df = pd.concat([df_train, df_test], ignore_index=True)

        if self.drop_constant_columns:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            constant_cols = list(X.columns[X.nunique() == 1])
            if constant_cols:
                X = X.drop(columns=constant_cols)
                df = pd.concat([X, y], axis=1)

        if df.shape[1] > self.MAX_NUM_COLUMNS:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            X = X.sample(n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed + bagging_index, replace=False)
            df = pd.concat([X, y], axis=1)

        df_train = df.iloc[:len(df_train)]
        df_test = df.iloc[len(df_train):]
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1:]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1:]

        data, labels, label_classes = self.tokenizer(X_train, y_train, X_test, y_test,
                                                     self.classification_or_regression)

        target_mean, target_std = 0, 0
        is_regression = self.classification_or_regression == 'regression'
        if is_regression and self.regression_type == 'l2':
            _, target_mean, target_std = self.tokenizer.standard_scale_column(y_train, y_test)

        return {
            'data': data,
            'num_rows': df.shape[0],
            'num_cols': df.shape[1],
            'labels': None,
            'is_regression': torch.tensor(is_regression),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[list, np.ndarray]:
        pass


class SAP_RPT_OSS_Classifier(ClassifierMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = 'classification'

    def task_specific_fit(self):
        # Store the classes seen during fit
        self.classes_ = unique_labels(self.y_)

    def reorder_logits(self, logits, tokenized_classes):
        class_mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        indices = np.array([class_mapping[cls] for cls in tokenized_classes])
        new_logits = torch.full((logits.shape[0], len(self.classes_)),
                                float('-inf'),
                                dtype=logits.dtype,
                                device=logits.device)
        new_logits[:, indices] = logits[:, :len(tokenized_classes)]
        return new_logits

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        # Check if fit has been called
        check_is_fitted(self)

        all_logits = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)

            try:
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False, dtype=self.dtype)
            except TypeError:
                # Legacy compatibility
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            logits_classif = self.model(**tokenized_data)

            _, logits = self.model.extract_prediction_classification(logits_classif, tokenized_data['data']['target'],
                                                                     tokenized_data['label_classes'])

            all_logits.append(self.reorder_logits(logits, tokenized_data['label_classes']))

        all_logits = torch.stack(all_logits)
        if self.classification_type in ['clustering', 'clustering-cosine']:
            # Pick the class of the most similar element across folds, then softmax
            all_logits = torch.max(all_logits, dim=0).values.cpu().float()
            probs = torch.nn.functional.softmax(all_logits, dim=-1)
        else:
            # Average probabilities across folds, i.e. first softmax, then mean
            all_probs = torch.nn.functional.softmax(all_logits, dim=-1)
            probs = all_probs.mean(dim=0)

        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> list:
        """Predict the class labels for the provided input dataframe.

        Args:
            X: The input dataframe.

        Returns:
            The predicted class labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            probs.append(pred)
        probs = torch.cat(probs)

        preds = probs.argmax(dim=-1).numpy()
        return [self.classes_[p] for p in preds]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input dataframe.

        Args:
            X: The input data.

        Returns:
            The predicted probabilities of the classes.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            probs.append(pred)
        probs = torch.cat(probs)
        return probs.numpy()


class SAP_RPT_OSS_Regressor(RegressorMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = 'regression'

    def task_specific_fit(self):
        self.y_ = self.y_.astype(float)

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        # Check if fit has been called
        check_is_fitted(self)

        all_preds = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)
            tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            logits_reg = self.model(**tokenized_data)
            label_classes = tokenized_data['label_classes']

            if self.regression_type != 'l2':
                if len(label_classes) != self.num_regression_bins:
                    raise ValueError(f'Expected {self.num_regression_bins} classes, got {len(label_classes)}')

            preds, _ = self.model.extract_prediction_regression(logits_reg,
                                                                tokenized_data['data']['target'],
                                                                tokenized_data['label_classes'],
                                                                target_mean=tokenized_data.get('target_mean'),
                                                                target_std=tokenized_data.get('target_std'))
            all_preds.append(preds)

        preds = np.mean(all_preds, axis=0)

        return preds

    @torch.no_grad()
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        preds = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            preds.append(pred)
        preds = np.concatenate(preds)
        return preds
