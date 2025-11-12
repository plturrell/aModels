#!/usr/bin/env python3
"""
Unified Multi-Modal Extraction Pipeline (Phase 6)
Integrates DeepSeek-OCR, RelationalTransformer, sap-rpt-1-oss, and SentencePiece
for comprehensive multi-modal data extraction and embedding generation.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import base64
from io import BytesIO

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# DeepSeek-OCR path
DEEPSEEK_OCR_PATH = REPO_ROOT / "models" / "DeepSeek-OCR" / "DeepSeek-OCR-master"
if DEEPSEEK_OCR_PATH.exists():
    sys.path.insert(0, str(DEEPSEEK_OCR_PATH / "DeepSeek-OCR-hf"))

# RelationalTransformer path
RELATIONAL_TRANSFORMER_PATH = REPO_ROOT / "models" / "relational_transformer"
if RELATIONAL_TRANSFORMER_PATH.exists():
    sys.path.insert(0, str(RELATIONAL_TRANSFORMER_PATH))

# sap-rpt-1-oss path
SAP_RPT_PATH = REPO_ROOT / "models" / "sap-rpt-1-oss-main"
if SAP_RPT_PATH.exists():
    sys.path.insert(0, str(SAP_RPT_PATH))

try:
    import torch
    import pandas as pd
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print(json.dumps({"error": f"Missing dependencies: {exc}"}))
    sys.exit(1)

# Optional imports with fallbacks
try:
    from transformers import AutoModel, AutoTokenizer
    HAS_DEEPSEEK_OCR = True
except ImportError:
    HAS_DEEPSEEK_OCR = False

try:
    from relational_transformer.data import (
        RelationalDatabase,
        RelationalTableSpec,
        CellTokenizer,
    )
    from relational_transformer.model import RelationalTransformer
    HAS_RELATIONAL_TRANSFORMER = True
except ImportError:
    HAS_RELATIONAL_TRANSFORMER = False

try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
    from sap_rpt_oss.data.tokenizer import Tokenizer
    HAS_SAP_RPT = True
except ImportError:
    HAS_SAP_RPT = False


class UnifiedMultiModalExtractor:
    """Unified extractor combining OCR, relational embeddings, and semantic embeddings."""
    
    def __init__(self):
        self.deepseek_model = None
        self.deepseek_tokenizer = None
        self.relational_model = None
        self.sap_rpt_classifier = None
        self.sap_rpt_regressor = None
        self._initialized = False
    
    def initialize(self):
        """Initialize all models (lazy loading)."""
        if self._initialized:
            return
        
        # Initialize DeepSeek-OCR if available
        if HAS_DEEPSEEK_OCR and os.getenv("USE_DEEPSEEK_OCR", "false").lower() == "true":
            try:
                model_name = 'deepseek-ai/DeepSeek-OCR'
                self.deepseek_tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                self.deepseek_model = AutoModel.from_pretrained(
                    model_name,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True
                )
                if torch.cuda.is_available():
                    self.deepseek_model = self.deepseek_model.eval().cuda().to(torch.bfloat16)
                else:
                    self.deepseek_model = self.deepseek_model.eval()
            except Exception as e:
                print(f"Failed to initialize DeepSeek-OCR: {e}", file=sys.stderr)
        
        # RelationalTransformer is loaded on-demand in embedding generation
        # sap-rpt-1-oss is loaded on-demand in classification/regression
        
        self._initialized = True
    
    def extract_from_image(self, image_path: str, prompt: str = None) -> Dict[str, Any]:
        """Extract text and tables from image using DeepSeek-OCR."""
        if not HAS_DEEPSEEK_OCR or self.deepseek_model is None:
            return {
                "error": "DeepSeek-OCR not available",
                "text": "",
                "tables": [],
            }
        
        try:
            if prompt is None:
                prompt = "<image>\n<|grounding|>Convert the document to markdown."
            
            # Run OCR inference
            output_path = "/tmp/deepseek_ocr_output"
            os.makedirs(output_path, exist_ok=True)
            
            res = self.deepseek_model.infer(
                self.deepseek_tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_path,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False
            )
            
            # Parse result to extract text and tables
            extracted_text = res.get("text", "") if isinstance(res, dict) else str(res)
            
            # Try to extract tables from markdown
            tables = self._extract_tables_from_markdown(extracted_text)
            
            return {
                "text": extracted_text,
                "tables": tables,
                "method": "deepseek-ocr",
            }
        except Exception as e:
            return {
                "error": str(e),
                "text": "",
                "tables": [],
            }
    
    def extract_from_image_base64(self, image_base64: str, prompt: str = None) -> Dict[str, Any]:
        """Extract text and tables from base64-encoded image."""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Save to temp file
            temp_path = "/tmp/temp_ocr_image.png"
            image.save(temp_path)
            
            # Extract
            result = self.extract_from_image(temp_path, prompt)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "text": "",
                "tables": [],
            }
    
    def _extract_tables_from_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Extract table structures from markdown text."""
        tables = []
        lines = markdown_text.split('\n')
        
        in_table = False
        current_table = []
        headers = []
        
        for line in lines:
            line = line.strip()
            if '|' in line and not line.startswith('|--'):
                if not in_table:
                    in_table = True
                    headers = [h.strip() for h in line.split('|') if h.strip()]
                    current_table = []
                else:
                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(row) == len(headers):
                        current_table.append(row)
            elif in_table:
                # End of table
                if current_table:
                    tables.append({
                        "headers": headers,
                        "rows": current_table,
                        "row_count": len(current_table),
                        "column_count": len(headers),
                    })
                in_table = False
                current_table = []
                headers = []
        
        # Handle last table
        if in_table and current_table:
            tables.append({
                "headers": headers,
                "rows": current_table,
                "row_count": len(current_table),
                "column_count": len(headers),
            })
        
        return tables
    
    def generate_unified_embeddings(
        self,
        text: str = None,
        tables: List[Dict] = None,
        image_path: str = None,
        table_name: str = None,
        columns: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate unified embeddings using all available models."""
        result = {
            "relational_embedding": None,
            "semantic_embedding": None,
            "tokenized_text": None,
            "embeddings": {},
        }
        
        # Step 1: Extract from image if provided
        if image_path:
            ocr_result = self.extract_from_image(image_path)
            if ocr_result.get("text"):
                text = ocr_result.get("text", text)
            if ocr_result.get("tables"):
                tables = ocr_result.get("tables", tables)
        
        # Step 2: Generate relational embedding if we have structured data
        if (table_name and columns) or tables:
            try:
                if HAS_RELATIONAL_TRANSFORMER:
                    relational_emb = self._generate_relational_embedding(
                        table_name, columns, tables
                    )
                    result["relational_embedding"] = relational_emb
                    result["embeddings"]["relational_transformer"] = relational_emb
            except Exception as e:
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Relational embedding: {str(e)}")
        
        # Step 3: Generate semantic embedding using sap-rpt-1-oss
        if text or (table_name and columns):
            try:
                if HAS_SAP_RPT:
                    semantic_emb = self._generate_semantic_embedding(text, table_name, columns)
                    result["semantic_embedding"] = semantic_emb
                    result["embeddings"]["sap_rpt_semantic"] = semantic_emb
            except Exception as e:
                result["errors"] = result.get("errors", [])
                result["errors"].append(f"Semantic embedding: {str(e)}")
        
        # Step 4: Tokenize text (for SentencePiece integration via Go)
        if text:
            result["tokenized_text"] = {
                "text": text,
                "length": len(text),
                "word_count": len(text.split()),
            }
        
        return result
    
    def _generate_relational_embedding(
        self,
        table_name: str = None,
        columns: List[Dict] = None,
        tables: List[Dict] = None,
    ) -> Optional[List[float]]:
        """Generate relational embedding using RelationalTransformer."""
        if not HAS_RELATIONAL_TRANSFORMER:
            return None
        
        try:
            # Convert input to RelationalDatabase format
            if tables:
                # Use first table if multiple
                table = tables[0]
                columns_data = []
                for i, header in enumerate(table.get("headers", [])):
                    columns_data.append({
                        "name": header,
                        "type": "string",  # Default type
                    })
                
                # Create RelationalTableSpec
                table_spec = RelationalTableSpec(
                    table_name=table_name or "extracted_table",
                    columns=columns_data,
                )
            elif table_name and columns:
                # Create from provided columns
                table_spec = RelationalTableSpec(
                    table_name=table_name,
                    columns=columns,
                )
            else:
                return None
            
            # Create database
            db = RelationalDatabase(tables=[table_spec])
            
            # Note: Full embedding generation would require a loaded model
            # For now, return a placeholder structure
            # In production, this would use the actual RelationalTransformer model
            return {
                "table_name": table_spec.table_name,
                "column_count": len(table_spec.columns),
                "method": "relational_transformer",
            }
        except Exception as e:
            print(f"Relational embedding generation failed: {e}", file=sys.stderr)
            return None
    
    def _generate_semantic_embedding(
        self,
        text: str = None,
        table_name: str = None,
        columns: List[Dict] = None,
    ) -> Optional[List[float]]:
        """Generate semantic embedding using sap-rpt-1-oss."""
        if not HAS_SAP_RPT:
            return None
        
        try:
            # Use sap-rpt-1-oss tokenizer for semantic embeddings
            # This would use the ZMQ server if available
            tokenizer = Tokenizer()
            
            # Prepare text for embedding
            if text:
                input_text = text
            elif table_name and columns:
                # Create table description
                col_names = [col.get("name", "") for col in columns]
                input_text = f"Table: {table_name}, Columns: {', '.join(col_names)}"
            else:
                return None
            
            # Generate embedding using sap-rpt-1-oss tokenizer
            # Note: Full implementation would use the ZMQ server or direct tokenizer
            # For now, return metadata. In production, this would:
            # 1. Use the tokenizer to generate embeddings
            # 2. Return the actual embedding vector
            
            # The actual embedding would be generated by calling embed_sap_rpt.py
            return {
                "text": input_text,
                "method": "sap-rpt-1-oss",
                "note": "Full embedding requires ZMQ server - use embed_sap_rpt.py for actual embeddings",
            }
        except Exception as e:
            print(f"Semantic embedding generation failed: {e}", file=sys.stderr)
            return None
    
    def classify_and_analyze(
        self,
        table_name: str,
        columns: List[Dict],
        text: str = None,
        training_data_path: str = None,
    ) -> Dict[str, Any]:
        """Classify and analyze using sap-rpt-1-oss multi-task learning."""
        if not HAS_SAP_RPT:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "quality_score": 0.0,
            }
        
        try:
            # Use the advanced sap-rpt script (Phase 5)
            import subprocess
            import json as json_lib
            
            columns_json = json_lib.dumps(columns)
            cmd = [
                "python3",
                str(SCRIPT_DIR / "sap_rpt_advanced.py"),
                "--table-name", table_name,
                "--columns", columns_json,
                "--context", text or "",
            ]
            
            if training_data_path:
                cmd.extend(["--training-data", training_data_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json_lib.loads(result.stdout)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "quality_score": 0.0,
                    "error": result.stderr,
                }
        except Exception as e:
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "quality_score": 0.0,
                "error": str(e),
            }


def main():
    """Main entry point for unified multi-modal extraction."""
    parser = argparse.ArgumentParser(
        description="Unified multi-modal extraction (Phase 6)"
    )
    parser.add_argument("--mode", required=True, choices=[
        "ocr", "embed", "classify", "unified"
    ])
    parser.add_argument("--image-path", help="Path to image file")
    parser.add_argument("--image-base64", help="Base64-encoded image")
    parser.add_argument("--table-name", help="Table name")
    parser.add_argument("--columns", help="JSON array of column definitions")
    parser.add_argument("--text", help="Text content")
    parser.add_argument("--training-data", help="Path to training data")
    parser.add_argument("--prompt", help="OCR prompt", default=None)
    
    args = parser.parse_args()
    
    extractor = UnifiedMultiModalExtractor()
    extractor.initialize()
    
    result = {}
    
    if args.mode == "ocr":
        if args.image_path:
            result = extractor.extract_from_image(args.image_path, args.prompt)
        elif args.image_base64:
            result = extractor.extract_from_image_base64(args.image_base64, args.prompt)
        else:
            result = {"error": "Image path or base64 required for OCR mode"}
    
    elif args.mode == "embed":
        columns = []
        if args.columns:
            columns = json.loads(args.columns)
        
        result = extractor.generate_unified_embeddings(
            text=args.text,
            image_path=args.image_path,
            table_name=args.table_name,
            columns=columns,
        )
    
    elif args.mode == "classify":
        columns = []
        if args.columns:
            try:
                columns = json.loads(args.columns)
            except:
                columns = []
        
        if not args.table_name:
            result = {"error": "Table name required for classification"}
        else:
            result = extractor.classify_and_analyze(
                args.table_name,
                columns,
                args.text,
                args.training_data,
            )
    
    elif args.mode == "unified":
        # Full pipeline: OCR → Embed → Classify
        columns = []
        if args.columns:
            try:
                columns = json.loads(args.columns)
            except:
                columns = []
        
        # Step 1: OCR if image provided
        ocr_result = {}
        if args.image_path:
            ocr_result = extractor.extract_from_image(args.image_path, args.prompt)
        elif args.image_base64:
            ocr_result = extractor.extract_from_image_base64(args.image_base64, args.prompt)
        
        # Step 2: Generate embeddings
        text = ocr_result.get("text", args.text)
        tables = ocr_result.get("tables", [])
        
        embed_result = extractor.generate_unified_embeddings(
            text=text,
            tables=tables,
            table_name=args.table_name,
            columns=columns,
        )
        
        # Step 3: Classify if table info available
        classify_result = {}
        if args.table_name and columns:
            classify_result = extractor.classify_and_analyze(
                args.table_name,
                columns,
                text,
                args.training_data,
            )
        
        result = {
            "ocr": ocr_result,
            "embeddings": embed_result,
            "classification": classify_result,
            "method": "unified-multimodal",
        }
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

