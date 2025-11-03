import argparse
import requests
from pathlib import Path

import opacus
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_tensor = torch.tensor(
            [item["Financed_Emission"], item["Total_Revenue_YTD"], item["Amount"]],
            dtype=torch.float32,
        )
        target_tensor = torch.tensor(item["Financed_Emission"], dtype=torch.float32)
        return input_tensor, target_tensor


def fetch_data_from_api(api_url: str) -> list:
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()["data"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Differentially private training helper")
    parser.add_argument("--model-path", required=True, help="Starting model checkpoint (HuggingFace path or dir)")
    parser.add_argument("--model-out", required=True, help="Directory to save the trained model")
    parser.add_argument("--data-api-url", required=True, help="API endpoint returning training data JSON")
    parser.add_argument("--epsilon", type=float, required=True, help="Target privacy epsilon")
    parser.add_argument("--delta", type=float, required=True, help="Target privacy delta")
    parser.add_argument("--resume-from", help="Resume training from an existing output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, help="Clip gradient norm before optimizer step")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision when CUDA is available")
    parser.add_argument("--loss-scale", type=float, help="Initial loss scale when AMP is enabled")
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0, help="Max grad norm for privacy engine")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    load_path = args.resume_from if args.resume_from else args.model_path
    model = AutoModelForCausalLM.from_pretrained(load_path)
    model.to(device)
    model.train()

    data = fetch_data_from_api(args.data_api_url)
    dataset = CustomDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    privacy_engine = opacus.PrivacyEngine(
        model,
        optimizer,
        max_grad_norm=args.dp_max_grad_norm,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
    )
    privacy_engine.attach(optimizer)

    scaler = None
    if use_amp:
        scaler_kwargs = {}
        if args.loss_scale is not None:
            scaler_kwargs["init_scale"] = args.loss_scale
        scaler = torch.cuda.amp.GradScaler(enabled=True, **scaler_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for epoch in range(args.epochs):
        for step, (features, target) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(features)
                loss = torch.nn.functional.mse_loss(outputs.logits.squeeze(-1), target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

    output_dir = Path(args.model_out)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
