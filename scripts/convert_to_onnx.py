# File: scripts/convert_to_onnx.py

import fire
from pathlib import Path

import torch
from hydra import initialize, compose

from gsasrec.utils import build_model, get_device


class ExportModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids):
        model_out, _ = self.model(input_ids)
        seq_emb = model_out[:, -1, :]
        output_embeddings = self.model.get_output_embeddings().weight
        return seq_emb, output_embeddings


def convert(checkpoint_path: str, output_path: str = None):
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    else:
        output_path = Path(output_path)

    with initialize(config_path="../configs", job_name="convert", version_base=None):
        cfg = compose(config_name="model")

    device = get_device()
    base_model = build_model(cfg).to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    base_model.load_state_dict(state)
    base_model.eval()

    export_model = ExportModel(base_model).to(device)
    seq_len = cfg.sequence_length
    dummy_input = torch.zeros((1, seq_len), dtype=torch.long, device=device)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        input_names=["input_ids"],
        output_names=["seq_emb", "output_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "seq_emb": {0: "batch_size"},
            "output_embeddings": {0: "num_items"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(convert)
