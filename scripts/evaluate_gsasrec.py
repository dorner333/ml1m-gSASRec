from dvc.repo import Repo

import hydra
from pathlib import Path

import torch
import onnxruntime
from omegaconf import DictConfig

from gsasrec.dataset_utils import get_test_dataloader
from gsasrec.utils import build_model, get_device
from gsasrec.eval_utils import evaluate_pt, evaluate_onnx


def pull_data():
    repo = Repo(".")
    repo.pull()


@hydra.main(config_path="../configs", config_name="model", version_base=None)
def main(cfg: DictConfig):
    checkpoint = Path(cfg.checkpoint_path)
    test_loader = get_test_dataloader(
        cfg.dataset_name, batch_size=cfg.eval_batch_size, max_length=cfg.sequence_length
    )

    if checkpoint.suffix == ".onnx":
        session = onnxruntime.InferenceSession(str(checkpoint))
        result = evaluate_onnx(
            session,
            test_loader,
            cfg.metrics,
            cfg.recommendation_limit,
            cfg.filter_rated,
        )
    else:
        device = get_device()
        model = build_model(cfg).to(device)
        state = torch.load(str(checkpoint), map_location=device)
        model.load_state_dict(state)
        result = evaluate_pt(
            model,
            test_loader,
            cfg.metrics,
            cfg.recommendation_limit,
            cfg.filter_rated,
            device,
        )

    print(result)


if __name__ == "__main__":
    pull_data()
    main()
