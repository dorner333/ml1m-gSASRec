import hydra
import torch
from omegaconf import DictConfig

from gsasrec.dataset_utils import get_test_dataloader
from gsasrec.eval_utils import evaluate
from gsasrec.utils import build_model, get_device


@hydra.main(config_path="../configs", config_name="model", version_base=None)
def main(cfg: DictConfig):
    device = get_device()
    model = build_model(cfg)
    model = model.to(device)
    state = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(state)

    test_loader = get_test_dataloader(
        cfg.dataset_name, batch_size=cfg.eval_batch_size, max_length=cfg.sequence_length
    )
    result = evaluate(
        model,
        test_loader,
        cfg.metrics,
        cfg.recommendation_limit,
        cfg.filter_rated,
        device=device,
    )
    print(result)


if __name__ == "__main__":
    main()
