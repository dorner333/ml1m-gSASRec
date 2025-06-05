from dvc.repo import Repo

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

from gsasrec.lightning_module import GSASRecLightning


def pull_data():
    repo = Repo(".")
    repo.pull()


@hydra.main(config_path="../configs", config_name="model", version_base=None)
def main(cfg: DictConfig):
    mlflow_logger = MLFlowLogger(
        experiment_name=f"gsasrec_{cfg.dataset_name}", tracking_uri=cfg.mlflow_uri
    )

    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = {}

    def _flatten(d, parent_key=""):
        for k, v in d.items():
            new_key = parent_key + "." + k if parent_key else k
            if isinstance(v, dict):
                _flatten(v, new_key)
            else:
                flat_cfg[new_key] = v

    _flatten(cfg_dict)
    # Добавляем git_commit в гиперпараметры
    flat_cfg["git_commit"] = commit_hash
    mlflow_logger.log_hyperparams(flat_cfg)

    model = GSASRecLightning(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.max_batches_per_epoch,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        log_every_n_steps=1,
    )
    trainer.fit(model)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    epochs = list(range(len(model.train_losses)))

    plt.figure()
    plt.plot(epochs, model.train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "train_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, model.val_ndcg10)
    plt.xlabel("Epoch")
    plt.ylabel("Validation nDCG@10")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "val_nDCG10.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, model.val_r10)
    plt.xlabel("Epoch")
    plt.ylabel("Validation R@10")
    plt.tight_layout()
    plt.savefig(str(plots_dir / "val_R10.png"))
    plt.close()


if __name__ == "__main__":
    pull_data()
    main()
