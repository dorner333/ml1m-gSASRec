import mlflow.pyfunc
import torch
import yaml
from pathlib import Path
from gsasrec.dataset_utils import get_num_items
from gsasrec.gsasrec import GSASRec


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        base = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((base / "configs" / "model.yaml").read_text())
        num_items = get_num_items(cfg["dataset_name"])
        self.model = GSASRec(
            num_items=num_items,
            sequence_length=cfg["sequence_length"],
            embedding_dim=cfg["embedding_dim"],
            num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"],
            dropout_rate=cfg["dropout_rate"],
            reuse_item_embeddings=cfg["reuse_item_embeddings"],
        )
        sd = torch.load(
            Path(context.artifacts["model_path"]) / "model.pt", map_location="cpu"
        )
        self.model.load_state_dict(sd)
        self.model.eval()
        self.limit = cfg["recommendation_limit"]

    def predict(self, context, model_input):
        seqs = model_input["instances"].tolist()
        seq_tensor = torch.tensor(seqs, dtype=torch.long)
        indices, _ = self.model.get_predictions(seq_tensor, limit=self.limit)
        return indices.numpy().tolist()
