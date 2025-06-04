import torch

from gsasrec.dataset_utils import get_num_items
from gsasrec.gsasrec import GSASRec


def build_model(config):
    num_items = get_num_items(config.dataset_name)
    model = GSASRec(
        num_items,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
    )
    return model


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device
