import json
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import requests
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    DATASET_DIR = Path(cfg.dataset.data_dir)
    TRAIN_DIR = DATASET_DIR / "train"
    VAL_DIR = DATASET_DIR / "val"
    TEST_DIR = DATASET_DIR / "test"
    FILE_NAME = DATASET_DIR / "ml-1m.txt"

    if not Path(FILE_NAME).exists():
        request = requests.get(cfg.dataset.url, timeout=10, stream=True)
        with open(FILE_NAME, "wb") as fh:
            for chunk in request.iter_content(1024 * 1024):
                fh.write(chunk)

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    rng = np.random.RandomState(cfg.split.random_seed)
    val_users = rng.choice(
        cfg.split.num_all_users, cfg.split.val_user_count, replace=False
    )
    user_items = defaultdict(list)
    num_interactions = 0
    items = set()
    with open(FILE_NAME) as f:
        for line in f:
            if len(line.strip().split(" ")) != 2:
                continue
            user, item = line.strip().split(" ")
            user = int(user)
            item = int(item)
            items.add(item)
            num_interactions += 1
            user_items[user].append(item)
    dataset_stats = {
        "num_users": len(user_items),
        "num_items": len(items),
        "num_interactions": num_interactions,
    }

    print("Dataset stats: ", json.dumps(dataset_stats, indent=4))
    with open(DATASET_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)

    train_sequences = []

    val_input_sequences = []
    val_gt_actions = []

    test_input_sequences = []
    test_gt_actions = []

    for user in user_items:
        if user in val_users:
            train_input_sequence = user_items[user][:-3]
            train_sequences.append(train_input_sequence)

            val_input_sequence = user_items[user][:-2]
            val_gt_action = user_items[user][-2]
            val_input_sequences.append(val_input_sequence)
            val_gt_actions.append(val_gt_action)

            test_input_sequence = user_items[user][:-1]
            test_input_sequences.append(test_input_sequence)

            test_gt_action = user_items[user][-1]
            test_gt_actions.append(test_gt_action)
        else:
            train_input_sequence = user_items[user][:-2]
            train_sequences.append(train_input_sequence)

            test_input_sequence = user_items[user][:-1]
            test_input_sequences.append(test_input_sequence)
            test_gt_action = user_items[user][-1]
            test_gt_actions.append(test_gt_action)

    with open(TRAIN_DIR / "input.txt", "w") as f:
        for sequence in train_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")

    with open(VAL_DIR / "input.txt", "w") as f:
        for sequence in val_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")

    with open(VAL_DIR / "output.txt", "w") as f:
        for action in val_gt_actions:
            f.write(str(action) + "\n")

    with open(TEST_DIR / "input.txt", "w") as f:
        for sequence in test_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")
    with open(TEST_DIR / "output.txt", "w") as f:
        for action in test_gt_actions:
            f.write(str(action) + "\n")


if __name__ == "__main__":
    main()
