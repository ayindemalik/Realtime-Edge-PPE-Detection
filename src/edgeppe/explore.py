"""Look at the data before training: how many boxes of each class, per split."""

from pathlib import Path

import pandas as pd
from ultralytics.data.utils import check_det_dataset

from edgeppe.config import CLASSES, DATA_YAML


def label_counts(data_yaml: str = DATA_YAML) -> pd.DataFrame: # 
    info = check_det_dataset(data_yaml)          # downloads the dataset the first time
    table = {}
    for split in ["train", "val", "test"]:
        label_dir = Path(str(info[split]).replace("images", "labels"))
        counts = [0] * len(CLASSES)
        files = list(label_dir.glob("*.txt"))
        for f in files:
            for line in f.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
        table[f"{split} ({len(files)} img)"] = counts
    return pd.DataFrame(table, index=CLASSES)
