"""Speed AND accuracy of each model format, measured the same way.

A faster model is only useful if it is still accurate enough. So for every
format we measure both, on the same test images, on the same machine.
"""

import multiprocessing as mp
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from edgeppe.config import DATA_YAML, IMGSZ, images_in


def size_mb(path: Path) -> float:
    files = path.rglob("*") if path.is_dir() else [path]
    return sum(f.stat().st_size for f in files if f.is_file()) / 1e6


def latency_ms(model: YOLO, images: list[Path], imgsz: int, warmup: int = 10):
    """Two timings per image: the whole call, and the neural network alone."""
    for img in images[:warmup]:              # the first runs are slow: not counted
        model.predict(img, imgsz=imgsz, verbose=False)
    total, network = [], []
    for img in images:
        start = time.perf_counter()
        result = model.predict(img, imgsz=imgsz, verbose=False)[0]   # read + resize + infer + NMS
        total.append((time.perf_counter() - start) * 1000)
        network.append(result.speed["inference"])                     # the model only
    return total, network


def measure(path: Path, n_images: int, imgsz: int, data: str, accuracy: bool) -> dict:
    images = images_in(check_det_dataset(data)["test"])[:n_images]
    model = YOLO(path, task="detect")
    total, network = latency_ms(model, images, imgsz)
    row = {
        "model": path.name,
        "size_mb": round(size_mb(path), 1),
        "network_ms": round(statistics.median(network), 1),
        "p50_ms": round(statistics.median(total), 1),
        "p95_ms": round(statistics.quantiles(total, n=20)[18], 1),
        "fps": round(1000 / statistics.mean(total), 1),
    }
    if accuracy:
        m = model.val(data=data, split="test", imgsz=imgsz, batch=1,
                      device="cpu", plots=False, verbose=False)
        row["mAP50"] = round(float(m.box.map50), 3)
        row["mAP50_95"] = round(float(m.box.map), 3)
    return row


def run(model_paths: list[Path], n_images: int = 100, imgsz: int = IMGSZ, data: str = DATA_YAML,
        accuracy: bool = True) -> pd.DataFrame:
    rows = []
    for path in model_paths:
        # Each format runs in its OWN fresh process. In one shared process, the
        # thread pools of PyTorch and ONNX Runtime keep spinning after they finish
        # and steal CPU from the next runtime: OpenVINO looked 12x slower that way.
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
            row = pool.submit(measure, path, n_images, imgsz, data, accuracy).result()
        rows.append(row)
        print(row)
    return pd.DataFrame(rows)
