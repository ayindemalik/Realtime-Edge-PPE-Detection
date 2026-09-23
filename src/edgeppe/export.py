"""Turn the trained PyTorch model into faster formats for CPU and edge devices.

  .pt                  PyTorch, FP32         the training format (the baseline)
  .onnx                ONNX Runtime, FP32    a portable format most runtimes read
  _openvino_model/     OpenVINO, FP32        Intel's runtime, tuned for Intel CPUs
  _int8_openvino_model OpenVINO, INT8        weights and maths in 8-bit integers
"""

from pathlib import Path

from ultralytics import YOLO

from edgeppe.config import DATA_YAML, IMGSZ


def export_all(weights: Path, imgsz: int = IMGSZ, data: str = DATA_YAML) -> list[str]:
    model = YOLO(weights)
    return [
        model.export(format="onnx", imgsz=imgsz, simplify=True),
        model.export(format="openvino", imgsz=imgsz),
        # INT8 needs real images to measure the range of values in each layer
        # ("calibration"); Ultralytics takes them from the dataset's val split.
        model.export(format="openvino", imgsz=imgsz, int8=True, data=data),
    ]
