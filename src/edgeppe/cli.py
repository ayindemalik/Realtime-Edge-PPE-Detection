"""Command line: explore, export, benchmark, make-video, monitor, gif."""

from pathlib import Path

import typer

from edgeppe.config import DATA_YAML, IMGSZ, MODELS, RESULTS, images_in

app = typer.Typer(help="Real-time PPE detection on the edge", no_args_is_help=True)


@app.command() # 
def explore():
    """Download the dataset (first time) and count boxes per class and split."""
    from edgeppe.explore import label_counts
    table = label_counts()
    typer.echo(table.to_string())
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "class_counts.csv")


@app.command()
def export(weights: Path = MODELS / "ppe_yolo11n.pt", imgsz: int = IMGSZ):
    """Export the model to ONNX, OpenVINO FP32 and OpenVINO INT8."""
    from edgeppe.export import export_all
    for path in export_all(weights, imgsz):
        typer.echo(f"Exported: {path}")


@app.command()
def benchmark(images: int = 100, imgsz: int = IMGSZ, accuracy: bool = True,
              weights: Path = MODELS / "ppe_yolo11n.pt"):
    """Latency, FPS, size and mAP for every exported format."""
    from edgeppe.benchmark import run
    stem = weights.stem
    candidates = [weights, weights.with_suffix(".onnx"),
                  MODELS / f"{stem}_openvino_model", MODELS / f"{stem}_int8_openvino_model"]
    found = [p for p in candidates if p.exists()]
    table = run(found, images, imgsz, DATA_YAML, accuracy)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "benchmark.csv", index=False)
    typer.echo("\n" + table.to_markdown(index=False))


@app.command("make-video")
def make_video(out: Path = Path("data/videos/test_split.mp4"), seconds_per_image: float = 1.5, fps: int = 15):
    """Stitch the dataset's test images into a video (a stand-in when you have no site footage)."""
    import cv2
    from ultralytics.data.utils import check_det_dataset

    images = images_in(check_det_dataset(DATA_YAML)["test"])
    out.parent.mkdir(parents=True, exist_ok=True)
    size = (1280, 720)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for img_path in images:
        img = cv2.imread(str(img_path))
        scale = min(size[0] / img.shape[1], size[1] / img.shape[0])
        img = cv2.resize(img, None, fx=scale, fy=scale)
        canvas = cv2.copyMakeBorder(img, 0, size[1] - img.shape[0], 0, size[0] - img.shape[1],
                                    cv2.BORDER_CONSTANT)
        for _ in range(int(seconds_per_image * fps)):
            writer.write(canvas)
    writer.release()
    typer.echo(f"Wrote {out} from {len(images)} images")


@app.command()
def monitor(source: str, weights: Path = MODELS / "ppe_yolo11n_openvino_model",
            conf: float = 0.35, min_frames: int = 5, show: bool = False, max_frames: int = 0):
    """Run the site monitor on a video file, or a webcam ("0")."""
    from edgeppe.monitor import run
    summary = run(weights, source, conf=conf, min_frames=min_frames, show=show,
                  max_frames=max_frames or None)
    typer.echo(summary)


@app.command()
def gif(video: Path = RESULTS / "monitor" / "annotated.mp4", out: Path = Path("assets/demo.gif"),
        start: float = 0, seconds: float = 8, width: int = 640):
    """Cut a short GIF from the annotated video, for the README."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames, step = [], max(1, round(fps / 8))        # keep about 8 frames per second
    for i in range(int(seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (width, int(h * width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / 8), loop=0)
    typer.echo(f"Wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()