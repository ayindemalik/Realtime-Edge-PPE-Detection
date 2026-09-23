"""The real-time site monitor: video in, annotated video + violation log out.

For every frame: detect PPE, follow each object from frame to frame
(tracking), and raise a violation only when the same object has been seen
without PPE for several frames. One bad frame must not trigger an alarm.
"""

import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from edgeppe.config import IMGSZ, RESULTS, VIOLATIONS, colour


def draw(frame, box, label: str, name: str) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour(name), 2)
    cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour(name), 2)


def panel(frame, fps: float, counts: Counter, n_events: int) -> None:
    lines = [f"FPS {fps:5.1f}", f"People {counts['Person']}",
             f"Violations now {sum(counts[v] for v in VIOLATIONS)}", f"Events logged {n_events}"]
    cv2.rectangle(frame, (0, 0), (250, 22 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def run(weights: Path, source: str, out_dir: Path = RESULTS / "monitor", conf: float = 0.35,
        min_frames: int = 5, imgsz: int = IMGSZ, show: bool = False, max_frames: int | None = None) -> dict:
    model = YOLO(weights, task="detect")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)   # "0" = webcam
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    log = open(out_dir / "events.csv", "w", newline="", encoding="utf-8")
    events = csv.writer(log)
    events.writerow(["time", "frame", "track_id", "violation", "confidence", "snapshot"])

    seen = defaultdict(int)     # track id -> frames seen as a violation
    logged = set()              # track ids already reported (one alert per object)
    fps, n, frame_times = 0.0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        start = time.perf_counter()
        result = model.track(frame, imgsz=imgsz, conf=conf, persist=True,
                             tracker="bytetrack.yaml", verbose=False)[0]
        counts = Counter()
        for box in result.boxes:
            name = result.names[int(box.cls)]
            counts[name] += 1
            tid = int(box.id) if box.id is not None else -1
            draw(frame, box.xyxy[0].tolist(), f"{name} {float(box.conf):.2f}", name)
            if name in VIOLATIONS and tid >= 0:
                seen[tid] += 1
                if seen[tid] == min_frames and tid not in logged:
                    logged.add(tid)
                    snap = out_dir / "snapshots" / f"frame{n:06d}_id{tid}_{name}.jpg"
                    cv2.imwrite(str(snap), frame)
                    events.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), n, tid, name,
                                     round(float(box.conf), 3), snap.name])
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed * 1000)
        fps = 0.9 * fps + 0.1 * (1 / elapsed) if fps else 1 / elapsed   # smoothed
        panel(frame, fps, counts, len(logged))
        writer.write(frame)
        if show:
            cv2.imshow("PPE monitor (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1

    cap.release()
    writer.release()
    log.close()
    cv2.destroyAllWindows()
    return {"frames": n, "events": len(logged),
            "mean_ms_per_frame": round(sum(frame_times) / max(n, 1), 1),
            "output": str(out_dir / "annotated.mp4")}
