"""Names and settings used across the project. One place for every name."""

from pathlib import Path

DATA_YAML = "construction-ppe.yaml"   # built into Ultralytics; downloads itself (178 MB)
MODELS = Path("models")
RESULTS = Path("results")
IMGSZ = 640  # Image size for YOLOv8n (640x640) and ByteTrack (640x640)

# The 11 classes of the Construction-PPE dataset
CLASSES = ["helmet", "gloves", "vest", "boots", "goggles", "none",
           "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"]

# A detection of one of these classes is a safety violation
VIOLATIONS = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}

# Box colours in BGR (OpenCV order): red for violations, green for PPE, blue for people
RED, GREEN, BLUE = (40, 40, 220), (60, 170, 60), (200, 120, 40)


def images_in(folder: Path) -> list[Path]:
    """All pictures in a folder. The test split has both .jpg and .jpeg files."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def colour(name: str) -> tuple[int, int, int]:
    if name in VIOLATIONS:
        return RED
    return BLUE if name == "Person" else GREEN
