import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    config = {}
    names = {}
    in_names = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.startswith("names:"):
            in_names = True
            continue

        if in_names and raw_line.startswith("  "):
            key, value = line.strip().split(":", 1)
            names[int(key.strip())] = value.strip().strip("'\"")
            continue

        in_names = False
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip("'\"")
            if key.strip() == "nc":
                value = int(value)
            config[key.strip()] = value

    if names:
        config["names"] = names

    return config


def resolve_split_dir(config_path: Path, config: dict, split: str) -> Path:
    dataset_root = Path(config.get("path", config_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (config_path.parent / dataset_root).resolve()
    return (dataset_root / config[split]).resolve()


def label_path_for(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return (label_dir / relative).with_suffix(".txt")


def check_split(config_path: Path, config: dict, split: str) -> dict:
    image_dir = resolve_split_dir(config_path, config, split)
    label_dir = Path(str(image_dir).replace(f"{Path('images')}", f"{Path('labels')}"))

    images = []
    if image_dir.exists():
        images = [
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    missing_labels = [
        str(label_path_for(path, image_dir, label_dir))
        for path in images
        if not label_path_for(path, image_dir, label_dir).exists()
    ]

    return {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "image_count": len(images),
        "missing_label_count": len(missing_labels),
        "missing_labels": missing_labels[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check YOLO dataset structure.")
    parser.add_argument("--data", default="DataSet/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    config_path = Path(args.data).resolve()
    config = load_dataset_config(config_path)

    required = {"train", "val", "test", "nc", "names"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing keys in data.yaml: {sorted(missing)}")

    print("Dataset config:", config_path)
    print("Classes:", config["names"])

    for split in ("train", "val", "test"):
        report = check_split(config_path, config, split)
        print(
            f"{split}: {report['image_count']} images, "
            f"{report['missing_label_count']} missing labels"
        )
        if report["missing_labels"]:
            print("  Examples:")
            for label in report["missing_labels"]:
                print("   -", label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
