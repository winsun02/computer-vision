"""Prepare the Construction Site Safety Image Dataset for this project.

Selected dataset:
    Construction Site Safety Image Dataset Roboflow on Kaggle
    https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow

The Kaggle export is already close to YOLO format, but its class names use
variants such as `Hardhat`, `NO-Hardhat`, and `Safety Vest`. This script copies
(or symlinks) images/labels into the project layout and rewrites label class IDs
into the canonical names used by the rest of the codebase.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # Keep --help usable in minimal environments.
    yaml = None

KAGGLE_DATASET = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
CANONICAL_NAMES = [
    "helmet",
    "mask",
    "no_helmet",
    "no_mask",
    "no_vest",
    "person",
    "safety_cone",
    "safety_vest",
    "machinery",
    "vehicle",
]

# Known class map from the selected dataset. The script also reads data.yaml
# from the downloaded dataset when available, but this map makes preparation
# deterministic even when the source YAML is missing.
DEFAULT_SOURCE_NAMES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}
ALIASES_TO_CANONICAL = {
    "hardhat": "helmet",
    "hard_hat": "helmet",
    "helmet": "helmet",
    "mask": "mask",
    "no_hardhat": "no_helmet",
    "no_helmet": "no_helmet",
    "no_mask": "no_mask",
    "no_safety_vest": "no_vest",
    "no_vest": "no_vest",
    "person": "person",
    "safety_cone": "safety_cone",
    "cone": "safety_cone",
    "safety_vest": "safety_vest",
    "vest": "safety_vest",
    "machinery": "machinery",
    "machine": "machinery",
    "vehicle": "vehicle",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitPaths:
    """Input image/label folders for one split."""

    split: str
    images: Path
    labels: Path


def normalize_name(name: str) -> str:
    """Normalize class names from data.yaml into snake_case."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Kaggle Construction Site Safety dataset")
    parser.add_argument(
        "--source",
        type=Path,
        help="Downloaded Kaggle zip file or extracted dataset folder. Optional when --download is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/construction_site_safety"),
        help="Output dataset folder with images/train, labels/train, ...",
    )
    parser.add_argument(
        "--yaml-output",
        type=Path,
        default=Path("data/data.yaml"),
        help="Where to write the YOLO data.yaml for this prepared dataset",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Try downloading from Kaggle using the kaggle CLI. Requires Kaggle credentials.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking them. Labels are always rewritten/copied.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    return parser.parse_args()


def run_kaggle_download(destination: Path) -> Path:
    """Download the dataset with the Kaggle CLI and return the zip path."""
    destination.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(destination)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is not installed. Run `pip install kaggle` or pass --source manually.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Check ~/.kaggle/kaggle.json credentials or download the dataset manually."
        ) from exc
    zip_files = sorted(destination.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not zip_files:
        raise FileNotFoundError(f"Kaggle CLI completed but no zip file was found in {destination}")
    return zip_files[0]


def extract_if_needed(source: Path, work_dir: Path) -> Path:
    """Return an extracted dataset root for a zip file or a folder source."""
    source = source.expanduser().resolve()
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_dir = work_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extract_dir)
        return extract_dir
    raise FileNotFoundError(f"Source must be an extracted folder or .zip file: {source}")


def read_source_names(root: Path) -> dict[int, str]:
    """Read class names from source YAML when present, otherwise use known map."""
    if yaml is None:
        return dict(DEFAULT_SOURCE_NAMES)
    for yaml_path in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            content = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except UnicodeDecodeError:
            continue
        names = content.get("names")
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
    return dict(DEFAULT_SOURCE_NAMES)


def build_class_id_map(source_names: dict[int, str]) -> dict[int, int]:
    """Map source class IDs to canonical class IDs."""
    class_id_map: dict[int, int] = {}
    canonical_to_id = {name: index for index, name in enumerate(CANONICAL_NAMES)}
    for source_id, raw_name in source_names.items():
        canonical = ALIASES_TO_CANONICAL.get(normalize_name(raw_name))
        if canonical is None:
            print(f"Warning: source class `{raw_name}` (id={source_id}) is not used by this project and will be skipped.")
            continue
        class_id_map[source_id] = canonical_to_id[canonical]
    return class_id_map


def find_split_paths(root: Path) -> list[SplitPaths]:
    """Find train/val/test folders in common YOLO export layouts."""
    candidates = []
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for folder in root.rglob("*"):
        if not folder.is_dir() or folder.name.lower() not in split_aliases:
            continue
        images = folder / "images"
        labels = folder / "labels"
        if images.is_dir() and labels.is_dir():
            candidates.append(SplitPaths(split_aliases[folder.name.lower()], images, labels))

    # Also support images/train + labels/train layout if someone already rearranged it.
    images_root = root / "images"
    labels_root = root / "labels"
    if images_root.is_dir() and labels_root.is_dir():
        for source_split, target_split in split_aliases.items():
            images = images_root / source_split
            labels = labels_root / source_split
            if images.is_dir() and labels.is_dir():
                candidates.append(SplitPaths(target_split, images, labels))

    unique: dict[str, SplitPaths] = {}
    for item in candidates:
        unique.setdefault(item.split, item)
    if not unique:
        raise FileNotFoundError("Could not find YOLO split folders such as train/images and train/labels in the source.")
    return [unique[key] for key in ("train", "val", "test") if key in unique]


def iter_images(images_dir: Path) -> Iterable[Path]:
    """Yield image files from a folder."""
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def link_or_copy_image(source: Path, destination: Path, copy: bool) -> None:
    """Copy or symlink one image."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def remap_label(source_label: Path, destination_label: Path, class_id_map: dict[int, int]) -> tuple[int, int]:
    """Rewrite a YOLO label file with canonical class IDs.

    Returns:
        kept_objects, skipped_objects
    """
    destination_label.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    skipped = 0
    if source_label.is_file():
        for line in source_label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                skipped += 1
                continue
            try:
                source_id = int(float(parts[0]))
            except ValueError:
                skipped += 1
                continue
            target_id = class_id_map.get(source_id)
            if target_id is None:
                skipped += 1
                continue
            kept.append(" ".join([str(target_id), *parts[1:5]]))
    destination_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return len(kept), skipped


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    """Write the project's YOLO data.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(dataset_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CANONICAL_NAMES)},
    }
    if yaml is not None:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CANONICAL_NAMES))
        text = (
            f"path: {dataset_root.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"names:\n{names}\n"
        )
    path.write_text(text, encoding="utf-8")


def prepare_dataset(source_root: Path, output_root: Path, yaml_output: Path, copy_images: bool, force: bool) -> None:
    """Prepare the selected dataset for training/evaluation."""
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}. Use --force to overwrite.")

    source_names = read_source_names(source_root)
    class_id_map = build_class_id_map(source_names)
    splits = find_split_paths(source_root)
    summary: dict[str, dict[str, int]] = {}

    for split in splits:
        split_summary = {"images": 0, "objects": 0, "skipped_objects": 0}
        for image_path in iter_images(split.images):
            target_image = output_root / "images" / split.split / image_path.name
            target_label = output_root / "labels" / split.split / f"{image_path.stem}.txt"
            source_label = split.labels / f"{image_path.stem}.txt"
            link_or_copy_image(image_path, target_image, copy_images)
            kept, skipped = remap_label(source_label, target_label, class_id_map)
            split_summary["images"] += 1
            split_summary["objects"] += kept
            split_summary["skipped_objects"] += skipped
        summary[split.split] = split_summary

    write_data_yaml(yaml_output, output_root)
    metadata = {
        "source": KAGGLE_DATASET,
        "canonical_names": CANONICAL_NAMES,
        "source_names": source_names,
        "summary": summary,
    }
    (output_root / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Dataset prepared successfully.")
    print(f"Output dataset : {output_root}")
    print(f"YOLO data.yaml : {yaml_output.expanduser().resolve()}")
    for split, values in summary.items():
        print(f"- {split}: {values['images']} images, {values['objects']} objects, {values['skipped_objects']} skipped objects")


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="css_dataset_") as tmp:
        work_dir = Path(tmp)
        source = args.source
        if args.download:
            source = run_kaggle_download(work_dir / "download")
        if source is None:
            raise ValueError("Provide --source <zip_or_folder> or use --download with configured Kaggle credentials.")
        source_root = extract_if_needed(source, work_dir)
        prepare_dataset(source_root, args.output, args.yaml_output, args.copy, args.force)


if __name__ == "__main__":
    main()
