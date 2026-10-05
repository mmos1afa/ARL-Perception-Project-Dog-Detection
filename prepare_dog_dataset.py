from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_DOG_CLASS_BY_DATASET = {
    "dog.yolov11": 1,
    "yolo-dog.yolov11": 0,
}


def _read_label_rows(label_path: Path) -> List[Tuple[int, List[str]]]:
    rows: List[Tuple[int, List[str]]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}, line {line_number}")
        try:
            class_id = int(fields[0])
            [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(
                f"Invalid YOLO values in {label_path}, line {line_number}"
            ) from error
        rows.append((class_id, fields[1:]))
    return rows


def prepare_dog_only_data(
    raw_dir: Path,
    output_dir: Path,
    dog_class_by_dataset: Dict[str, int] | None = None,
    target_class_id: int = 0,
) -> Dict[str, int]:
    """
    Copy only dog-only images into a clean staging directory.

    The original raw data is never modified. Images containing a non-dog class
    are ignored, and retained dog annotations are normalized to class 0.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    class_map = dog_class_by_dataset or DEFAULT_DOG_CLASS_BY_DATASET
    dataset_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not dataset_dirs:
        raise ValueError(f"No dataset folders found in {raw_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    skipped_images = 0
    skipped_missing_labels = 0
    copied_instances = 0

    for dataset_dir in dataset_dirs:
        dog_class_id = class_map.get(dataset_dir.name, 0)
        image_paths = sorted(
            path
            for path in dataset_dir.rglob("images/*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

        for image_path in image_paths:
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            if not label_path.is_file():
                skipped_missing_labels += 1
                continue

            rows = _read_label_rows(label_path)
            if any(class_id != dog_class_id for class_id, _ in rows):
                skipped_images += 1
                continue

            relative_image_path = image_path.relative_to(dataset_dir)
            destination_image = output_dir / dataset_dir.name / relative_image_path
            destination_label = (
                destination_image.parent.parent / "labels" / f"{image_path.stem}.txt"
            )
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image_path, destination_image)
            normalized_lines = [
                " ".join([str(target_class_id), *coordinates])
                for _, coordinates in rows
            ]
            destination_label.write_text(
                "\n".join(normalized_lines) + ("\n" if normalized_lines else ""),
                encoding="utf-8",
            )
            copied_images += 1
            copied_instances += len(rows)

    if copied_images == 0:
        raise ValueError("No dog-only image-label pairs were found")

    return {
        "copied_images": copied_images,
        "skipped_images": skipped_images,
        "skipped_missing_labels": skipped_missing_labels,
        "dog_instances": copied_instances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a dog-only YOLO staging dataset without changing raw_data."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("filtered_raw_data"))
    parser.add_argument(
        "--dog-class",
        type=int,
        default=None,
        help="Override the dog class ID for all source datasets.",
    )
    args = parser.parse_args()

    class_map = None
    if args.dog_class is not None:
        class_map = {
            dataset_dir.name: args.dog_class
            for dataset_dir in args.raw_dir.iterdir()
            if dataset_dir.is_dir()
        }

    summary = prepare_dog_only_data(args.raw_dir, args.output_dir, class_map)
    print(f"Dog-only dataset written to {args.output_dir.resolve()}")
    print(f"Copied images: {summary['copied_images']}")
    print(f"Skipped images containing non-dog labels: {summary['skipped_images']}")
    print(f"Skipped images without labels: {summary['skipped_missing_labels']}")
    print(f"Dog instances: {summary['dog_instances']}")


if __name__ == "__main__":
    main()
