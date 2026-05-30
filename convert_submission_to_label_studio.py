#!/usr/bin/env python3
"""Convert Kaggle Severstal submission.csv RLE masks to Label Studio predictions.json.

Default output assumes Label Studio config:
  <Image name="image" value="$image"/>
  <BrushLabels name="label" toName="image"> ... </BrushLabels>

For local files, default image URL uses:
  /data/local-files/?d=test_images/<image>.jpg
Set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT to this project's data/ directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 256
CLASS_LABELS = {
    "1": "Patch",
    "2": "Crack / Crazing",
    "3": "Pitted Surface",
    "4": "Scratch",
}


def access_bit(data: bytes, num: int) -> int:
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bits2byte(arr_str: str, n: int = 8) -> list[int]:
    return [int(arr_str[i : i + n], 2) for i in range(0, len(arr_str), n)]


def base_rle_encode(inarray: np.ndarray):
    ia = np.asarray(inarray)
    n = len(ia)
    if n == 0:
        return None, None, None
    y = ia[1:] != ia[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return z, p, ia[i]


def encode_label_studio_rle(arr: np.ndarray, wordsize: int = 8, rle_sizes: list[int] | None = None) -> list[int]:
    """Encode flattened RGBA-like mask using Label Studio brush RLE format."""
    if rle_sizes is None:
        rle_sizes = [3, 4, 8, 16]

    num = len(arr)
    base_str = f"{num:032b}" + f"{wordsize - 1:05b}" + "".join(f"{x - 1:04b}" for x in rle_sizes)

    out_str = ""
    for length_reeks, _p, value in zip(*base_rle_encode(arr)):
        value = int(value)
        length_reeks = int(length_reeks)
        if length_reeks == 1:
            out_str += "0" + "00" + "000" + f"{value:08b}"
        elif length_reeks <= 8:
            out_str += "1" + "00" + f"{length_reeks - 1:03b}" + f"{value:08b}"
        elif length_reeks <= 16:
            out_str += "1" + "01" + f"{length_reeks - 1:04b}" + f"{value:08b}"
        elif length_reeks <= 256:
            out_str += "1" + "10" + f"{length_reeks - 1:08b}" + f"{value:08b}"
        else:
            length_temp = length_reeks
            while length_temp > 2**16:
                out_str += "1" + "11" + f"{2**16 - 1:016b}" + f"{value:08b}"
                length_temp -= 2**16
            out_str += "1" + "11" + f"{length_temp - 1:016b}" + f"{value:08b}"

    total_str = base_str + out_str
    pad = (8 - len(total_str) % 8) % 8
    total_str += pad * "0"
    return bits2byte(total_str)


def kaggle_rle_to_mask(rle: str, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH) -> np.ndarray:
    """Decode Severstal/Kaggle column-major RLE into 2D uint8 mask for Label Studio."""
    mask = np.zeros(height * width, dtype=np.uint8)
    rle = (rle or "").strip()
    if not rle:
        return mask.reshape((width, height)).T

    values = np.asarray([int(x) for x in rle.split()], dtype=np.int64)
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    for start, end in zip(starts, ends):
        mask[start:end] = 255
    return mask.reshape((width, height)).T


def mask_to_label_studio_rle(mask: np.ndarray) -> list[int]:
    assert mask.ndim == 2
    mask = mask.astype(np.uint8, copy=False)
    rgba_like = np.repeat(mask.ravel(), 4)
    return encode_label_studio_rle(rgba_like)


def stable_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:10]


def convert(
    submission_csv: Path,
    output_json: Path,
    image_prefix: str,
    model_version: str,
    from_name: str,
    to_name: str,
    data_key: str,
) -> dict[str, int]:
    grouped: dict[str, list[dict]] = {}
    image_order: list[str] = []
    total_rows = non_empty = 0

    with submission_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["ImageId_ClassId", "EncodedPixels"]:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

        for row in reader:
            total_rows += 1
            image_class = row["ImageId_ClassId"]
            image_id, class_id = image_class.rsplit("_", 1)
            if image_id not in grouped:
                grouped[image_id] = []
                image_order.append(image_id)

            encoded_pixels = (row["EncodedPixels"] or "").strip()
            if not encoded_pixels:
                continue

            non_empty += 1
            mask = kaggle_rle_to_mask(encoded_pixels)
            ls_rle = mask_to_label_studio_rle(mask)
            label = CLASS_LABELS.get(class_id, f"Class {class_id}")

            grouped[image_id].append(
                {
                    "id": stable_id(image_id, class_id),
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "brushlabels",
                    "origin": "prediction",
                    "original_width": IMAGE_WIDTH,
                    "original_height": IMAGE_HEIGHT,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": ls_rle,
                        "brushlabels": [label],
                    },
                }
            )

    tasks = []
    for image_id in image_order:
        image_url = f"{image_prefix}{image_id}"
        tasks.append(
            {
                "data": {data_key: image_url},
                "predictions": [
                    {
                        "model_version": model_version,
                        "score": 1.0,
                        "result": grouped[image_id],
                    }
                ],
            }
        )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, separators=(",", ":"))

    return {
        "csv_rows": total_rows,
        "tasks": len(tasks),
        "non_empty_masks": non_empty,
        "tasks_with_masks": sum(1 for results in grouped.values() if results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="submission.csv", type=Path)
    parser.add_argument("--out", default="predictions.json", type=Path)
    parser.add_argument("--image-prefix", default="/data/local-files/?d=test_images/")
    parser.add_argument("--model-version", default="efficientnet-b3-fpn-submission")
    parser.add_argument("--from-name", default="label")
    parser.add_argument("--to-name", default="image")
    parser.add_argument("--data-key", default="image")
    args = parser.parse_args()

    stats = convert(
        submission_csv=args.csv,
        output_json=args.out,
        image_prefix=args.image_prefix,
        model_version=args.model_version,
        from_name=args.from_name,
        to_name=args.to_name,
        data_key=args.data_key,
    )
    print(json.dumps({"output": str(args.out), **stats}, indent=2))


if __name__ == "__main__":
    main()
