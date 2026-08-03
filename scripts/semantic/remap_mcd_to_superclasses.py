#!/usr/bin/env python3
"""Collapse the 29 official MCD labels into 11 map-oriented superclasses."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SUPERCLASSES = {
    0: "unknown",
    1: "ground",
    2: "building-structure",
    3: "vegetation",
    4: "barrier-fence",
    5: "pole-sign",
    6: "vehicle",
    7: "pedestrian",
    8: "bike",
    9: "furniture-object",
    10: "cliff",
    11: "other-noise",
}

# Input IDs are official MCD IDs + 1 because 0 is reserved for unknown.
GROUPS = {
    1: [7, 11, 14, 17, 19, 20],       # curb, lane, parking, road, sidewalk, stairs
    2: [3, 18, 21],                    # building, shelter, structure-other
    3: [25, 26],                       # treetrunk, vegetation
    4: [1, 8],                         # barrier, fence
    5: [9, 10, 16, 23],                # hydrant, infosign, pole, traffic-sign
    6: [27, 28, 29],                   # dynamic/other/static vehicle
    7: [15],                           # pedestrian
    8: [2],                            # bike
    9: [4, 6, 22, 24],                 # chair, container, cone, trashbin
    10: [5],                           # cliff
    11: [12, 13],                      # noise, other
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    (args.output / "labels").mkdir(parents=True)
    (args.output / "confidence").mkdir()

    lookup = np.zeros(256, dtype=np.uint8)
    for superclass, source_labels in GROUPS.items():
        lookup[source_labels] = superclass
    counts = np.zeros(len(SUPERCLASSES), dtype=np.int64)
    label_paths = sorted((args.source / "labels").glob("*.png"))
    for index, label_path in enumerate(label_paths, start=1):
        labels = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        confidence = cv2.imread(
            str(args.source / "confidence" / label_path.name),
            cv2.IMREAD_GRAYSCALE,
        )
        if labels is None or confidence is None:
            raise RuntimeError(f"cannot read {label_path.name}")
        remapped = lookup[labels]
        cv2.imwrite(str(args.output / "labels" / label_path.name), remapped)
        cv2.imwrite(str(args.output / "confidence" / label_path.name), confidence)
        counts += np.bincount(remapped.ravel(), minlength=len(SUPERCLASSES))
        if index % 500 == 0 or index == len(label_paths):
            print(f"processed={index}/{len(label_paths)}")

    source_metadata = json.loads(
        (args.source / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = dict(source_metadata)
    metadata.update(
        {
            "source": "MCD ntu_day_02 projected LiDAR annotations, 11 superclasses",
            "class_count_with_unknown": len(SUPERCLASSES),
            "classes": {str(key): value for key, value in SUPERCLASSES.items()},
            "class_pixel_counts": counts.tolist(),
            "remap_from_shifted_mcd_ids": {
                str(key): value for key, value in GROUPS.items()
            },
        }
    )
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
