#!/usr/bin/env python3
"""Project MCD labelled Livox scans into FAST-LIVO2 camera frames."""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np


MCD_CLASSES = [
    "barrier", "bike", "building", "chair", "cliff", "container", "curb",
    "fence", "hydrant", "infosign", "lanemarking", "noise", "other",
    "parkinglot", "pedestrian", "pole", "road", "shelter", "sidewalk",
    "stairs", "structure-other", "traffic-cone", "traffic-sign", "trashbin",
    "treetrunk", "vegetation", "vehicle-dynamic", "vehicle-other",
    "vehicle-static",
]


def lzf_decompress(data: bytes, expected_size: int) -> bytes:
    output = bytearray()
    cursor = 0
    while cursor < len(data):
        control = data[cursor]
        cursor += 1
        if control < 32:
            length = control + 1
            output.extend(data[cursor : cursor + length])
            cursor += length
            continue
        length = control >> 5
        reference_offset = (control & 0x1F) << 8
        if length == 7:
            length += data[cursor]
            cursor += 1
        reference_offset += data[cursor]
        cursor += 1
        length += 2
        reference = len(output) - reference_offset - 1
        if reference < 0:
            raise ValueError("invalid LZF back-reference")
        for _ in range(length):
            output.append(output[reference])
            reference += 1
    if len(output) != expected_size:
        raise ValueError(
            f"LZF size mismatch: decoded={len(output)} expected={expected_size}"
        )
    return bytes(output)


def read_labelled_pcd(path: Path):
    values = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: missing DATA line")
            parts = line.decode("ascii").strip().split()
            if parts:
                values[parts[0]] = parts[1:]
            if parts and parts[0] == "DATA":
                break
        if values["DATA"] != ["binary_compressed"]:
            raise ValueError(f"{path}: expected binary_compressed")
        compressed_size, uncompressed_size = struct.unpack("<II", stream.read(8))
        compressed = stream.read(compressed_size)
    raw = lzf_decompress(compressed, uncompressed_size)
    points = int(values["POINTS"][0])
    fields = values["FIELDS"]
    sizes = list(map(int, values["SIZE"]))
    types = values["TYPE"]
    offsets = {}
    offset = 0
    for field, size, field_type in zip(fields, sizes, types):
        offsets[field] = (offset, size, field_type)
        offset += size * points

    xyz = np.empty((points, 3), dtype=np.float32)
    for column, field in enumerate(("x", "y", "z")):
        field_offset, size, field_type = offsets[field]
        if size != 4 or field_type != "F":
            raise ValueError(f"{path}: unsupported {field} format")
        xyz[:, column] = np.frombuffer(
            raw, dtype="<f4", count=points, offset=field_offset
        )
    label_offset, label_size, label_type = offsets["label"]
    if label_size != 1 or label_type != "U":
        raise ValueError(f"{path}: unsupported label format")
    labels = np.frombuffer(
        raw, dtype=np.uint8, count=points, offset=label_offset
    ).copy()
    return xyz, labels


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.astype(np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + amount * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return (
        np.sin((1.0 - amount) * angle) * first
        + np.sin(amount * angle) * second
    ) / np.sin(angle)


def interpolate_pose(times, quaternions, translations, timestamp):
    right = int(np.searchsorted(times, timestamp))
    if right <= 0:
        return quaternions[0], translations[0]
    if right >= len(times):
        return quaternions[-1], translations[-1]
    left = right - 1
    amount = float((timestamp - times[left]) / (times[right] - times[left]))
    return (
        slerp(quaternions[left], quaternions[right], amount),
        translations[left] * (1.0 - amount) + translations[right] * amount,
    )


def read_lidar_times(database: Path, topic_name: str) -> np.ndarray:
    connection = sqlite3.connect(str(database))
    try:
        topic = connection.execute(
            "SELECT id FROM topics WHERE name = ?", (topic_name,)
        ).fetchone()
        if topic is None:
            raise RuntimeError(f"topic not found: {topic_name}")
        rows = connection.execute(
            "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic[0],),
        ).fetchall()
    finally:
        connection.close()
    return np.asarray([row[0] * 1e-9 for row in rows], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--ros2-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--fx", required=True, type=float)
    parser.add_argument("--fy", required=True, type=float)
    parser.add_argument("--cx", required=True, type=float)
    parser.add_argument("--cy", required=True, type=float)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--max-time-difference", type=float, default=0.061)
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    (args.output / "labels").mkdir(parents=True)
    (args.output / "confidence").mkdir()
    (args.output / "visualization").mkdir()

    manifest = [
        line.split()
        for line in (args.export_dir / "manifest.txt").read_text().splitlines()
        if line.strip()
    ]
    timestamp_rows = [
        line.split()
        for line in (args.export_dir / "timestamps.txt").read_text().splitlines()
        if line.strip()
    ]
    frame_times = np.asarray([float(row[1]) for row in timestamp_rows])
    quaternions = np.asarray(
        [[float(value) for value in row[2:6]] for row in manifest]
    )
    translations = np.asarray(
        [[float(value) for value in row[6:9]] for row in manifest]
    )
    if not (len(manifest) == len(timestamp_rows) == len(frame_times)):
        raise RuntimeError("manifest/timestamps length mismatch")

    lidar_times = read_lidar_times(args.ros2_db, args.topic)
    label_paths = sorted(args.label_dir.glob("cloud_*.pcd"))
    label_ids = np.asarray(
        [int(path.stem.split("_")[-1]) for path in label_paths], dtype=np.int64
    )
    if label_ids.min() < 0 or label_ids.max() >= len(lidar_times):
        raise RuntimeError("label IDs are incompatible with zero-based LiDAR indices")
    label_times = lidar_times[label_ids]

    rotation_cl = np.asarray(
        [
            [-0.015378077319, -0.999870730693, -0.004694320213],
            [-0.010852919054, 0.004861513738, -0.999929287416],
            [0.999822848752, -0.015326042818, -0.010926276839],
        ],
        dtype=np.float64,
    )
    translation_cl = np.asarray(
        [-0.019612411606, -0.077283226016, -0.067023152499],
        dtype=np.float64,
    )

    total_frames = len(manifest)
    if args.smoke_frames:
        total_frames = min(total_frames, args.smoke_frames)
    cache_id = None
    cache_points = None
    cache_labels = None
    matched_frames = 0
    labelled_pixels = 0
    time_differences = []
    class_pixels = np.zeros(len(MCD_CLASSES) + 1, dtype=np.int64)

    for frame_index in range(total_frames):
        timestamp = frame_times[frame_index]
        insertion = int(np.searchsorted(label_times, timestamp))
        candidates = [max(0, insertion - 1), min(len(label_times) - 1, insertion)]
        label_position = min(
            candidates, key=lambda index: abs(label_times[index] - timestamp)
        )
        time_difference = abs(float(label_times[label_position] - timestamp))
        label_image = np.zeros((args.height, args.width), dtype=np.uint8)
        confidence = np.zeros_like(label_image)

        if time_difference <= args.max_time_difference:
            label_id = int(label_ids[label_position])
            if cache_id != label_id:
                cache_points, raw_labels = read_labelled_pcd(
                    args.label_dir / f"cloud_{label_id:04d}.pcd"
                )
                cache_labels = raw_labels.astype(np.uint8) + 1
                cache_id = label_id

            scan_q, scan_t = interpolate_pose(
                frame_times, quaternions, translations, label_times[label_position]
            )
            image_q = quaternions[frame_index]
            image_t = translations[frame_index]
            rotation_wcs = quaternion_to_rotation(scan_q)
            rotation_wci = quaternion_to_rotation(image_q)
            rotation_wl = rotation_wcs @ rotation_cl
            translation_wl = rotation_wcs @ translation_cl + scan_t
            points_w = (rotation_wl @ cache_points.T).T + translation_wl
            points_c = (rotation_wci.T @ (points_w - image_t).T).T
            depth = points_c[:, 2]
            valid = (depth > 0.05) & (depth <= args.max_depth)
            points_c = points_c[valid]
            depth = depth[valid]
            point_labels = cache_labels[valid]
            u = np.rint(args.fx * points_c[:, 0] / depth + args.cx).astype(np.int32)
            v = np.rint(args.fy * points_c[:, 1] / depth + args.cy).astype(np.int32)
            inside = (u >= 0) & (u < args.width) & (v >= 0) & (v < args.height)
            u, v, depth, point_labels = (
                u[inside], v[inside], depth[inside], point_labels[inside]
            )
            pixel = v.astype(np.int64) * args.width + u
            order = np.lexsort((depth, pixel))
            if order.size:
                sorted_pixel = pixel[order]
                first = np.empty(order.size, dtype=bool)
                first[0] = True
                first[1:] = sorted_pixel[1:] != sorted_pixel[:-1]
                keep = order[first]
                u, v, point_labels = u[keep], v[keep], point_labels[keep]
                label_image[v, u] = point_labels
                confidence[v, u] = 255
                np.add.at(class_pixels, point_labels, 1)
                labelled_pixels += len(keep)
            matched_frames += 1
            time_differences.append(time_difference)

        stem = Path(manifest[frame_index][0]).stem
        cv2.imwrite(str(args.output / "labels" / f"{stem}.png"), label_image)
        cv2.imwrite(str(args.output / "confidence" / f"{stem}.png"), confidence)
        if frame_index in {0, total_frames // 2, total_frames - 1}:
            image = cv2.imread(
                str(args.export_dir / manifest[frame_index][0]), cv2.IMREAD_COLOR
            )
            palette = cv2.applyColorMap(
                np.rint(label_image * (255.0 / max(len(MCD_CLASSES), 1))).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )
            known = label_image > 0
            overlay = image.copy()
            overlay[known] = (
                0.35 * image[known] + 0.65 * palette[known]
            ).astype(np.uint8)
            cv2.imwrite(
                str(args.output / "visualization" / f"{stem}_overlay.png"), overlay
            )
        if (frame_index + 1) % 250 == 0 or frame_index + 1 == total_frames:
            print(
                f"processed={frame_index + 1}/{total_frames} "
                f"matched={matched_frames} labelled_pixels={labelled_pixels}"
            )

    metadata = {
        "source": "MCD ntu_day_02 annotated Livox PCD",
        "label_indexing": "cloud_NNNN uses zero-based /livox/lidar message index",
        "class_count_with_unknown": len(MCD_CLASSES) + 1,
        "classes": {"0": "unknown"}
        | {str(index + 1): name for index, name in enumerate(MCD_CLASSES)},
        "frames": total_frames,
        "matched_frames": matched_frames,
        "labelled_pixels": labelled_pixels,
        "mean_labelled_pixels_per_matched_frame": (
            labelled_pixels / matched_frames if matched_frames else 0
        ),
        "mean_time_difference_seconds": (
            float(np.mean(time_differences)) if time_differences else None
        ),
        "max_time_difference_seconds": (
            float(np.max(time_differences)) if time_differences else None
        ),
        "class_pixel_counts": class_pixels.tolist(),
        "intrinsics": {
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        "T_camera_lidar": {
            "R": rotation_cl.tolist(),
            "t": translation_cl.tolist(),
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
