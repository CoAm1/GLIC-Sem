#!/usr/bin/env python3
"""Write the labelled LiDAR scan used by each FAST-LIVO2 camera frame."""

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamps", required=True, type=Path)
    parser.add_argument("--ros2-db", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--max-time-difference", type=float, default=0.061)
    args = parser.parse_args()

    frame_times = np.asarray(
        [
            float(line.split()[1])
            for line in args.timestamps.read_text().splitlines()
            if line.strip()
        ]
    )
    connection = sqlite3.connect(str(args.ros2_db))
    try:
        topic_id = connection.execute(
            "SELECT id FROM topics WHERE name = ?", (args.topic,)
        ).fetchone()[0]
        lidar_times = np.asarray(
            [
                row[0] * 1e-9
                for row in connection.execute(
                    "SELECT timestamp FROM messages WHERE topic_id = ? "
                    "ORDER BY timestamp",
                    (topic_id,),
                )
            ]
        )
    finally:
        connection.close()
    label_ids = np.asarray(
        sorted(
            int(path.stem.split("_")[-1])
            for path in args.label_dir.glob("cloud_*.pcd")
        ),
        dtype=np.int64,
    )
    label_times = lidar_times[label_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame_index", "frame_timestamp", "label_scan_index",
             "label_timestamp", "absolute_time_difference_seconds"]
        )
        matched = 0
        for frame_index, frame_time in enumerate(frame_times):
            insertion = int(np.searchsorted(label_times, frame_time))
            candidates = [
                max(0, insertion - 1),
                min(len(label_times) - 1, insertion),
            ]
            position = min(
                candidates, key=lambda index: abs(label_times[index] - frame_time)
            )
            difference = abs(float(label_times[position] - frame_time))
            if difference <= args.max_time_difference:
                scan_id = int(label_ids[position])
                scan_time = float(label_times[position])
                matched += 1
            else:
                scan_id = -1
                scan_time = float("nan")
            writer.writerow(
                [frame_index, f"{frame_time:.9f}", scan_id,
                 f"{scan_time:.9f}", f"{difference:.9f}"]
            )
    print(f"frames={len(frame_times)} matched={matched} output={args.output}")


if __name__ == "__main__":
    main()
