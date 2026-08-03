#!/usr/bin/env python3
"""Compare matching scalar vertex properties in two PLY files."""

import argparse

import numpy as np


PLY_DTYPES = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "uint8": "u1",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
}


def read_binary_vertex_ply(path: str) -> np.ndarray:
    properties: list[tuple[str, str]] = []
    vertex_count = None
    reading_vertices = False
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"unterminated PLY header: {path}")
            line = raw.decode("ascii").strip()
            fields = line.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                pass
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                reading_vertices = True
            elif fields and fields[0] == "element":
                reading_vertices = False
            elif fields and fields[0] == "property" and reading_vertices:
                if fields[1] == "list":
                    raise ValueError("list vertex properties are not supported")
                properties.append((fields[2], PLY_DTYPES[fields[1]]))
            elif line == "end_header":
                offset = stream.tell()
                break
    if vertex_count is None:
        raise ValueError(f"missing vertex element: {path}")
    return np.memmap(
        path,
        dtype=np.dtype(properties),
        mode="r",
        offset=offset,
        shape=(vertex_count,),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()

    first = read_binary_vertex_ply(args.first)
    second = read_binary_vertex_ply(args.second)
    if len(first) != len(second):
        raise SystemExit(f"vertex count differs: {len(first)} != {len(second)}")

    names = [
        name
        for name in first.dtype.names
        if name in second.dtype.names
        and not any(name.startswith(prefix) for prefix in args.exclude_prefix)
    ]
    differences = {
        name: float(
            np.max(
                np.abs(
                    first[name].astype(np.float64)
                    - second[name].astype(np.float64)
                )
            )
        )
        for name in names
    }
    changed = {name: value for name, value in differences.items() if value != 0.0}
    print(f"vertices={len(first)}")
    print(f"compared_fields={len(names)}")
    print(f"max_abs={max(differences.values(), default=0.0):.9g}")
    print(f"changed_fields={changed}")

    for label, data in (("first", first), ("second", second)):
        semantic = []
        for name in data.dtype.names:
            if name.startswith("semantic_"):
                values = data[name].astype(np.float64)
                semantic.append(
                    (name, float(values.min()), float(values.max()), float(values.std()))
                )
        if semantic:
            print(f"{label}_semantic_stats={semantic}")


if __name__ == "__main__":
    main()
