# Source-only semantic Gaussian-LIC2 release manifest

This source-only patch is intended for the `feature/open-vocab-error-decomposition`
branch of `easymoneysniper-kd7/G-lic_fastlivo2`.

It contains no checkpoints, TensorRT engines, datasets, point clouds, rendered
images, experiment outputs, credentials, or machine secrets. Model and data
paths are supplied only at runtime.

## Incremental mapper runtime

- `CMakeLists.txt`
- `src/camera.h`
- `src/fastlivo_incremental.cpp`
- `src/gaussian.cpp`
- `src/gaussian.h`
- `src/mapping.h`
- `src/language_targets.cpp`
- `src/language_targets.h`
- `src/test_language_targets.cpp`
- `config/mcd_ntu_day_02_fastlivo2_640x480_nospnet.yaml`
- `config/mcd_ntu_day_02_fastlivo2_640x480_semantic12_headonly.yaml`

The runtime files above are copied from the GPU-server revision that produced
the recorded MCD 60-frame Geometry/Joint/Head-only experiment. They implement
offline FAST-LIVO2 input, PCA language targets, independent semantic
optimization, semantic PLY persistence, and raw language-score rendering.

## Teacher, compression, query, and evaluation code

- `scripts/semantic/extract_langsplat_teacher.py`
- `scripts/semantic/fit_universal_language_pca.py`
- `scripts/semantic/export_pca_language_targets.py`
- `scripts/semantic/export_pca_text_queries.py`
- `scripts/semantic/evaluate_pca_text_query_preservation.py`
- `scripts/semantic/evaluate_pca_langsplat_relevancy.py`
- `scripts/semantic/evaluate_mcd_open_vocab_scores.py`
- `scripts/semantic/test_evaluate_mcd_open_vocab_scores.py`
- `scripts/semantic/evaluate_semantic_error_decomposition.py`
- `scripts/semantic/test_evaluate_semantic_error_decomposition.py`
- `scripts/semantic/compare_ply_fields.py`
- `scripts/semantic/colorize_pca_language_queries.py`
- `scripts/semantic/make_mcd_frame_scan_alignment.py`
- `scripts/semantic/prepare_mcd_lidar_semantics.py`
- `scripts/semantic/remap_mcd_to_superclasses.py`
- `scripts/run_mcd_semantic_error_decomposition_60.sh`

## Reproducibility and safety

- `docs/OPEN_VOCAB_SEMANTIC_ERROR_DECOMPOSITION_PROTOCOL.md`
- `docs/SOURCE_ONLY_RELEASE_MANIFEST.md`
- `tools/audit_git_payload.py`
- `tools/test_audit_git_payload.py`
- `.gitignore`

Before committing, run:

```bash
python3 scripts/semantic/test_evaluate_semantic_error_decomposition.py
python3 scripts/semantic/test_evaluate_mcd_open_vocab_scores.py
python3 tools/test_audit_git_payload.py
python3 tools/audit_git_payload.py --repo .
```

The final command audits staged files and fails on model/data extensions,
generated output directories, likely credentials, unapproved binaries, or
files larger than 10 MiB.

## External assets intentionally omitted

The code expects users to obtain and configure the following separately:

- SAM1 ViT-H checkpoint;
- OpenCLIP ViT-B/16 checkpoint;
- SPNet/TensorRT assets when depth completion is explicitly tested;
- LPIPS weights used by the original Gaussian-LIC evaluation path;
- FAST-LIVO2 or MCD datasets;
- fixed PCA basis and generated per-frame semantic targets.

Their paths must never be committed to source control as actual files. The
experiment report records hashes so an external asset can be verified without
placing it in Git.
