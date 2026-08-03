# GLIC-Sem

GLIC-Sem is a research prototype for incremental semantic Gaussian mapping. It uses FAST-LIVO2 as the front end, initializes 3D Gaussians from frame-wise world-coordinate LiDAR observations, preserves the Gaussian-LIC2 photometric and geometric mapper, and learns a separate open-vocabulary language feature head.

The current source snapshot focuses on one scientific question: **can semantic features be added without damaging RGB/depth reconstruction?** It therefore includes a geometry-preserving `head-only` training path and a reproducible error-decomposition evaluator. It does not include datasets, pretrained models, checkpoints, point clouds, rendered results, or credentials.

## Pipeline

```text
FAST-LIVO2
  image + T_world_camera + current-frame world LiDAR cloud
        |
        v
Gaussian-LIC2 backend
  frustum projection + nearest-Z + Gaussian initialization
        |
        +--> RGB/depth Gaussian optimization
        |
        +--> frozen-geometry language-feature optimization
                    |
                    v
          arbitrary text queries / semantic rendering
```

The semantic teacher follows the LangSplat-style idea of combining image segmentation with CLIP-family language embeddings. PCA is used only as a storage and optimization representation; the included diagnostics separately measure teacher quality, PCA distortion, Gaussian fitting error, and text-query decoding error.

## Current evidence

On the 60-frame MCD pilot used during development:

| Method | Held-out PSNR | Held-out Depth-L1 | Train language cosine | Held-out language cosine |
|---|---:|---:|---:|---:|
| Geometry baseline | 22.20 dB | 2.6650 m | - | - |
| Joint RGB/depth/language optimization | 17.61 dB | 3.2490 m | 0.8549 | 0.8466 |
| Head-only language optimization | 22.20 dB | 2.6643 m | 0.8677 | 0.8571 |

These are pilot results, not a final open-vocabulary claim. Sparse disjoint-scan evaluation still shows poor small-class IoU, and PCA top-1 agreement is not yet sufficient for scientific expansion. See the protocol document for the required gates and adversarial checks.

## Repository layout

- `src/`: Gaussian mapper and language-target integration.
- `config/`: MCD geometry and semantic experiment configurations.
- `scripts/semantic/`: teacher preparation, PCA tools, rendering, scoring, and error decomposition.
- `scripts/run_mcd_semantic_error_decomposition_60.sh`: reproducible 60-frame experiment driver.
- `docs/OPEN_VOCAB_SEMANTIC_ERROR_DECOMPOSITION_PROTOCOL.md`: first-principles validation protocol.
- `docs/experiments/MCD_S0_S2_60FRAME_20260803.md`: audited 60-frame S0-S2 results and the next single-variable experiment.
- `docs/SOURCE_ONLY_RELEASE_MANIFEST.md`: source release boundary and exclusions.
- `tools/audit_git_payload.py`: pre-commit guard against weights, datasets, results, binaries, and credentials.

## Safety and release boundary

Before publishing changes, run:

```bash
python tools/audit_git_payload.py --repo .
```

The guard rejects common model, dataset, point-cloud, archive, result, build, and credential paths. Large third-party dependencies and pretrained weights must be obtained from their official sources and must not be committed here.

## Status

This repository is an experimental source snapshot rather than a turnkey release. The next milestone is to pass the full error-decomposition gates on independent frames and scenes before tuning semantic loss weights or claiming open-scene/open-vocabulary performance.
