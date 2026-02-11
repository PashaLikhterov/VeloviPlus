<a href="https://scvi-tools.org/">
  <img
    src="https://github.com/scverse/scvi-tools/blob/main/docs/_static/scvi-tools-horizontal.svg?raw=true"
    width="400"
    alt="scvi-tools"
  >
</a>

[![Stars][gh-stars-badge]][gh-stars-link]
[![PyPI][pypi-badge]][pypi-link]
[![PyPIDownloads][pepy-badge]][pepy-link]
[![CondaDownloads][conda-badge]][conda-link]
[![Docs][docs-badge]][docs-link]
[![Build][build-badge]][build-link]
[![Coverage][coverage-badge]][coverage-link]

[scvi-tools] (single-cell variational inference tools) is a package for probabilistic modeling and
analysis of single-cell omics data, built on top of [PyTorch] and [AnnData].

# Analysis of single-cell omics data

scvi-tools is composed of models that perform many analysis tasks across single-cell, multi, and
spatial omics data:

- Dimensionality reduction
- Data integration
- Automated annotation
- Factor analysis
- Doublet detection
- Spatial deconvolution
- and more!

In the [user guide], we provide an overview of each model. All model implementations have a
high-level API that interacts with [Scanpy] and includes standard save/load functions, GPU
acceleration, etc.

# Rapid development of novel probabilistic models

scvi-tools contains the building blocks to develop and deploy novel probabilistic models. These
building blocks are powered by popular probabilistic and machine learning frameworks such as
[PyTorch Lightning] and [Pyro]. For an overview of how the scvi-tools package is structured, you
may refer to the [codebase overview] page.

We recommend checking out the [skeleton repository] as a starting point for developing and
deploying new models with scvi-tools.

# Basic installation

For conda,

```bash
conda install scvi-tools -c conda-forge
```

and for pip,

```bash
pip install scvi-tools
```

Please be sure to install a version of [PyTorch] that is compatible with your GPU (if applicable).

# Experimental: RNA Velocity Benchmarking (veloVI improvements)

This repository is the upstream `scvi-tools` codebase, **plus** an experimental RNA velocity
benchmarking + training pipeline under:

- `src/scvi/experimental/velovi_improvements/`

The goal is to compare multiple velocity variants under a shared preprocessing / evaluation setup
and log **figures + tables** to Weights & Biases (W&B), including a re-plot workflow that allows you
to regenerate streamlines later from a W&B run ID.

## What’s implemented

The runner trains/evaluates a set of velocity variants (depending on flags/config):

- `scvelo_dynamic`: scVelo dynamical model baseline
- `baseline`: veloVI baseline
- `baseline_latent`: baseline + post-hoc latent smoothing
- `baseline_gnn`: veloVI with a GNN encoder
- `baseline_gnn_latent`: GNN + latent smoothing
- `baseline_transformer`: baseline + transformer velocity refinement
- `baseline_transformer_latent`: transformer + latent smoothing
- Optional: TIvelo guidance (as supervision/prior for refinements)

## Quickstart (local)

Prerequisites (beyond core `scvi-tools`):

```bash
pip install -e .
pip install scvelo wandb umap-learn
```

Run a single dataset locally:

```bash
python -m scvi.experimental.velovi_improvements.runner \
  /path/to/data_dir \
  --datasets pancreas_endocrinogenesis \
  --output-dir results/velovi_benchmark \
  --plot-results \
  --use-wandb \
  --wandb-project RNA-Velocity \
  --wandb-run-group benchmark
```

If your `.h5ad` files are already preprocessed, add:

```bash
--skip-preprocess
```

## Running on the cluster (Run:AI wrapper scripts)

Most jobs are launched via `scripts/` (they call the same runner module):

- `scripts/run_benchmark_transformer.sh` (benchmark suite on one/multiple datasets)
- `scripts/run_tivelo_dynamic.sh` (dynamic + TIvelo-guidance focused run)
- `scripts/run_adata_combined.sh` (adata_combined experiments)

Example:

```bash
DATA_DIR=/gpfs0/.../proj/datasets \
DATASETS="dentate_gyrus" \
OUTPUT_DIR=/gpfs0/.../proj/scvi-tools/results/velovi_benchmark \
WANDB_PROJECT=RNA-Velocity \
./scripts/run_benchmark_transformer.sh
```

### Enabling TIvelo guidance in scripts

Scripts support toggling guidance with env vars (example for `run_benchmark_transformer.sh`):

```bash
USE_TIVELO_GUIDANCE=1 \
TIVELO_PRIOR_STRENGTH=0.4 \
TIVELO_TREE_GENE=Cplx2 \
TIVELO_EPOCHS=100 \
./scripts/run_benchmark_transformer.sh
```

## Outputs (on disk)

For each dataset `X`, the runner writes to:

- `results/velovi_X/` (or whatever `--output-dir` you passed)
  - `*_velocity_streamlines.png` (per-variant stream plots)
  - `figures/` (panels + comparisons)
  - `tables/`
    - `*_advanced_summary.csv` / `*_advanced_edges.csv` (metrics)
    - `*_runtime.csv` / `*_training_runtime.csv`
    - `*_stage_status.csv` (which stages succeeded/failed)

## Reproducible re-plotting via W&B run ID

When W&B is enabled, each dataset logs a **plot bundle artifact** containing:

- `adata.h5ad` (obs + embeddings + required layers)
- `velocities.npz` (velocity matrices for all plotted variants)
- `metadata.json` (dataset/config metadata)

This makes a run “reproducible for plotting” via the W&B run path:

```
ENTITY/PROJECT/RUN_ID
```

### Re-plot streamlines from an existing W&B run

```bash
python -m scvi.experimental.velovi_improvements.replot \
  --wandb-run ENTITY/PROJECT/RUN_ID \
  --dataset dentate_gyrus \
  --output-dir replot_outputs/RUN_ID \
  --figsize 20,12 \
  --legend-right 0.84
```

If you are on a cluster where `~/.cache/wandb` is not writable, `replot` automatically redirects
W&B cache/config directories into your `--output-dir`.

### Plot on the dataset’s original UMAP

UMAPs are only identifiable up to rotation/flip; if the pipeline recomputes UMAP you can see an
“upside down” embedding.

The pipeline is designed to **reuse any existing** `adata.obsm["X_umap*"]` embedding and avoid
recomputing UMAP during preprocessing/plotting. If you still want to force plotting on a specific
AnnData file’s embedding, pass `--adata-path` to `replot`:

```bash
python -m scvi.experimental.velovi_improvements.replot \
  --wandb-run ENTITY/PROJECT/RUN_ID \
  --dataset dentate_gyrus \
  --adata-path /gpfs0/.../proj/datasets/DentateGyrus_processed.h5ad \
  --output-dir replot_outputs/RUN_ID_original_umap
```

## Dataset registry

Datasets are defined in:

- `src/scvi/experimental/velovi_improvements/dataset_configs/*.yaml`

These YAMLs define:

- dataset name + expected file(s)
- layer keys (spliced/unspliced)
- plotting keys (clusters/cell types)
- optional cluster edges for cross-boundary metrics
- whether preprocessing should run

If a dataset file is not found in `data_dir` and a `scvelo_loader` exists, the pipeline can
download via `scvelo.datasets.<loader>()`.

## Troubleshooting

- **One model fails and the whole run stops**: optional stages are executed defensively; the run
  should continue and record failures in `tables/*_stage_status.csv`.
- **Legend squeezes UMAP**: re-plot with a larger `--figsize` and adjust `--legend-right`.
- **UMAP not “original”**: verify which embedding keys exist in `adata.obsm` (e.g. `X_umap_orig`);
  dataset configs can be adjusted to prefer a specific key.

# Resources

- Tutorials, API reference, and installation guides are available in the [documentation].
- For discussion of usage, check out our [forum].
- Please use the [issues] to submit bug reports.
- If you'd like to contribute, check out our [contributing guide].
- If you find a model useful for your research, please consider citing the corresponding
    publication.

# Reference

If you use `scvi-tools` in your work, please cite

> **A Python library for probabilistic analysis of single-cell omics data**
>
> Adam Gayoso, Romain Lopez, Galen Xing, Pierre Boyeau, Valeh Valiollah Pour Amiri, Justin Hong,
> Katherine Wu, Michael Jayasuriya, Edouard Mehlman, Maxime Langevin, Yining Liu, Jules Samaran,
> Gabriel Misrachi, Achille Nazaret, Oscar Clivio, Chenling Xu, Tal Ashuach, Mariano Gabitto,
> Mohammad Lotfollahi, Valentine Svensson, Eduardo da Veiga Beltrame, Vitalii Kleshchevnikov,
> Carlos Talavera-López, Lior Pachter, Fabian J. Theis, Aaron Streets, Michael I. Jordan,
> Jeffrey Regier & Nir Yosef
>
> _Nature Biotechnology_ 2022 Feb 07. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w).

along with the publication describing the model used.

You can cite the scverse publication as follows:

> **The scverse project provides a computational ecosystem for single-cell omics data analysis**
>
> Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor Sturm, Adam Gayoso,
> Ilia Kats, Mikaela Koutrouli, Scverse Community, Bonnie Berger, Dana Pe’er, Aviv Regev,
> Sarah A. Teichmann, Francesca Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle &
> Fabian J. Theis
>
> _Nature Biotechnology_ 2023 Apr 10. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).

scvi-tools is part of the scverse® project ([website](https://scverse.org),
[governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).

If you like scverse® and want to support our mission, please consider making a tax-deductible
[donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time,
professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

Copyright (c) 2025, Yosef Lab, Weizmann Institute of Science

[anndata]: https://anndata.readthedocs.io/en/latest/
[build-badge]: https://github.com/scverse/scvi-tools/actions/workflows/build.yml/badge.svg
[build-link]: https://github.com/scverse/scvi-tools/actions/workflows/build.yml/
[codebase overview]: https://docs.scvi-tools.org/en/stable/user_guide/background/codebase_overview.html
[conda-badge]: https://img.shields.io/conda/dn/conda-forge/scvi-tools?logo=Anaconda
[conda-link]: https://anaconda.org/conda-forge/scvi-tools
[contributing guide]: https://docs.scvi-tools.org/en/stable/contributing/index.html
[coverage-badge]: https://codecov.io/gh/scverse/scvi-tools/branch/main/graph/badge.svg
[coverage-link]: https://codecov.io/gh/scverse/scvi-tools
[docs-badge]: https://readthedocs.org/projects/scvi/badge/?version=latest
[docs-link]: https://scvi.readthedocs.io/en/stable/?badge=stable
[documentation]: https://docs.scvi-tools.org/
[forum]: https://discourse.scvi-tools.org
[gh-stars-badge]: https://img.shields.io/github/stars/scverse/scvi-tools?style=flat&logo=GitHub&color=blue
[gh-stars-link]: https://github.com/scverse/scvi-tools/stargazers
[issues]: https://github.com/scverse/scvi-tools/issues
[pepy-badge]: https://static.pepy.tech/badge/scvi-tools
[pepy-link]: https://pepy.tech/project/scvi-tools
[pypi-badge]: https://img.shields.io/pypi/v/scvi-tools.svg
[pypi-link]: https://pypi.org/project/scvi-tools
[pyro]: https://pyro.ai/
[pytorch]: https://pytorch.org
[pytorch lightning]: https://lightning.ai/docs/pytorch/stable/
[scanpy]: http://scanpy.readthedocs.io/
[scvi-tools]: https://scvi-tools.org/
[skeleton repository]: https://github.com/scverse/simple-scvi
[user guide]: https://docs.scvi-tools.org/en/stable/user_guide/index.html
