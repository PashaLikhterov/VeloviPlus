#!/usr/bin/env python3
from pathlib import Path

import scvelo as scv
import scanpy as sc

DATA_DIR = Path("/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets")
DATASET_FILE = DATA_DIR / "endocrinogenesis_day15_processed.h5ad"  # pancreas_endocrinogenesis
OUTPUT_FILE = Path("for_reference/pancreas_umap.png")

def main():
    adata = sc.read(DATASET_FILE)
    if "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=30)
        sc.tl.umap(adata)
    color_key = "clusters" if "clusters" in adata.obs else adata.obs.columns[0]
    ax = scv.pl.scatter(
        adata,
        basis="umap",
        color=color_key,
        title="Umap - Pancreas",
        legend_loc="on data",
        linewidths=0,
        size=25,
        alpha=0.85,
        show=False,
    )
    if ax is None:  # scv may return a figure
        ax = sc.pl.scatter(
            adata,
            basis="umap",
            color=color_key,
            title="Umap - Pancreas",
            show=False,
        )
    fig = ax if hasattr(ax, "savefig") else ax.figure
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight")
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
