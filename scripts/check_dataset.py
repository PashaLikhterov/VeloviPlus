#!/usr/bin/env python
from pathlib import Path
import sys
import anndata as ad

EMBED_KEYS = ("X_umap", "X_tsne")
COLOR_KEYS = ("clusters", "celltype", "leiden")  # extend per dataset

path = Path(sys.argv[1])
adata = ad.read_h5ad(path)

print(f"File: {path}")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"obsm keys: {list(adata.obsm.keys())}")

missing_embed = all(key not in adata.obsm for key in EMBED_KEYS)
missing_color = all(key not in adata.obs for key in COLOR_KEYS)

if missing_embed:
    print("ERROR: no embedding (X_umap/X_tsne) in `.obsm`")
if missing_color:
    print("ERROR: expected color/cluster column not found in `.obs`")

sys.exit(1 if (missing_embed or missing_color) else 0)
