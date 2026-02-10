from __future__ import annotations

from pathlib import Path

import scanpy as sc
import scvelo as scv


def build_and_save_knn_variants(
    source: str | Path,
    ks: tuple[int, ...] = (15, 25, 45),
    n_pcs: int = 30,
) -> None:
    """Rebuild neighbors with multiple k values and save new .h5ad files near the source."""
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)

    adata_full = sc.read_h5ad(source)

    for k in ks:
        adata_k = adata_full.copy()
        if "X_pca" not in adata_k.obsm_keys():
            sc.tl.pca(adata_k, n_comps=n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata_k, n_neighbors=k, n_pcs=n_pcs)
        scv.pp.moments(adata_k, n_pcs=n_pcs, n_neighbors=k)

        out_path = source.parent / f"{source.stem}_k{k}{source.suffix}"
        adata_k.write_h5ad(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild neighbor graphs for multiple k values.")
    parser.add_argument("adata", help="Path to input .h5ad with fixed UMAP")
    parser.add_argument("--ks", nargs="*", type=int, default=(15, 25, 45))
    parser.add_argument("--n_pcs", type=int, default=30)
    args = parser.parse_args()
    build_and_save_knn_variants(args.adata, ks=tuple(args.ks), n_pcs=args.n_pcs)
