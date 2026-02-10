from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    U_KEY: str = "U"
    NEIGHBOR_INDEX_KEY: str = "neighbor_index"
    NEIGHBOR_WEIGHT_KEY: str = "neighbor_weight"
    NEIGHBOR_INDEX_KEY_LATENT: str = "neighbor_index_latent"
    NEIGHBOR_WEIGHT_KEY_LATENT: str = "neighbor_weight_latent"
    FUTURE_INDEX_KEY: str = "future_index"
    FUTURE_WEIGHT_KEY: str = "future_weight"


VELOVI_REGISTRY_KEYS = _REGISTRY_KEYS_NT()
