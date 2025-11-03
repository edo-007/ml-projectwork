"""
Package principale del progetto ML.
"""

from src.models import (
    get_model_and_grid,
    get_parametrized_estimator,
    MODEL_REGISTRY,
    list_available_models
)

from src.utils import (
    get_embeddings,
    load_embeddings,
    compute_embeddings,
    load_dataset,
    Colors
)

__all__ = [
    'get_model_and_grid',
    'get_parametrized_estimator',
    'MODEL_REGISTRY',
    'list_available_models',
    'get_embeddings',
    'load_embeddings',
    'compute_embeddings',
    'load_dataset',
    'Colors'
]
