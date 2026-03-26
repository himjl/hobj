"""
This module provides an alternative interface for instantiating a linear learning model.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

import hobj.learning_models.update_rules as update_rules
from hobj.learning_models import LinearLearner, RepresentationalModel
from hobj.types import ImageId


# %%
@lru_cache(maxsize=None)
def _get_calibration_image_ids(cachedir: Path | None = None) -> list[ImageId]:
    """Return the warmup image IDs used for feature calibration.

    Args:
        cachedir: Optional directory containing the packaged ``data`` tree.

    Returns:
        Sorted image IDs used for calibration.
    """
    from hobj.data_loaders.images import load_imageset_meta_warmup

    images_df = load_imageset_meta_warmup(cachedir=cachedir)
    return sorted(images_df["image_id"].tolist())


# %%
def make_linear_learner_from_features(
    ref_to_features: dict[ImageId, np.ndarray],
    update_rule_name: Literal[
        "Prototype",
        "Square",
        "Perceptron",
        "Hinge",
        "MAE",
        "Exponential",
        "CE",
        "REINFORCE",
    ] = "Square",
    alpha: float = 1,
    cachedir: Path | None = None,
) -> LinearLearner:
    """Instantiate a linear learner from precomputed features.

    Args:
        ref_to_features: Precomputed features keyed by image ID.
        update_rule_name: Name of the update rule to use.
        alpha: Learning-rate parameter for the update rule.
        cachedir: Optional directory containing the packaged ``data`` tree.

    Returns:
        A calibrated linear learner.
    """

    f_calibration = np.array(
        [ref_to_features[ref] for ref in _get_calibration_image_ids(cachedir=cachedir)]
    )
    mu_calibration = np.mean(f_calibration, axis=0)
    norms_calibration = np.linalg.norm(f_calibration - mu_calibration, axis=1)
    norm_cutoff = np.quantile(norms_calibration, 0.999)  # Will clip the rest

    ref_to_calibrated_features = {}
    for ref in ref_to_features:
        f = ref_to_features[ref]
        fc = f - mu_calibration
        fcn = fc / norm_cutoff
        norm_cur = np.linalg.norm(fcn)
        if norm_cur > 1:
            fcn = fcn / norm_cur
        ref_to_calibrated_features[ref] = np.array(fcn)

    update_rule_name = getattr(update_rules, update_rule_name)
    return LinearLearner(
        representational_model=RepresentationalModel.from_precomputed_features(
            image_ref_to_features=ref_to_calibrated_features
        ),
        update_rule=update_rule_name(alpha=alpha),
    )
