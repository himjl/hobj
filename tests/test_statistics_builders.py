import numpy as np
import xarray as xr

from hobj.benchmarks.binary_classification.estimator import (
    build_learning_curve_statistics,
)
from hobj.benchmarks.binary_classification.simulation import (
    BinaryClassificationSubtaskResult,
)
from hobj.benchmarks.generalization.estimator import build_generalization_statistics
from hobj.benchmarks.generalization.simulator import GeneralizationSessionResult


def test_build_learning_curve_statistics_returns_expected_dataset_shape():
    stats = build_learning_curve_statistics(
        subtask_name_to_results={
            "task-b": [
                BinaryClassificationSubtaskResult(
                    perf_seq=np.array([True, False, True]),
                    worker_id="worker-1",
                ),
                BinaryClassificationSubtaskResult(
                    perf_seq=np.array([False, False, True]),
                    worker_id="worker-2",
                ),
            ],
            "task-a": [
                BinaryClassificationSubtaskResult(
                    perf_seq=np.array([True, True, False]),
                    worker_id="worker-3",
                ),
                BinaryClassificationSubtaskResult(
                    perf_seq=np.array([True, False, False]),
                    worker_id="worker-4",
                ),
            ],
        },
        nbootstrap_samples=4,
        bootstrap_by_worker=False,
    )

    assert isinstance(stats, xr.Dataset)
    assert set(stats.data_vars) == {
        "phat",
        "varhat_phat",
        "boot_phat",
        "boot_varhat_phat",
    }
    assert dict(stats.sizes) == {"subtask": 2, "trial": 3, "boot_iter": 4}
    assert list(stats.subtask.values) == ["task-a", "task-b"]
    np.testing.assert_allclose(
        stats.phat.values,
        np.array(
            [
                [1.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
            ]
        ),
    )
    assert stats.boot_phat.shape == (4, 2, 3)
    assert stats.boot_varhat_phat.shape == (4, 2, 3)


def test_build_generalization_statistics_returns_expected_dataset_shape():
    stats = build_generalization_statistics(
        results=[
            GeneralizationSessionResult(
                transformation_to_kn={
                    "blur | 0.1": [1, 2],
                    "noise | 0.2": [0, 1],
                },
                kcatch=1,
                ncatch=2,
                worker_id="worker-1",
            ),
            GeneralizationSessionResult(
                transformation_to_kn={
                    "blur | 0.1": [1, 1],
                    "noise | 0.2": [1, 2],
                },
                kcatch=2,
                ncatch=2,
                worker_id="worker-2",
            ),
        ],
        perform_lapse_rate_correction=False,
        n_bootstrap_iterations=5,
        bootstrap_by_worker=False,
    )

    assert isinstance(stats, xr.Dataset)
    assert set(stats.data_vars) == {
        "phat",
        "varhat_phat",
        "boot_phat",
        "boot_varhat_phat",
    }
    assert dict(stats.sizes) == {"transformation": 2, "boot_iter": 5}
    assert list(stats.transformation.values) == ["blur | 0.1", "noise | 0.2"]
    np.testing.assert_allclose(stats.phat.values, np.array([2 / 3, 1 / 3]))
    assert stats.boot_phat.shape == (5, 2)
    assert stats.boot_varhat_phat.shape == (5, 2)
