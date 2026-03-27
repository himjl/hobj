import pandas as pd
import xarray as xr

import hobj.benchmarks.mut_highvar_benchmark as highvar_module
import hobj.benchmarks.mut_oneshot_benchmark as oneshot_module
from hobj.learning_models.random_guesser import RandomGuesser


def _make_highvar_images_df() -> pd.DataFrame:
    rows = []
    for category in ["CatA", "CatB", "CatC", "CatD"]:
        for index in range(50):
            rows.append(
                {
                    "image_id": f"{category}_{index:03d}",
                    "category": category,
                }
            )
    return pd.DataFrame(rows)


def _make_highvar_behavior_df() -> pd.DataFrame:
    rows = []
    for assignment_id, worker_id, class_a, class_b in [
        ("assignment-1", "worker-1", "CatA", "CatB"),
        ("assignment-2", "worker-2", "CatC", "CatD"),
    ]:
        for trial in range(100):
            category = class_a if trial % 2 == 0 else class_b
            rows.append(
                {
                    "assignment_id": assignment_id,
                    "worker_id": worker_id,
                    "trial": trial,
                    "image_id": f"{category}_{trial // 2:03d}",
                    "perf": trial % 3 == 0,
                }
            )
    return pd.DataFrame(rows)


def _make_oneshot_images_df() -> pd.DataFrame:
    rows = []
    for category in ["CatA", "CatB"]:
        rows.append(
            {
                "image_id": f"{category}_original",
                "category": category,
                "transformation": "original",
                "transformation_level": 0.0,
            }
        )
        for index in range(30):
            rows.append(
                {
                    "image_id": f"{category}_blur_{index:02d}",
                    "category": category,
                    "transformation": "blur",
                    "transformation_level": 0.1,
                }
            )
        for index in range(30):
            rows.append(
                {
                    "image_id": f"{category}_noise_{index:02d}",
                    "category": category,
                    "transformation": "noise",
                    "transformation_level": 0.2,
                }
            )
    return pd.DataFrame(rows)


def _make_oneshot_behavior_df() -> pd.DataFrame:
    transformed_a = [
        "CatA_blur_00",
        "CatB_noise_00",
        "CatA_noise_01",
        "CatB_blur_01",
        "CatA_blur_02",
        "CatB_noise_02",
        "CatA_noise_03",
        "CatB_blur_03",
    ]
    transformed_b = [
        "CatB_blur_04",
        "CatA_noise_04",
        "CatB_noise_05",
        "CatA_blur_05",
        "CatB_blur_06",
        "CatA_noise_06",
        "CatB_noise_07",
        "CatA_blur_07",
    ]
    rows = []
    for assignment_id, slot, worker_id, transformed_images in [
        ("assignment-1", 0, "worker-1", transformed_a),
        ("assignment-2", 0, "worker-2", transformed_b),
    ]:
        transformed_iter = iter(transformed_images)
        for trial in range(20):
            if trial in {9, 14, 19} or trial < 9:
                image_id = "CatA_original" if trial % 2 == 0 else "CatB_original"
            else:
                image_id = next(transformed_iter)

            rows.append(
                {
                    "assignment_id": assignment_id,
                    "slot": slot,
                    "worker_id": worker_id,
                    "trial": trial,
                    "image_id": image_id,
                    "perf": trial % 2 == 0,
                    "subtask": "CatA,CatB",
                }
            )
    return pd.DataFrame(rows)


def test_highvar_target_and_model_statistics_behave_like_datasets(monkeypatch):
    monkeypatch.setattr(
        highvar_module,
        "load_imageset_meta_highvar",
        lambda cachedir=None: _make_highvar_images_df(),
    )
    monkeypatch.setattr(
        highvar_module,
        "load_highvar_behavior",
        lambda remove_probe_trials=True, cachedir=None: _make_highvar_behavior_df(),
    )
    monkeypatch.setattr(
        highvar_module.MutatorHighVarBenchmark,
        "num_bootstrap_samples",
        4,
    )
    monkeypatch.setattr(
        highvar_module.MutatorHighVarBenchmark,
        "num_simulations_per_subtask",
        3,
    )

    benchmark = highvar_module.MutatorHighVarBenchmark()
    target_stats = benchmark.target_statistics

    assert isinstance(target_stats, xr.Dataset)
    assert target_stats.phat.mean("subtask").sizes == {"trial": 100}
    assert target_stats.boot_phat.mean("subtask").sizes == {
        "boot_iter": 4,
        "trial": 100,
    }
    assert target_stats.sel(subtask=benchmark.subtask_names[0]).phat.sizes == {
        "trial": 100
    }

    result = benchmark.score_model(RandomGuesser(seed=0))
    assert isinstance(result.model_statistics, xr.Dataset)
    assert result.model_statistics.boot_phat.sizes == {
        "boot_iter": 4,
        "subtask": 2,
        "trial": 100,
    }


def test_oneshot_target_and_model_statistics_behave_like_datasets(monkeypatch):
    monkeypatch.setattr(
        oneshot_module,
        "load_imageset_meta_oneshot",
        lambda cachedir=None: _make_oneshot_images_df(),
    )
    monkeypatch.setattr(
        oneshot_module,
        "load_oneshot_behavior",
        lambda cachedir=None: _make_oneshot_behavior_df(),
    )
    monkeypatch.setattr(
        oneshot_module.MutatorOneshotBenchmark,
        "subtask_names",
        ["CatA,CatB"],
    )
    monkeypatch.setattr(
        oneshot_module.MutatorOneshotBenchmark,
        "transformation_ids",
        ["blur | 0.1", "noise | 0.2"],
    )
    monkeypatch.setattr(
        oneshot_module.MutatorOneshotBenchmark,
        "num_bootstrap_samples",
        4,
    )
    monkeypatch.setattr(
        oneshot_module.MutatorOneshotBenchmark,
        "num_simulations_per_subtask",
        3,
    )
    monkeypatch.setattr(
        oneshot_module.MutatorOneshotBenchmark,
        "bootstrap_target_by_worker",
        False,
    )

    benchmark = oneshot_module.MutatorOneshotBenchmark()
    target_stats = benchmark.target_statistics

    assert isinstance(target_stats, xr.Dataset)
    assert [name for name, _ in target_stats.groupby("transformation_type")] == [
        "blur",
        "noise",
    ]
    assert target_stats.sel(transformation="blur | 0.1").phat.sizes == {}
    assert target_stats.boot_phat.sizes == {
        "boot_iter": 4,
        "transformation": 2,
    }

    result = benchmark.score_model(RandomGuesser(seed=0))
    assert isinstance(result.model_statistics, xr.Dataset)
    assert result.model_statistics.sel(
        transformation=target_stats.transformation
    ).phat.sizes == {"transformation": 2}
