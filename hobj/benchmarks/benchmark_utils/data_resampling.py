import xarray as xr
import numpy as np
from tqdm import tqdm, trange


def get_subtask_grain_splithalf_and_bootstrap_resampled_data(ds_raw, nsplits):
    """
    Performs split-half and bootstrap resampling of the data, by worker.

    ds_raw: xr.Dataset with mandatory variables:
        stimulus_url: (trial, session)
        worker_id: (trial, session)
        perf: (trial, session)
        subtask: (session)

    ds_splits:
        perf: (worker_split, split_iter, trial, subtask)
    ds_boot:
        perf: (boot_iter, trial, subtask)
    """

    all_subtasks = sorted(set(ds_raw.subtask.values))
    subtask_to_i_subtask = {sub: i_subtask for (i_subtask, sub) in enumerate(all_subtasks)}
    ntrials = len(ds_raw.trial.values)
    nsubtasks = len(all_subtasks)

    # %% Get workerwise data
    make_subtask_kn_table = lambda: np.zeros((nsubtasks, ntrials, 2)).astype(int)
    worker_to_kn = {}
    nworkers = len(set(ds_raw.worker_id.values))
    for worker_id, ws in tqdm(ds_raw.groupby('worker_id'), total=nworkers):

        worker_to_kn[worker_id] = make_subtask_kn_table()

        perfs = ws.perf.transpose('session', 'trial').values
        subtask_seq = ws.subtask.values

        for i_sess in range(perfs.shape[0]):
            subtask_cur = subtask_seq[i_sess]
            i_subtask = subtask_to_i_subtask[subtask_cur]
            perfs_cur = perfs[i_sess]

            worker_to_kn[worker_id][i_subtask, :, 0] += perfs_cur.astype(int)
            worker_to_kn[worker_id][i_subtask, :, 1] += 1
    all_workers = sorted(worker_to_kn.keys())

    # %% Perform splitting and bootstrapping
    i_split_to_k = []
    i_split_to_n = []

    i_boot_to_k = []
    i_boot_to_n = []

    for i_split in trange(nsplits):
        RS = np.random.RandomState(i_split)
        wcur = RS.permutation(all_workers)
        workers0 = wcur[:len(wcur) // 2]
        workers1 = wcur[len(wcur) // 2:]
        workers_boot = RS.choice(all_workers, replace=True, size=nworkers)

        dat0 = make_subtask_kn_table()
        dat1 = make_subtask_kn_table()
        datboot = make_subtask_kn_table()

        for workers, dat in zip([workers0, workers1, workers_boot], [dat0, dat1, datboot]):
            for w in workers:
                dat += worker_to_kn[w]

        i_split_to_k.append((dat0[..., 0], dat1[..., 0]))
        i_split_to_n.append((dat0[..., 1], dat1[..., 1]))
        i_boot_to_k.append(datboot[..., 0])
        i_boot_to_n.append(datboot[..., 1])

    ds_splits = xr.Dataset(
        data_vars={
            'k': (['split_iter', 'worker_split', 'subtask', 'trial'], i_split_to_k),
            'n': (['split_iter', 'worker_split', 'subtask', 'trial'], i_split_to_n)
        },
        coords={
            'subtask': all_subtasks,
        }
    )
    ds_splits['perf'] = ds_splits.k / ds_splits.n

    ds_splits = ds_splits.transpose('worker_split', 'split_iter', 'trial', 'subtask')

    ds_boot = xr.Dataset(
        data_vars={
            'k': (['boot_iter', 'subtask', 'trial'], i_boot_to_k),
            'n': (['boot_iter', 'subtask', 'trial'], i_boot_to_n)
        },
        coords={
            'subtask': all_subtasks,
        }
    )
    ds_boot['perf'] = ds_boot.k / ds_boot.n
    ds_boot = ds_boot.transpose('boot_iter', 'trial', 'subtask')


    rvalues = {
        'ds_boot':ds_boot,
        'ds_splits':ds_splits,
    }
    return rvalues


def get_policy_grain_splithalf_and_bootstrap_resampled_data(ds_raw, nsplits):
    """
    Performs split-half and bootstrap resampling of the data, by worker.

    ds_raw: xr.Dataset with mandatory variables:
        stimulus_url: (trial, session)
        worker_id: (trial, session)
        perf: (trial, session)
        subtask: (session)

    ds_splits:
        perf: (worker_split, split_iter, trial, subtask, stimulus_category)
    """

    all_subtasks = sorted(set(ds_raw.subtask.values))
    subtask_to_i_subtask = {sub: i_subtask for (i_subtask, sub) in enumerate(all_subtasks)}
    ntrials = len(ds_raw.trial.values)
    nsubtasks = len(all_subtasks)

    # %% Get workerwise data
    nobjects_per_subtask = 2
    make_subtask_kn_table = lambda: np.zeros((nsubtasks, nobjects_per_subtask, ntrials, 2)).astype(int)

    # k: the number of times that action 1 was chosen
    # n: the number of observations

    worker_to_kn = {}
    nworkers = len(set(ds_raw.worker_id.values))
    import hobj.images.imagesets as imagesets
    ds_meta = imagesets.MutatorHighVarImageset().ds_meta

    for worker_id, ws in tqdm(ds_raw.groupby('worker_id'), total=nworkers):

        worker_to_kn[worker_id] = make_subtask_kn_table()

        perfs = ws.perf.transpose('session', 'trial').values
        subtask_seq = ws.subtask.values

        for i_sess in range(perfs.shape[0]):
            subtask_cur = subtask_seq[i_sess]
            i_subtask = subtask_to_i_subtask[subtask_cur]
            perfs_cur = perfs[i_sess]
            stimseq_cur = ws.stimulus_url.values[i_sess]
            objseq_cur = ds_meta.sel(image_url = stimseq_cur).obj.values
            subtask_objects = sorted(set(objseq_cur))
            assert len(subtask_objects) == nobjects_per_subtask
            i_objseq_cur = [subtask_objects.index(obj) for obj in objseq_cur]
            for t, (i_obj, p) in enumerate(zip(i_objseq_cur, perfs_cur)):
                if p > 0:
                    performed_action1 = i_obj
                else:
                    performed_action1 = 1 - i_obj
                worker_to_kn[worker_id][i_subtask, i_obj, t, 0] += int(performed_action1)
                worker_to_kn[worker_id][i_subtask, i_obj, t, 1] += 1
    all_workers = sorted(worker_to_kn.keys())



    # %% Perform splitting and bootstrapping
    i_split_to_k = []
    i_split_to_n = []

    i_boot_to_k = []
    i_boot_to_n = []

    for i_split in trange(nsplits):
        RS = np.random.RandomState(i_split)
        wcur = RS.permutation(all_workers)
        workers0 = wcur[:len(wcur) // 2]
        workers1 = wcur[len(wcur) // 2:]
        workers_boot = RS.choice(all_workers, replace=True, size=nworkers)

        dat0 = make_subtask_kn_table()
        dat1 = make_subtask_kn_table()
        datboot = make_subtask_kn_table()

        for workers, dat in zip([workers0, workers1, workers_boot], [dat0, dat1, datboot]):
            for w in workers:
                dat += worker_to_kn[w]

        i_split_to_k.append((dat0[..., 0], dat1[..., 0]))
        i_split_to_n.append((dat0[..., 1], dat1[..., 1]))
        i_boot_to_k.append(datboot[..., 0])
        i_boot_to_n.append(datboot[..., 1])

    ds_splits = xr.Dataset(
        data_vars={
            'k': (['split_iter', 'worker_split', 'subtask', 'stimulus_category', 'trial'], i_split_to_k),
            'n': (['split_iter', 'worker_split', 'subtask',  'stimulus_category','trial'], i_split_to_n)
        },
        coords={
            'subtask': all_subtasks,
        }
    )

    ds_splits['action'] = ds_splits.k / ds_splits.n
    ds_splits = ds_splits.transpose('worker_split', 'split_iter', 'trial', 'subtask',  'stimulus_category',)

    ds_boot = xr.Dataset(
        data_vars={
            'k': (['boot_iter', 'subtask',  'stimulus_category', 'trial'], i_boot_to_k),
            'n': (['boot_iter', 'subtask',  'stimulus_category', 'trial'], i_boot_to_n)
        },
        coords={
            'subtask': all_subtasks,
        }
    )
    ds_boot['action'] = ds_boot.k / ds_boot.n
    ds_boot = ds_boot.transpose('boot_iter', 'trial', 'subtask', 'stimulus_category')


    rvalues = {
        'ds_boot':ds_boot,
        'ds_splits':ds_splits,
    }
    return rvalues


if __name__ == '__main__':
    import hobj.behavioral_data.human_data as human_data
    ds_raw = human_data.MutatorHighVarDataset().load_ds_data()
    result = get_policy_grain_splithalf_and_bootstrap_resampled_data(ds_raw = ds_raw, nsplits = 1000)
