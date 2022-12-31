import xarray as xr
import hobj.behavioral_data.template.sr_data as template
import os
import numpy as np
import collections
from tqdm import trange

_loc = os.path.dirname(__file__)

class MutatorHighVarDataset(template.SR_Dataset):
    """
    "Experiment 1" from the paper.
    """
    data_url = 'https://hlbdatasets.s3.amazonaws.com/behavioral_data/ds_human_raw_MutatorHighVar.nc'
    probe_trials = [25, 51, 77, 103]

    def apply_experiment_specific_filtering(self, ds_flat):
        # Remove data variables
        ds_flat = ds_flat[['perf', 'reaction_time_msec']]

        # Assign subtask names
        import hobj.images.imagesets as imagesets
        ds_meta = imagesets.MutatorHighVarImageset().ds_meta
        urls = ds_flat.stimulus_url.transpose('session', 'trial').values
        url_to_obj = {url: obj for url, obj in zip(ds_meta.image_url.values, ds_meta.obj.values)}
        sseq = []
        for i in range(urls.shape[0]):
            urlseq = urls[i]
            urlseq = [u for u in urlseq if 'orange' not in u and 'blue' not in u]
            objects = [url_to_obj[url] for url in urlseq]
            objects = sorted(set(objects))
            assert len(objects) == 2
            subtask = ','.join(objects)

            sseq.append(subtask)
        ds_flat = ds_flat.assign_coords(subtask = (['session'], sseq))
        return ds_flat

    def load_ds_worker_table(self):
        """
        A worker-wise table of the human data.

        :return:
            ds_data:
                k: (worker_id, subtask, trial)
                n:  (worker_id, subtask, trial)
        """

        ds_data = self.load_ds_data()

        all_subtasks = sorted(np.unique(ds_data.subtask.values))
        assert len(all_subtasks) == 64

        ntrials = len(ds_data.trial.values)
        subtask_to_workers = collections.defaultdict(list)
        for sub, worker in zip(ds_data.subtask.values, ds_data.worker_id.values):
            subtask_to_workers[sub].append(worker)
        max_reps_per_subtask = max(len(v) for v in subtask_to_workers.values())
        assert max_reps_per_subtask == 50

        all_workers = sorted(list(set(ds_data.worker_id.values)))
        nworkers = len(all_workers)

        k = np.zeros((nworkers, len(all_subtasks), ntrials), dtype=int)
        n = np.zeros((nworkers, len(all_subtasks), ntrials), dtype=int)

        ds_data = ds_data.transpose('session', 'trial')
        for i_session in trange(len(ds_data.session.values), desc='unpacking data', disable = True):
            subtask = ds_data.subtask.values[i_session]
            worker_id = ds_data.worker_id.values[i_session]

            i_subtask = all_subtasks.index(subtask)
            i_worker = all_workers.index(worker_id)
            perf_seq = ds_data.perf.values[i_session]
            k[i_worker, i_subtask, :] = perf_seq
            n[i_worker, i_subtask, :] += 1

        ds_table = xr.Dataset(
            data_vars=dict(
                k=(['worker_id', 'subtask', 'trial'], k),
                n=(['worker_id', 'subtask', 'trial'], n),
            ),
            coords=dict(
                worker_id=all_workers,
                subtask=all_subtasks,
                trial=np.arange(ntrials),
            )
        )

        assert np.max(ds_table.n == 1)
        assert np.max(ds_table.k == 1)

        return ds_table


class MutatorOneShotGeneralizationDataset(template.SR_Dataset):

    """
    "Experiment 2" from the paper.
    """

    data_url = 'https://hlbdatasets.s3.amazonaws.com/behavioral_data/ds_human_raw_MutatorOneShotGeneralization.nc'

    def apply_experiment_specific_filtering(self, ds_flat):

        import numpy as np
        import hobj.images.imagesets as imagesets
        ds_meta = imagesets.MutatorOneshotImageset().ds_meta
        image_url_to_transformation = {url:trans for url, trans in zip(ds_meta.image_url.values, ds_meta.transformation.values)}
        image_url_to_transformation_level = {url:trans_level for url, trans_level in zip(ds_meta.image_url.values, ds_meta.transformation_level.values)}

        valid_sess = (((ds_flat.stimulus_url == '').sum('trial') == 0))
        ds_flat = ds_flat.sel(session=valid_sess)



        get_trans_level = np.vectorize(lambda url: float(image_url_to_transformation_level[url]))
        get_trans = np.vectorize(lambda url: str(image_url_to_transformation[url]))

        trans = get_trans(ds_flat.stimulus_url.values)
        trans_level = get_trans_level(ds_flat.stimulus_url.values)

        trans_ids = []
        for i in range((trans.shape[0])):
            cur = []
            for j in range((trans.shape[1])):
                t = trans[i, j]
                l = trans_level[i, j]
                cur.append(' | '.join([str(t), str(l)]))
            trans_ids.append(cur)


        ds_flat = ds_flat.assign_coords(
            transformation_type=(ds_flat.stimulus_url.dims, trans),
            transformation_level=(ds_flat.stimulus_url.dims, trans_level),
            transformation=(ds_flat.stimulus_url.dims, trans_ids)
        )

        ds_flat = ds_flat.transpose('trial', 'session')
        ds_flat.transformation.values[9] = 'probe | 10'
        ds_flat.transformation.values[14] = 'probe | 15'
        ds_flat.transformation.values[19] = 'probe | 20'

        # Remove data variables
        ds_flat = ds_flat[['perf', 'reaction_time_msec']]
        return ds_flat

if __name__ == '__main__':
    dataset = MutatorHighVarDataset()
    ds_data_example = dataset.load_ds_data()