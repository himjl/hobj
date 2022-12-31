import os
from typing import List

import xarray as xr
import numpy as np

import hobj.benchmarks.experiment_simulation.environment as env_template
from hobj.benchmarks.experiment_simulation.experiment_simulation import ExperimentSimulation
import hobj.images.imagesets as imagesets

_loc = os.path.dirname(__file__)


class OneshotTestEnvironment(env_template.Environment):
    """
    Trials 0-9 are training trials involving two support images. They are shown an equal amount of times.
    Trials 10-13 are test trials involving transformed versions of the support images.
    Trial 14 is a catch trial.
    Trials 15-18 are test trials involving transformed versions of the support images.
    Trial 19 is a catch trial.

    In all of the test trials, the test images are shown *without replacement*.

    """

    def __init__(
            self,
            support_imageA_url: str,
            support_imageB_url: str,
            test_imageA_urls: List[str],
            test_imageB_urls: List[str],
    ):
        super().__init__()

        assert isinstance(support_imageA_url, str)
        assert isinstance(support_imageB_url, str)
        assert isinstance(test_imageA_urls, list)
        assert isinstance(test_imageB_urls, list)

        assert len(test_imageA_urls) == len(test_imageB_urls), 'Imbalanced test images'
        assert np.all([isinstance(url, str) for url in test_imageA_urls])
        assert np.all([isinstance(url, str) for url in test_imageB_urls])
        assert support_imageA_url not in test_imageA_urls
        assert support_imageB_url not in test_imageB_urls
        assert support_imageA_url != support_imageB_url

        self.support_image_urls = [support_imageA_url, support_imageB_url]
        self.test_imageA_urls = test_imageA_urls
        self.test_imageB_urls = test_imageB_urls
        self.step = None
        return

    def initialize(self, RS: np.random.RandomState):
        self.step = 0

        url_sequence = []
        correct_action_sequence = []

        # Sample the order of the support images during the training phase, balancing the times they appear
        i_training_sequence = RS.permutation([0] * 5 + [1] * 5)
        for i in i_training_sequence:
            url_sequence.append(self.support_image_urls[i])
            correct_action_sequence.append(i)

        # Assemble the test sequence
        test_class_sequence = RS.choice([0, 1], size=10, replace=True)
        testA_seq = RS.permutation(self.test_imageA_urls)
        testB_seq = RS.permutation(self.test_imageB_urls)

        for i_test_trial in np.arange(10):
            class_cur = test_class_sequence[i_test_trial]

            if i_test_trial in [4, 9]:
                # Catch trial
                url_sequence.append(self.support_image_urls[class_cur])
                correct_action_sequence.append(class_cur)
                continue

            if class_cur == 0:
                url_sequence.append(testA_seq[i_test_trial])
            else:
                url_sequence.append(testB_seq[i_test_trial])

            correct_action_sequence.append(class_cur)

        self.url_seq = url_sequence
        self.correct_action_seq = correct_action_sequence


    def sample_image(self):

        image_url = self.url_seq[self.step]
        return image_url

    def provide_feedback(self, action: int):

        correct_action = self.correct_action_seq[self.step]
        assert action in [0, 1]
        assert correct_action in [0, 1]

        if action == correct_action:
            reward = 1.
        else:
            reward = -1.

        self.step += 1

        return reward


class MutatorOneShotExperiment(ExperimentSimulation):
    nreps = 500  # per subtask (n=32)
    ntrials = 20
    imageset = imagesets.MutatorOneshotImageset()
    environment_name_dim = 'subtask'

    @property
    def experiment_name(self):
        return 'MutatorOneShotExperiment'

    @property
    def ds_meta(self):
        if not hasattr(self, '_ds_meta'):
            self._ds_meta = self.imageset.ds_meta
        return self._ds_meta

    @property
    def url_to_transformation_id(self):
        if not hasattr(self, '_url_to_transformation_id'):
            self._url_to_transformation_id = {url: trans for (url, trans) in zip(self.ds_meta.image_url.values, self.ds_meta.transformation_id.values)}
        return self._url_to_transformation_id

    @property
    def transformation_ids(self):
        if not hasattr(self, '_transformation_ids'):
            self._transformation_ids = sorted(np.unique(self.ds_meta.transformation_id.values))
        return self._transformation_ids

    def get_i_transformation_id_from_url(self, url: str):
        transformation_id = self.url_to_transformation_id[url]
        if not hasattr(self, '_transformation_id_to_i'):
            self._transformation_id_to_i = {transformation_id: i for (i, transformation_id) in enumerate(self.transformation_ids)}
        i_transformation_id = self._transformation_id_to_i[transformation_id]
        return i_transformation_id

    def postprocess_behavioral_data(self, ds: xr.Dataset, environment: env_template.Environment):

        """

        :param ds: [rep, trial]
        :param environment:
        :return:

        k: (transformation_id)
        n: (transformation_id)

        k_train: (train_trial)
        n_train: (train_trial)

        k_test: (test_trial)
        n_test: (test_trial)

        k_catch: (catch_trial)
        n_catch: (catch_trial)

        """
        assert set(ds.dims) == {'rep', 'trial'}
        assert 'reward' in ds.data_vars
        ds['perf'] = ds.reward > 0
        ds_table = xr.Dataset(
            dict(
                k=(['transformation_id'], np.zeros(len(self.transformation_ids), dtype=int)),
                n=(['transformation_id'], np.zeros(len(self.transformation_ids), dtype=int)),
                k_train=(['train_trial'], np.zeros(10, dtype=int)),
                n_train=(['train_trial'], np.zeros(10, dtype=int)),
                k_catch=(['catch_trial'], np.zeros(3, dtype=int)),
                n_catch=(['catch_trial'], np.zeros(3, dtype=int)),
            ),
            coords={
                'transformation_id': self.transformation_ids,
                'train_trial': np.arange(10),
                'catch_trial': [9, 14, 19],
            }
        )

        perf_dat = ds.transpose('rep', 'trial').perf.values
        url_dat = ds.transpose('rep', 'trial').image_url.values
        for i_rep in ds.rep.values:
            for i_trial in range(self.ntrials):
                url = url_dat[i_rep, i_trial]
                perf = int(perf_dat[i_rep, i_trial])

                if i_trial < 10:
                    ds_table.k_train.values[i_trial] += perf
                    ds_table.n_train.values[i_trial] += 1
                if i_trial == 9:
                    ds_table.k_catch.values[0] += perf
                    ds_table.n_catch.values[0] += 1
                if i_trial == 14:
                    ds_table.k_catch.values[1] += perf
                    ds_table.n_catch.values[1] += 1
                if i_trial == 19:
                    ds_table.k_catch.values[2] += perf
                    ds_table.n_catch.values[2] += 1

                i_transformation_id = self.get_i_transformation_id_from_url(url)

                ds_table.k.values[i_transformation_id] += perf
                ds_table.n.values[i_transformation_id] += 1

        return ds_table

    @property
    def environments(self):
        subtasks = ['MutatorB2000_2292,MutatorB2000_2444',
                    'MutatorB2000_138,MutatorB2000_2344',
                    'MutatorB2000_1251,MutatorB2000_953',
                    'MutatorB2000_3043,MutatorB2000_694',
                    'MutatorB2000_3496,MutatorB2000_496',
                    'MutatorB2000_1219,MutatorB2000_296',
                    'MutatorB2000_1825,MutatorB2000_2757',
                    'MutatorB2000_3077,MutatorB2000_4703',
                    'MutatorB2000_270,MutatorB2000_3615',
                    'MutatorB2000_3066,MutatorB2000_3585',
                    'MutatorB2000_2139,MutatorB2000_746',
                    'MutatorB2000_116,MutatorB2000_2365',
                    'MutatorB2000_2130,MutatorB2000_4628',
                    'MutatorB2000_462,MutatorB2000_926',
                    'MutatorB2000_2304,MutatorB2000_3733',
                    'MutatorB2000_1363,MutatorB2000_3278',
                    'MutatorB2000_4049,MutatorB2000_663',
                    'MutatorB2000_2722,MutatorB2000_3527',
                    'MutatorB2000_2832,MutatorB2000_801',
                    'MutatorB2000_1258,MutatorB2000_3123',
                    'MutatorB2000_1865,MutatorB2000_613',
                    'MutatorB2000_1164,MutatorB2000_2106',
                    'MutatorB2000_1229,MutatorB2000_1280',
                    'MutatorB2000_1767,MutatorB2000_2122',
                    'MutatorB2000_2198,MutatorB2000_701',
                    'MutatorB2000_3636,MutatorB2000_4305',
                    'MutatorB2000_3035,MutatorB2000_46',
                    'MutatorB2000_3601,MutatorB2000_4792',
                    'MutatorB2000_2092,MutatorB2000_288',
                    'MutatorB2000_1424,MutatorB2000_2314',
                    'MutatorB2000_3308,MutatorB2000_3525',
                    'MutatorB2000_2909,MutatorB2000_4256']

        # Replicate the same kinds of trials done in humans
        if not hasattr(self, '_environments'):
            ds_meta = imagesets.MutatorOneshotImageset().ds_meta

            obj_to_support_url = {}
            obj_to_test_urls = {}
            for obj, ds_obj in ds_meta.groupby('obj'):
                ds_support = ds_obj.sel(image_url=ds_obj.transformation == 'original')
                ds_test = ds_obj.sel(image_url=ds_obj.transformation != 'original')
                assert len(ds_support.image_url) + len(ds_test.image_url) == len(ds_obj.image_url)
                assert len(ds_support.image_url) == 1
                support_url = ds_support.image_url.values[0]
                test_urls = ds_test.image_url.values
                obj_to_support_url[obj] = str(np.array(support_url))
                obj_to_test_urls[obj] = list(np.array(test_urls))

            environments = []
            for subtask in subtasks:
                objA, objB = subtask.split(',')
                support_imageA_url = obj_to_support_url[objA]
                support_imageB_url = obj_to_support_url[objB]
                test_imageA_urls = obj_to_test_urls[objA]
                test_imageB_urls = obj_to_test_urls[objB]

                env_cur = OneshotTestEnvironment(
                    support_imageA_url=support_imageA_url,
                    support_imageB_url=support_imageB_url,
                    test_imageA_urls=test_imageA_urls,
                    test_imageB_urls=test_imageB_urls,
                )
                environments.append(env_cur)
            self._environments = environments
        return self._environments
