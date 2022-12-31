import os

import xarray as xr
import numpy as np

import hobj.benchmarks.experiment_simulation.environment as env_template
from hobj.benchmarks.experiment_simulation.experiment_simulation import ExperimentSimulation
import hobj.utils.file_io as io
import hobj.images.imagesets as imagesets

_loc = os.path.dirname(__file__)


class BinaryClassificationEnvironment(env_template.Environment):
    """
    Implements a binary classification subtask
    """

    def __init__(
            self,
            category0_urls: list,
            category1_urls: list,
            replace=False,  # sample with replacement
    ):
        super().__init__()

        assert isinstance(category0_urls, list)
        assert isinstance(category1_urls, list)

        assert len(category0_urls) == len(category1_urls)
        self.labels = [0 for _ in category0_urls] + [1 for _ in category1_urls]
        self.image_urls = category0_urls + category1_urls
        self.i_seq = None
        self.step = None
        self.replace = replace
        return

    @property
    def current_state_meta(self):
        # Returns meta information for the current environmental state, that will be logged in a simulation. Not allowed to be used by the learner.
        return {}

    def initialize(self, RS: np.random.RandomState):
        if not self.replace:
            self.i_seq = RS.permutation(len(self.image_urls))
        else:
            self.i_seq = RS.choice(len(self.image_urls), replace=True, size=len(self.image_urls))
        self.step = 0

    def sample_image(self):
        icur = self.i_seq[self.step]
        image_url = self.image_urls[icur]
        return image_url

    def provide_feedback(self, action: int):
        assert action in [0, 1]
        icur = self.i_seq[self.step]

        if action == self.labels[icur]:
            reward = 1.
        else:
            reward = -1.

        self.step += 1

        return reward


class MutatorHighVarExperiment(ExperimentSimulation):
    nreps = 500
    ntrials = 100
    imageset = imagesets.MutatorHighVarImageset()
    environment_name_dim = 'subtask'

    @property
    def experiment_name(self):
        return 'MutatorHighVarExperiment'

    def postprocess_behavioral_data(self, ds: xr.Dataset, environment: env_template.Environment):

        """

        :param ds:
        :param environment:
        :return:
        action: (trial, stimulus_category). Probability of taking action 1, given this trial, and this stimulus category.
        """
        ds = ds.transpose('rep', 'trial')

        ds_meta = self.imageset.ds_meta
        nreps = len(ds.rep)
        ntrials = len(ds.trial)
        k_action = np.zeros((nreps, ntrials, 2))
        n_action = np.zeros((nreps, ntrials, 2))
        for i_rep in range(len(ds.rep)):
            objseq = ds_meta.sel(image_url=ds.image_url.values[i_rep]).obj.values
            objects = sorted(np.unique(objseq))
            assert len(objects) == 2

            for i_trial in range(len(ds.trial)):
                i_obj = objects.index(objseq[i_trial])
                reward = ds.reward.values[i_rep, i_trial]

                if i_obj == 1:
                    if reward > 0:
                        took_action1 = True
                    else:
                        took_action1 = False
                elif i_obj == 0:
                    if reward > 0:
                        took_action1 = False
                    else:
                        took_action1 = True
                else:
                    raise Exception

                k_action[i_rep, i_trial, i_obj] = int(took_action1)
                n_action[i_rep, i_trial, i_obj] = 1

        assert np.all(k_action <= n_action)

        k_action = k_action.sum(0)
        n_action = n_action.sum(0)
        prob_action = k_action / n_action
        del ds['action']
        del ds['reward']
        del ds['image_url']
        ds['action'] = (('trial', 'stimulus_category'), prob_action)
        ds['k'] =  (('trial', 'stimulus_category'), k_action)
        ds['n'] = (('trial', 'stimulus_category'), n_action)

        return ds

    @property
    def environments(self):
        if not hasattr(self, '_environments'):
            subtasks = io.load_json(json_path=os.path.join(_loc, 'MutatorHighVarSubtasks.json'))
            """
            subtasks:
                {subtask:obj:[image urls]}
            """

            environments = []
            for subtask in subtasks:
                obj0, obj1 = subtask.split(',')
                urls0, urls1 = subtasks[subtask][obj0], subtasks[subtask][obj1]
                env_cur = BinaryClassificationEnvironment(category0_urls=urls0, category1_urls=urls1)
                env_cur.meta = {'subtask': subtask}
                environments.append(env_cur)
            self._environments = environments
        return self._environments



