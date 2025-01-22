import os

import xarray as xr
import numpy as np

import hobj.benchmarks.experiment_simulation.environment as env_template
import hobj.utils.file_io as io
import hobj.images.imagesets as imagesets
import hobj.config as config
import hobj.learning_models.learning_model as lm
import hobj.benchmarks.experiment_simulation.environment as env
import hobj.utils.file_io as io
import os
import xarray as xr
import numpy as np
from tqdm import tqdm
from typing import Union


class ExperimentSimulation(object):
    """
    Wraps a set of environments, and simulates them self.nreps times each
    """
    nreps = None
    ntrials = None
    environment_name_dim = 'environment'

    def __init__(self, cachedir=config.simulation_cachedir):
        self.cachedir = cachedir
        return

    @property
    def experiment_name(self):
        raise NotImplementedError
        return 'write_experiment_name_here'

    @property
    def environments(self):
        raise NotImplementedError
        return [env.Environment()]

    def run(self, learner: lm.LearningModel, seed: Union[type(None), int], force_recompute=False, show_pbar=True):

        savepath = os.path.join(self.cachedir, self.experiment_name, 'seed_' + str(seed), f'ds_{learner.learner_id}.nc')
        if os.path.exists(savepath) and not force_recompute:
            ds = xr.load_dataset(savepath)
            return ds

        dlist = []
        RS = np.random.RandomState(seed=seed)
        if show_pbar:
            pbar = tqdm(total=self.nreps * self.ntrials * len(self.environments), desc='simulation trials')
        else:
            pbar = None

        for environment in self.environments:
            ds = self.simulate_environment(learner=learner, environment=environment, RS=RS, nreps=self.nreps, ntrials=self.ntrials, pbar=pbar)
            ds = self.postprocess_behavioral_data(ds=ds, environment=environment)
            dlist.append(ds)

        ds = xr.concat(dlist, dim=self.environment_name_dim, )
        io.prepare_savepath(savepath)
        ds.to_netcdf(savepath)
        return ds

    def postprocess_behavioral_data(self, ds: xr.Dataset, environment: env.Environment):
        """
        Called following the simulation of a single environment. Allows for postprocessing of the behavioral data.
        :param ds:
        :param environment:
        :return:
        """
        return ds

    @staticmethod
    def simulate_environment(
            learner: lm.LearningModel,
            environment: env.Environment,
            RS: np.random.RandomState,
            nreps: int,
            ntrials: int,
            pbar: tqdm = None,
    ):

        d = {
            'image_url': (['trial', 'rep', ], [['' for _ in range(nreps)] for _ in range(ntrials)]),
            'action': (['trial', 'rep', ], np.zeros((ntrials, nreps))),
            'reward': (['trial', 'rep', ], np.zeros((ntrials, nreps))),
        }

        for i_rep in range(nreps):
            learner.reset()
            environment.initialize(RS=RS)
            for i_trial in range(ntrials):
                if pbar is not None:
                    pbar.update(1)

                image_url = environment.sample_image()
                action = learner.respond(image_url=image_url)
                action = int(action)
                reward = environment.provide_feedback(action=action)
                reward = float(reward)
                learner.learn(reward=reward)

                d['image_url'][1][i_trial][i_rep] = image_url
                d['action'][1][i_trial, i_rep] = action
                d['reward'][1][i_trial, i_rep] = reward

                for k in environment.current_state_meta:
                    d[k][1][i_trial, i_rep] = (environment.current_state_meta[k])

        ds = xr.Dataset(d)
        ds = ds.assign_coords(environment.meta)
        ds = ds.assign_coords(image_url=ds['image_url'])
        return ds


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



