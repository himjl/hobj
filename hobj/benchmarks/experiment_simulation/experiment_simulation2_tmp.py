import hobj.config as config
import hobj.learning_models
import hobj.learning_models.learning_model as lm
import hobj.benchmarks.experiment_simulation.environment_depr as env
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

    def __init__(self):
        return

    @property
    def experiment_name(self):
        raise NotImplementedError
        return 'write_experiment_name_here'

    @property
    def environments(self):
        raise NotImplementedError
        return [env.Environment()]

    def run(self, learner: hobj.learning_models.BinaryLearningModel, seed: Union[type(None), int], show_pbar=True):


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
            learner: hobj.learning_models.BinaryLearningModel,
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
            learner.reset_state()
            environment.initialize(RS=RS)
            for i_trial in range(ntrials):
                if pbar is not None:
                    pbar.update(1)

                image_url = environment.sample_image()
                action = learner.get_response(image=image_url)
                action = int(action)
                reward = environment.provide_feedback(action=action)
                reward = float(reward)
                learner.give_feedback(reward=reward)

                d['image_url'][1][i_trial][i_rep] = image_url
                d['action'][1][i_trial, i_rep] = action
                d['reward'][1][i_trial, i_rep] = reward

                for k in environment.current_state_meta:
                    d[k][1][i_trial, i_rep] = (environment.current_state_meta[k])

        ds = xr.Dataset(d)
        ds = ds.assign_coords(environment.meta)
        ds = ds.assign_coords(image_url=ds['image_url'])
        return ds
