import xarray as xr
import hobj.utils.hash
import hobj.utils.file_io as io
import hobj.config as config
import os

class SR_Dataset(object):
    """
        - Wraps data on disk
        - Ensures data is valid

    """

    """
    xr.DataArray

    mandatory coords: 
        stimulus_url: (trial, session)
        worker_id: (trial, session) 
        timestamp_start: (trial, session)

    mandatory data variables: 
        perf: (trial, session) 0s or 1s
    """

    data_url = None
    probe_trials = None

    def __init__(self, cachedir = config.behavior_cachedir):
        self.cachedir = cachedir
        self.savepath = io.get_local_save_location(url = self.data_url, cachedir = self.cachedir)

    @property
    def name(self):
        return str(type(self).__name__)


    def load_raw_ds(self):

        if not os.path.exists(self.savepath):
            io.download_file(url = self.data_url, cachedir=self.cachedir)

        ds_data = xr.load_dataset(self.savepath)
        return ds_data

    def load_ds_data(self):
        ds_data = self.load_raw_ds()
        mandatory_coords = [
            'stimulus_url',
            'worker_id',
            'timestamp_start',
        ]
        mandatory_vars = [
            'perf',
        ]

        mandatory_dims = ['trial', 'session']

        for d in mandatory_dims:
            assert d in ds_data.dims, d

        for v in mandatory_vars:
            assert v in ds_data.data_vars, v
        for c in mandatory_coords:
            assert c in ds_data.coords, c

        # ds_flat = ds_data.reset_coords()[mandatory_coords + mandatory_vars]
        ds_flat = ds_data.reset_coords().set_coords(mandatory_coords)
        ds_flat = ds_flat.assign_coords(trial = range(len(ds_flat.trial.values)))

        hash = ''
        for v in mandatory_vars + mandatory_coords:
            hash += hobj.utils.hash.hash_ndarray(ds_flat[v].values)
        ds_flat = ds_flat.assign_coords(behavioral_hash=hash)

        if self.probe_trials is not None:
            ds_flat = ds_flat.assign_coords(
                probe_perf=ds_flat.perf.sel(trial=self.probe_trials).mean('trial')
            )
            ds_flat = ds_flat.sel(trial=[v for v in ds_flat.trial.values if v not in self.probe_trials])
            ds_flat = ds_flat.assign_coords(trial=range(len(ds_flat.trial.values)))

        ds_flat = self.apply_experiment_specific_filtering(ds_flat = ds_flat)
        return ds_flat

    def apply_experiment_specific_filtering(self, ds_flat):
        return ds_flat
