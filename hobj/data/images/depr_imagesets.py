import os

import PIL.Image
import xarray as xr
from typing import List
from hobj.data.store import default_data_store
from pathlib import Path

image_meta_loc = os.path.join(os.path.dirname(__file__), 'image_meta')


# %%
class DeprImageset(object):
    meta_path: str = None

    def load_image(self, image_url: str) -> PIL.Image:
        raise NotImplementedError()
        return image_ref.get_image_data()

    @property
    def ds_meta(self) -> xr.Dataset:
        if not hasattr(self, '_ds_meta'):
            ds_meta = xr.load_dataset(self.meta_path)
            self._ds_meta = ds_meta

        return self._ds_meta

    @property
    def image_urls(self) -> List[str]:
        return list(self.ds_meta.image_url.values)


# %%

class MutatorHighVarDeprImageset(DeprImageset):
    meta_path = os.path.join(image_meta_loc, 'ds_MutatorB2000_Subset128_FullVar_Train.nc')


class MutatorOneshotDeprImageset(DeprImageset):
    meta_path = os.path.join(image_meta_loc, 'ds_MutatorB2000_Oneshot64.nc')

    @property
    def ds_meta(self):
        if not hasattr(self, '_ds_meta'):
            ds_meta = xr.load_dataset(self.meta_path)
            ds_meta['transformation_id'] = (['image_url'], [f'{trans} | {tlevel}' for (trans, tlevel) in zip(ds_meta.transformation.values, ds_meta.transformation_level.values)])
            self._ds_meta = ds_meta

        return self._ds_meta

# %%
if __name__ == '__main__':
    x = MutatorHighVarDeprImageset()
