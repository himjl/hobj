import hobj.config as config
import os
import xarray as xr
import hobj.images.load_image as load_image

image_meta_loc = os.path.join(os.path.dirname(__file__), 'image_meta')


class Imageset(object):
    meta_path: str = None

    def __init__(self, cachedir=config.image_cachedir):
        self.cachedir = cachedir
        return

    def load_image(self, image_url: str):
        return load_image.get_image(image_url=image_url, cachedir=self.cachedir)

    @property
    def ds_meta(self):
        if not hasattr(self, '_ds_meta'):
            ds_meta = xr.load_dataset(self.meta_path)
            self._ds_meta = ds_meta

        return self._ds_meta

    @property
    def image_urls(self):
        return list(self.ds_meta.image_url.values)


class WarmupImageset(Imageset):
    meta_path = os.path.join(image_meta_loc, 'ds_MutatorB0_BurnIn.nc')


class MutatorHighVarImageset(Imageset):
    meta_path = os.path.join(image_meta_loc, 'ds_MutatorB2000_Subset128_FullVar_Train.nc')


class MutatorOneshotImageset(Imageset):
    meta_path = os.path.join(image_meta_loc, 'ds_MutatorB2000_Oneshot64.nc')

    @property
    def ds_meta(self):
        if not hasattr(self, '_ds_meta'):
            ds_meta = xr.load_dataset(self.meta_path)
            ds_meta['transformation_id'] = (['image_url'], [f'{trans} | {tlevel}' for (trans, tlevel) in zip(ds_meta.transformation.values, ds_meta.transformation_level.values)])
            self._ds_meta = ds_meta

        return self._ds_meta
