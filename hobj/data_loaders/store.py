
import hobj.config


import mref
__all__ = [
    'default_data_store'
]


# %%
# Default data store
default_data_store = mref.FileSystemStorage(cachedir=hobj.config.cachedir)
