import os
hlb_cachedir = os.path.join(os.path.expanduser('~'), 'hlb_cache')

image_cachedir = os.path.join(hlb_cachedir, 'images')
behavior_cachedir = os.path.join(hlb_cachedir, 'behavior')
simulation_cachedir = os.path.join(hlb_cachedir, 'model_simulations')
features_cachedir = os.path.join(hlb_cachedir, 'image_features')

device = 'cpu'

import torch

if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
