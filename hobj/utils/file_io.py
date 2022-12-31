import urllib.request
import requests
import PIL.Image
from io import BytesIO
import os
import url_normalize
from urllib.parse import urlparse

from tqdm import tqdm

import json


def get_canonical_url(url:str):
    return url_normalize.url_normalize(url = url)

def get_local_save_location(url:str, cachedir:str):
    canonical_url = get_canonical_url(url)
    parsed = urlparse(canonical_url)

    path = parsed.path
    if path.startswith('/'):
        path = path[1:]

    domain = parsed.netloc # domain is the hostname
    save_location = os.path.join(cachedir, domain, path)
    return save_location

def prepare_savepath(savepath):
    savedir = os.path.dirname(savepath)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

def get_image_from_url(image_url:str):
    response = requests.get(image_url)
    img = PIL.Image.open(BytesIO(response.content))
    return img

def download_file(url:str, savepath=None,cachedir:str=None):
    if savepath is None:
        assert cachedir is not None
        savepath = get_local_save_location(url = url, cachedir=cachedir)

    prepare_savepath(savepath)

    def my_hook(t):

        last_b = [0]

        def update_to(b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks transferred so far [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return update_to

    with tqdm(desc = f'downloading {url}') as pbar:
        urllib.request.urlretrieve(url = url, filename=savepath, reporthook=my_hook(pbar))
    return savepath


def load_json(json_path):
    with open(json_path, 'r') as fb:
        val = json.load(fb)
    return val

