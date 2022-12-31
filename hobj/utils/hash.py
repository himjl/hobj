import numpy as np
import hashlib


def hash_ndarray(x:np.ndarray):
    assert isinstance(x, np.ndarray)
    bytes = x.tobytes(order = "C")
    m = hashlib.sha256(string=bytes)
    digest = m.hexdigest()
    return digest

def hash_image(image:np.ndarray):
    assert isinstance(image, np.ndarray)
    assert 2<= len(image.shape) <= 3
    image = image.astype(np.uint8)
    return hash_ndarray(image)

def hash_string(x:str):
    return hash_ndarray(np.fromstring(str(x), dtype=np.uint8))