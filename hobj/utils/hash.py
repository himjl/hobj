import numpy as np
import hashlib
import PIL.Image


def hash_ndarray(x: np.ndarray):
    assert isinstance(x, np.ndarray)
    bytes = x.tobytes(order="C")
    m = hashlib.sha256(string=bytes)
    digest = m.hexdigest()
    return digest


def hash_image(image: PIL.Image) -> str:
    """
    Hash an image based on its np.uint8 representation.
    :param image:
    :return:
    """
    sha256_hash = hashlib.sha256()

    # Always cast the image to RGBA format
    image = image.convert('RGBA')

    # Convert the image to a numpy array
    image_array = np.array(image).astype(np.uint8)

    # Update the hash with the image array
    sha256_hash.update(image_array.tobytes())

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def hash_string(input_string: str) -> str:
    sha256_hash = hashlib.sha256()

    # Convert string to bytes
    sha256_hash.update(input_string.encode('utf-8'))

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()
