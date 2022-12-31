import os

import PIL.Image

import hobj.utils.file_io as io
import hobj.config as config

def get_image_location(image_url: str, cachedir=config.image_cachedir):
    """
    Return the location of the image at the specified image_url on disk. If it is not present on disk yet, download it.
    :param image_url: URL of the image to download.
    :param cachedir: Directory to cache the image in.
    :return:
    """
    save_location = io.get_local_save_location(url=image_url, cachedir=cachedir)

    if not os.path.exists(save_location):
        io.prepare_savepath(save_location)
        image = io.get_image_from_url(image_url=image_url)
        image.save(save_location)
        print('Downloaded image to {}'.format(save_location))
    return save_location


def get_image(image_url: str, cachedir=config.image_cachedir):
    """
    Download an image from a URL and return it as a PIL.Image.
    If cached, return the cached image.
    :param image_url: URL of the image to retrieve.
    :param cachedir: Directory to cache the image in.
    :return:
    """

    save_location = get_image_location(image_url=image_url, cachedir=cachedir)
    image = PIL.Image.open(save_location)
    return image
