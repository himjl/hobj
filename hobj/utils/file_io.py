import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


def unzip_file(zip_path: Path, output_dir: Path) -> None:
    """
    Unzip a file to a given directory.
    :param zip_path:
    :param output_dir:
    :return:
    """

    if not zip_path.as_posix().endswith(".zip"):
        raise ValueError(f"Expected a .zip file, got {zip_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file {zip_path} does not exist.")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def download_file(url: str, output_path: Path) -> None:
    # Send a GET request to fetch the file
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    # Create the output directory if it does not exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with tqdm(
        total=total_size_in_bytes,
        unit="B",
        unit_scale=True,
        disable=False,
        desc="Download progress",
    ) as progress_bar:
        with open(output_path.as_posix(), "wb") as file:
            # Iterate over the response data in chunks and write to file
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
            file.flush()


def get_bytes_size(num_bytes: int, output_units: str = None) -> (float, str):
    """

    :param num_bytes:
    :param output_units: If None, automatically selected
    :return:
    """

    unit_conversion = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}

    # Automatically select the byte unit (KB, MB, etc.)
    if output_units is None:
        if num_bytes == 0:
            output_units = "B"
        elif num_bytes < 1e6:
            output_units = "KB"
        elif num_bytes < 1e9:
            output_units = "MB"
        else:
            output_units = "GB"

    # Convert the size to the selected unit
    size = num_bytes / unit_conversion[output_units]
    return size, output_units


def download_json(url: str) -> Any:
    """
    Download a JSON file from a URL.
    :param url:
    :return:
    """
    response = urllib.request.urlopen(url)
    if response.status != 200:
        raise ValueError(f"Could not load JSON data from {url}", response)
    data = response.read().decode("utf-8")
    json_data = json.loads(data)

    return json_data
