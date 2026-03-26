import tarfile
from pathlib import Path

source_dir = Path("/Users/mjl/PycharmProjects/hobj/data")
output_archive = Path("/Users/mjl/Desktop/lee-dicarlo-2023-learning-data.tar.gz")


EXCLUDED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}


def exclude_archive_junk(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Skip common OS-generated metadata files when building the archive."""
    if Path(tarinfo.name).name in EXCLUDED_NAMES:
        return None
    return tarinfo


with tarfile.open(output_archive, "w:gz") as tar:
    tar.add(source_dir, arcname=source_dir.name, filter=exclude_archive_junk)

print(f"Created: {output_archive}")
