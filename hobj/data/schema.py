import PIL.Image
import pydantic

from hobj.utils.hash import hash_image
from hobj.data.store import default_data_store


# %% Image
class ImageRef(pydantic.BaseModel):
    sha256: str = pydantic.Field(pattern=r'^[a-f0-9]{64}$')

    @classmethod
    def from_image(cls, image: PIL.Image, register: bool = True):
        if register:
            return default_data_store.register_image(image=image)

        return cls(
            sha256=hash_image(image=image)
        )

    def get_image_data(self) -> PIL.Image:
        return default_data_store.load_image(image_ref=self)

    def __lt__(self, other) -> bool:
        return self.sha256 < other.sha256

    def __hash__(self):
        return hash(self.sha256)

# %% Behavior
