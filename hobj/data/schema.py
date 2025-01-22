import PIL.Image
import pydantic

from hobj.utils.hash import hash_image


# %% Image
class ImageRef(pydantic.BaseModel):
    sha256: str = pydantic.Field(pattern=r'^[a-f0-9]{64}$')

    def __lt__(self, other) -> bool:
        return self.sha256 < other.sha256

    @classmethod
    def from_image(cls, image: PIL.Image):
        return cls(
            sha256=hash_image(image=image)
        )

    def get_image_data(self) -> PIL.Image:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.sha256)
