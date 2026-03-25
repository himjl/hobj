import pydantic

from hobj.data_loaders.images.template import Imageset


class MutatorWarmupAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'Mutator19',
            'Mutator30',
        ],
        pattern=r'^Mutator\d+$'
    )


class MutatorWarmupImageset(Imageset[MutatorWarmupAnnotation]):
    annotation_schema = MutatorWarmupAnnotation


if __name__ == '__main__':
    imageset = MutatorWarmupImageset()
