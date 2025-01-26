
# Coercing human data
from hobj.data.behavior import load_oneshot_behavior
from hobj.data.images import MutatorOneShotImageset

imageset = MutatorOneShotImageset()
oneshot_sessions = load_oneshot_behavior()

from hobj.benchmarks.generalization.task import GeneralizationSessionResult

image_ref_to_transformation_id = {}
for ref in imageset.image_refs:
    annotation = imageset.get_annotation(image_ref=ref)
    transformation_id = f"{annotation.transformation} | {annotation.transformation_level}"
    image_ref_to_transformation_id[ref] = transformation_id

ids = {
    'inplanerotation | 180.0',
    'outplanerotation | 45.0',
    'outplanerotation | 180.0', 'blur | 0.03125',
    'scale | 0.25',
    'noise | 0.5',
    'inplanetranslation | 0.75',
    'inplanetranslation | 0.5',
    'noise | 0.375',
    'backgrounds | 1.0',
    'noise | 0.125',
    'inplanetranslation | 0.25',
    'backgrounds | 0.215443',
    'delpixels | 0.25',
    'delpixels | 0.95',
    'scale | 0.5',
    'combinednat | 1.0', # excluded
    'backgrounds | 0.464159',
    'delpixels | 0.75',
    'original | 0.0', # excluded
    'outplanerotation | 90.0',
    'scale | 1.5',
    'inplanerotation | 135.0', 'blur | 0.015625', 'scale | 0.125', 'contrast | -0.4', 'backgrounds | 0.1', 'inplanerotation | 45.0', 'outplanerotation | 135.0', 'blur | 0.007812', 'inplanerotation | 90.0', 'noise | 0.25',
 'inplanetranslation | 0.125', 'blur | 0.0625', 'contrast | 0.8', 'contrast | 0.4', 'delpixels | 0.5', 'contrast | -0.8'}


for session in oneshot_sessions:
    # Parse raw data by worker