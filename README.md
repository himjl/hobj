# `hobj`: human object learning benchmarks

[![CI](https://github.com/himjl/hobj/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/hobj/actions/workflows/ci.yml)

This repository contains benchmarks for comparing models of object learning against measurements of human behavior, from Lee and DiCarlo 2023 (["How well do rudimentary plasticity rules predict adult visual object learning?"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011713)). It also lets you download the raw data and images from the experiments in the paper.

<div style="text-align: center;">
  <img src="site/readme_images/human_learning_curves.svg" alt="Alt text" >
</div>

If you just want to download the raw data and images without using the `hobj` library, check out the [OSF repository](https://osf.io/pj6wm/files/osfstorage) for this project.

## Quickstart

### Install

The `hobj` package works for Python >=3.12. After cloning this repository on your machine, navigate to this directory in your shell and run:

On first use, the packaged dataset is downloaded automatically from the OSF
repository into a versioned cache directory under `~/.hobj_cache`. The default
path is `~/.hobj_cache/pj6wm-v1/data`. In total, `hobj` takes up around ~1 GB
of space on your computer.


### Using `hobj` to comparing a linear learner against human learning data 

The template script below shows you how you can run a benchmark on a linear learning model based on your image encoding model. 

All you need to do is have a way to process a `PIL.Image` into a vector of image features (as an `np.ndarray`). There are ~18,000 images that you'd need to compute image features for.    

```python
import hobj
import numpy as np 

# Compute your features for the images 
my_image_features: dict[str, np.ndarray] = {}
for image_id in hobj.list_image_ids():
    image = hobj.load_image(image_id=image_id) # PIL.Image
    
    # Compute your features here:  
    my_image_features[image_id] = ... # replace right hand side with your image-computable model

# Assemble the learning model:
model = hobj.create_linear_learner(
    image_id_to_features=my_image_features,
    update_rule_name='Square', # "Square", "Perceptron", "Hinge", "MAE", "Exponential", "CE", "REINFORCE",
    alpha=1, # learning rate between [0, 1]
)

# Load the benchmark:
benchmark = hobj.MutatorHighVarBenchmark()  # or hobj.MutatorOneshotBenchmark()

# Score the model:
result = benchmark.score_model(model)

# Print its score and its CI:
print(result.msen, result.msen_CI95)

# You can also check out more granular statistics of the model's behavior, like its learning curves: 
# print(result.model_statistics)
```


For more details (e.g., how to load the raw behavioral data or images in Python), check out the Jupyter notebooks in `examples/`.

To use a different location, pass `cachedir=...` to a data loader or benchmark
constructor, or prefetch manually with `hobj-download-data --cachedir /path/to/data`.

## Contact 
If you have any questions, need help, or experience a bug, please don't hesitate to email me ([mil@mit.edu](mailto:name@example.com)), or open an issue on this repo!



## Changes to codebase since publication
This codebase was overhauled in 2026 to improve its accessibility, performance, and quality. Along the way, minor changes to the statistical analysis procedure were introduced, along with changes to the names of the original filenames (see [changelist](site/changelist.md)). To see the codebase at the time of publication, check out the repo with the `v1` tag [here](https://github.com/himjl/hobj/releases/tag/v1).


## Citation

```
@article{lee2023well,
  title={How well do rudimentary plasticity rules predict adult visual object learning?},
  author={Lee, Michael J and DiCarlo, James J},
  journal={PLOS Computational Biology},
  volume={19},
  number={12},
  pages={e1011713},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
