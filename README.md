# `hobj`: human object learning benchmarks

[![CI](https://github.com/himjl/hobj/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/hobj/actions/workflows/ci.yml)

This repository contains benchmarks for comparing models of object learning against measurements of human behavior, from Lee and DiCarlo 2023 (["How well do rudimentary plasticity rules predict adult visual object learning?"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011713)).

<div style="text-align: center;">
  <img src="site/readme_images/human_learning_curves.svg" alt="Alt text" >
</div>


### Example usage

Benchmarking a learning model simply requires the user to define a `hobj.learning_models.BinaryLearningModel` subclass.
Then, an instance of that subclass may be passed in as an argument to a benchmark object:

```python
import hobj

# Load model: 
model = hobj.learning_models.RandomGuesser() # A subclass of BinaryLearningModel

# Load benchmark:
benchmark = hobj.benchmarks.MutatorHighVarBenchmark() # Try benchmark 2: hobj.benchmarks.MutatorOneshotBenchmark()
result = benchmark(model)

# Print the score and its CI:
print(result.msen, result.msen_CI95)
```

For more details, check out `examples/`.

### Installation

The `hobj` package works for Python >=3.11. After cloning this repository on your machine, navigate to this directory in your shell and run:

``` pip install -e .```


### Changes to codebase since publication
This codebase was refactored in January 2025 to improve the performance and quality of the code, and is now designated as `v2`. Along the way, minor refinements to the statistical analysis of the original codebase were introduced (see [changelist](site/changelist.md)). To see the codebase at the time of publication, check out the repo with the `v1` tag [here](https://github.com/himjl/hobj/releases/tag/v1).

### Citation

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
}```
