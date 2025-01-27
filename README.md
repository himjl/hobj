# `hobj`: human object learning benchmarks

[![CI](https://github.com/himjl/hobj/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/hobj/actions/workflows/ci.yml)

This repository contains benchmarks for comparing models of object learning against  measurements of human behavior, from Lee and DiCarlo 2023 (["How well do rudimentary plasticity rules predict adult visual object learning?"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011713)).

<div style="text-align: center;">
  <img src="site/readme_images/human_learning_curves.svg" alt="Alt text" >
</div>

### Installation
This package is supported for Python >=3.11. After cloning `hobj` on your machine, simply navigate to this directory in your shell, then run

``` pip install hobj```

To see how to view behavioral data and/or score an example learning model, check out `examples/`.


### Changes to codebase since publication
This codebase was refactored in 2025, improving the performance and quality of the code, as well as introducing changes to the statistics (see [changelist](site/changelist.md)). To see the codebase that reflects its contents at the time of publication, check out the repo with the `v1` tag.
