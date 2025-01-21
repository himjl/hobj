# Human object learning benchmarks

This repository contains benchmarks for comparing models of object learning against psychometric measurements of human behavior. 

# Installation

We developed this repository using Python 3.9. Follow the steps below to use it: 

1. Begin by cloning the `hobj` repository to your local machine. 
2. Make and activate a new conda environment. 
3. Install the `hobj` package itself. To do so, `cd` to the `hobj` directory (the top-level one), then run:

```bash
pip install -e .
```
4. Then, install the following dependencies:
```bash
conda install -c conda-forge xarray dask netCDF4 bottleneck
```
5. Install [PyTorch](https://pytorch.org). If you are using a computer without a GPU, you can run the command:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### Download all images (recommended)

The code in this repository works without this step. But to save time, it is recommended to download the images in a batch.
They are located [here](https://hlbdatasets.s3.amazonaws.com/images/LeeDiCarlo_hobj_Images.zip).

Once it is downloaded, unzip it (it should turn into an `images` folder). Move that `images` folder to `~/hlb_cache/`.

# Usage

To see how to view the raw behavioral data and/or score an example learning model, check out the examples in
`examples/`.


# Paper
This codebase originally accompanied the 2023 paper ["How well do rudimentary plasticity rules predict adult visual object learning?"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011713).
Since that point, this codebase has been refactored and streamlined, introducing breaking changes to the API. However, to see the codebase that reflects its content at the time of publication, check out the `v1` tag.
