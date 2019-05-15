# Metropolis-Hastings GANs

This repository contains the source code supporting the paper [Metropolis-Hastings Generative Adversarial Networks](https://arxiv.org/abs/1811.11357):

```
@inproceedings{Turner2019,
    author={Ryan Turner and Jane Hung and Eric Frank and Yunus Saatci and Jason Yosinski},
    title={Metropolis-Hastings Generative Adversarial Networks},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    year={2019}
}
```

For more on this project, see the [Uber AI Labs Blog post](https://eng.uber.com/mh-gan/).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

This code has been tested on `Python 2.7.9` and the exact version of the dependencies are pinned in `requirements.pip`. It has been tested on Mac and Ubuntu.

### Installing

First clone the repo:
```
git clone git@github.com:uber-research/metropolis-hastings-gans.git
```
Inside your virtual environments folder `[ENVS]` make the environment:
```
cd [ENVS]
virtualenv mhgan
source [ENVS]/mhgan/bin/activate
```
Move back into the git repo and run
```
pip install -r requirements.txt
```
You may want to run `pip install -U pip` first if you have an old version of `pip`.

This package also depends on [benchmark tools](https://github.com/rdturnermtl/benchmark_tools). So also checkout that repo and install the package:
```
git clone https://github.com/rdturnermtl/benchmark_tools.git
cd benchmark_tools
pip install -e .
```
The environment should now all be setup to run the experiments.

## Running the experiments

The main experiment script in this project is based on the `pytorch` [DCGAN example](https://github.com/pytorch/examples/tree/master/dcgan). As such, it has the same command line interface.

One must specify a cache directory `[DATA TMP]` for the data to be stored and an output directory `[RESULTS]` for experiment results:
```
cd mhgan
python demo_mhgan.py --dataset cifar10 --dataroot [DATA TMP] --outf [RESULTS] --cuda --manualSeed 123
```
If there is no GPU on the system, drop the `--cuda` flag.

The script creates a new subdirectory to store the results to avoid overwriting the results of previous experiments. For instance: `[RESULTS]/tmpNbBkwr`. The created subdirectory name can be found in `stdout` at the start of `demo_mhgan.py`, which we call `[RESULTS SUBDIR]`.

From the dumps in the `[RESULTS SUBDIR]` directory, summary plots can be made from the scripts with the `.py` files starting with `plot_`.

The files `mh.py` and `classification.py` are designed to have general routines. The other files are scripts specific to this project.

## Plots

For plotting one must install `matplotlib` into the virtual environment. From
the root of the repo run:
```
pip install -r requirements_plots.txt
```

The following functions generate plots based on the csv files dumped by `demo_mhgan.py`:
```
plot_calibration.py
plot_incep_by_epoch.py
plot_score_distn.py
```
These are called in the following way by providing the relevant directories:
```
cd mhgan
python plot_score_distn.py --input [RESULTS SUBDIR] --output [FIGURES]
```

## License

This project is licensed under the Apache 2 License - see the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for details.
