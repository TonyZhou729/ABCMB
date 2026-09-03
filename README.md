<h1 align="center">
ABCMB<!-- omit from toc -->
</h1>
<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2602.15104%20-green.svg)](https://arxiv.org/abs/2602.15104)
[![Run Tests](https://github.com/TonyZhou729/ABCMB/actions/workflows/accuracy.yml/badge.svg)](https://github.com/TonyZhou729/ABCMB/actions/workflows/accuracy.yml)
<!--[![arXiv](https://img.shields.io/badge/arXiv-2408.14538%20-green.svg)](https://arxiv.org/abs/2408.14538) -->

</h4>

Autodifferentiable Boltzmann solver for the CMB (ABCMB) is a Python+JAX package for differentiable computation of the Cosmic Microwave Background.  ABCMB is **complete to linear order** in $\Lambda\rm{CDM}$ cosmology.  It computes the matter and CMB power spectra and includes effects like lensing, massive neutrinos, and a state-of-the-art treatment of the physics of recombination through the companion code [HyRex](https://github.com/TonyZhou729/HyRex).

## User Installation
ABCMB is pip installable!  Just run
```
pip install ABCMB
```
We recommend always doing so in a conda environment, preferably even a clean one.

If you'd like to clone the repo instead, after cloning you can run
```
pip install .
```
from the code directory. 

Note that both methods of installing will automatically attempt to install JAX for CPU; to install for GPU, refer to the [JAX documentation](https://docs.jax.dev/en/latest/installation.html) for a quick JAX installation guide.

## Developer Installation

If you would like to contribute back to ABCMB, please do the following.
(Development requires Python 3.11+)

* Fork the repository (see [here](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo)) and get a local copy on your computer
* Create a branch for your feature and switch to it (```git switch -c <feature-name>```)
* Set up the developer environment:
```
python -m venv .venv
source .venv/bin/activate
pip install -r pytest_requirements.txt   # runtime and test dependencies
pip install -r requirements-dev.txt      # ruff, pre-commit
pip install -e . --no-deps               # makes `abcmb` importable from anywhere
pre-commit install
```

Optionally, check the setup by running the hooks over the whole repository
once, so your first commit holds no surprises:
```
pre-commit run --all-files
```
* Make the changes
* Verify them with the linter and the test suite, which is what CI runs:
```
./check.sh
```
* Once finalized, push the changes to a remote branch:
```
git push -u origin <feature-name>
```
* Draft a pull request against `main` (see [here](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request))

## Examples
We have included several pedagogical jupyter notebooks to walk you through how to get started with ABCMB in our [example_notebooks](https://github.com/TonyZhou729/ABCMB/tree/main/example_notebooks) folder.  We suggest you start with [ABCMB_basics](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_basics.ipynb) to get a sense of how to run the code.  If you'd like to add new physics to ABCMB, check out [ABCMB_Fluids](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_Fluids.ipynb).  If you'd like to run ABCMB with the Big Bang Nucleosynthesis (BBN) code [LINX](https://github.com/cgiovanetti/LINX/tree/main) to do BBN+CMB joint analyses, check out [ABCMB_with_LINX](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_with_LINX.ipynb).

## Issues
Please feel free to open an issue if something is amiss in ABCMB!

## Citation

If you use ABCMB to publish scientific research, we suggest you cite
```
@misc{abcmb,
      title={{ABCMB: A Python+JAX Package for the Cosmic Microwave Background Power Spectrum}}, 
      author={Zilu Zhou and Cara Giovanetti and Hongwan Liu},
      year={2026},
      eprint={2602.15104},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2602.15104}, 
}
```



