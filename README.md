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

## Installation
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

## Examples
We have included several pedagogical jupyter notebooks to walk you through how to get started with ABCMB in our [example_notebooks](https://github.com/TonyZhou729/ABCMB/tree/main/example_notebooks) folder.  We suggest you start with [ABCMB_basics](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_basics.ipynb) to get a sense of how to run the code.  If you'd like to add new physics to ABCMB, check out [ABCMB_Fluids](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_Fluids.ipynb).  If you'd like to run ABCMB with the Big Bang Nucleosynthesis (BBN) code [LINX](https://github.com/cgiovanetti/LINX/tree/main) to do BBN+CMB joint analyses, check out [ABCMB_with_LINX](https://github.com/TonyZhou729/ABCMB/blob/main/example_notebooks/ABCMB_with_LINX.ipynb).

## Issues
Please feel free to open an issue if something is amiss in ABCMB!

## Third-party code
ABCMB's HALOFIT implementation (`abcmb/halofit.py`) is adapted from [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) (MIT License, Copyright (c) 2022 Differentiable Universe Initiative); see `LICENSE-jax_cosmo`.  If you use the non-linear matter power spectrum, we suggest you cite jax-cosmo ([Campagne et al. 2023](https://arxiv.org/abs/2302.05163)) and the HALOFIT papers, [Smith et al. 2003](https://arxiv.org/abs/astro-ph/0207664) and [Takahashi et al. 2012](https://arxiv.org/abs/1208.2701).

## Citation

If you use ABCMB to publish scientific research, we suggest you cite
```
@article{abcmb,
   title={{ABCMB: A Python+JAX Package for the Cosmic Microwave Background Power Spectrum}},
   volume={2026},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2026/08/078},
   DOI={10.1088/1475-7516/2026/08/078},
   number={08},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Zhou, Zilu and Giovanetti, Cara and Liu, Hongwan},
   year={2026},
   month=Aug, pages={078} }
```



