# Renormalized Operators with Multiscale Attention (ROMA)

<img src="https://github.com/nngabe/roma/blob/master/figures/ROMA_simple.png" width="512">

This repository contains code and data accompanying the manuscript titled [Connecting the Geometry and Dynamics of Many-Body Complex Systems with Neural Operators](https://arxiv.org/abs/xxxx.yyyyy)

## Abstract

The relationship between scale transformations and dynamics established by renormalization group techniques is a cornerstone of modern physical theories, from fluid mechanics to elementary particle physics. Integrating renormalization group methods into neural operators for many-body complex systems could provide a foundational inductive bias for learning their effective dynamics, while also uncovering multiscale organization. We introduce a scalable AI framework, ROMA (Renormalized Operators with Multiscale Attention), for learning multiscale evolution operators of many-body complex systems. In particular, we develop a renormalization procedure based on neural analogs of the geometric and laplacian renormalization groups, which can be co-learned with neural operators. An attention mechanism is used to model multiscale interactions by connecting geometric representations of local subgraphs and dynamical operators. We apply this framework in challenging conditions: large systems of more than 1M nodes, long-range interactions, and noisy input-output data for two contrasting examples: Kuramoto oscillators and Burgers-like social dynamics. We demonstrate that the ROMA framework improves scalability and positive transfer between forecasting and effective dynamics tasks compared to state-of-the-art operator learning techniques, while also giving insight into multiscale interactions. Additionally, we investigate power law scaling in the number of model parameters, and demonstrate a departure from typical power law exponents in the presence of hierarchical and multiscale interactions.

## Installation

Dependencies can be installed with pip using the following commands:

```
pip3 install -U pip
pip3 install --upgrade jax jaxlib
pip3 install --upgrade -r requirements.txt
```

Then install the `roma` package by running the following command:

```
git clone https://github.com/nngabe/roma.git
cd roma
pip install -e .
```

Lastly, to set environmental variables run 

```
bash set_env.sh
```

## Datasets

Instructions for downloading all datasets can be found here (upon preprint publication).

## Experiments

### Data Scaling & Noise

### Effective Dynamics

### Positional Embedding

### ROMA Scaling


## Citation
    @article{gabriel2024connecting,
      title={Connecting the Geometry and Dynamics of Many-Body Complex Systems with Neural Operators},
      author={Gabriel, Nicholas A and Johnson, Neil F and Karniadakis, George Em},
      journal={arXiv preprint arXiv:xxxx.yyyyy},
      year={2024}
    }
