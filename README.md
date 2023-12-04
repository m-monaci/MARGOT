# MARGOT

This repository contains the source code and the data related to the paper [Margin Optimal Classification Trees](https://arxiv.org/abs/2210.10567)
by Federico D'Onofrio, Giorgio Grani, Marta Monaci and Laura Palagi.

```
@article{DONOFRIO2024106441,
title = {Margin optimal classification trees},
journal = {Computers & Operations Research},
volume = {161},
pages = {106441},
year = {2024},
issn = {0305-0548},
doi = {https://doi.org/10.1016/j.cor.2023.106441},
url = {https://www.sciencedirect.com/science/article/pii/S0305054823003052},
author = {Federico D’Onofrio and Giorgio Grani and Marta Monaci and Laura Palagi}
}
```

## Installation

The MIP model for generating MARGOT Trees is implemented in [Gurobi Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/).

## Requirements

The file requirements.txt reports the list of packages that must be installed to run the code. You can add a package to your environment via pip or anaconda using either _pip install "package"_ or _conda install "package"_. 

Then, in order to install _pygraphviz_, we reccomend to use _conda install -c conda-forge pygraphviz_.

## Configuration and running

You just need to run `main.py`. 
The parameters are set according to the experiments in the paper, but you can simply modify them via `main.py`. 

## Results

The output of the experiments can be found in folder **results_margot/**, where:

- folder **plots/** will contain all the tree plots files related to the experiments performed.
- file **stats_margot.xlsx** will contain the statistics of all the experiments performed.

## Team

Contributors to this code:

* [Marta Monaci](https://github.com/m-monaci)
* [Federico D'Onofrio](https://github.com/fededonofrio)

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [MIT License](https://opensource.org/licenses/mit-license.php)
* Copyright 2022 © Marta Monaci, Federico D'Onofrio
