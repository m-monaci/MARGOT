# MARGOT

This repository contains the source code and the data related to the paper [Margin Optimal Classification Trees](https://arxiv.org/abs/2210.10567)
by Federico D'Onofrio, Giorgio Grani, Marta Monaci and Laura Palagi.

```
@misc{https://doi.org/10.48550/arxiv.2210.10567,
  doi = {10.48550/ARXIV.2210.10567},
  url = {https://arxiv.org/abs/2210.10567},
  author = {D'Onofrio, Federico and Grani, Giorgio and Monaci, Marta and Palagi, Laura},
  title = {Margin Optimal Classification Trees},
  publisher = {arXiv},
  year = {2022}
}
```

## Installation

The MIP model for generating MARGOT Trees is implemented in [Gurobi Optimizer](https://www.gurobi.com/solutions/gurobi-optimizer/).

## Requirements

The file requirements.txt reports the list of packages that must be installed to run the code. You can add a package to your environment via pip or anaconda using either pip install "package" or conda install "package".

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
