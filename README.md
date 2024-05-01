This repository contains a simple experiment demonstrating the value of residual connections in deep networks.

Residual\ Network\ Experiments.ipynb instantiates and trains four separate models, a shallow model with residual connections, a deep model with residual connections, a shallow model without residual connections, and a deep model without residual connections. Then it plots the % error they manage to achieve after training for 10 epochs on FashionMNIST.

From the resulting plots, it can be seen that deep models without residual connections suffer from optimization degradation, but the inclusion of residual connections resolves this degradation.
