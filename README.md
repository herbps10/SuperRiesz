# SuperRiesz
R package for ensemble estimation of Riesz Representers for a family of causal functionals.

## Installation
To install `SuperRiesz` directly from Github:
```
remotes::install_github("herbps10/SuperRiesz")
```

## Mathematical Background
Suppose we observe data $O = (W, A, Y)$ where $W$ is a vector of covariates, $A$ a vector of binary or continuous treatment variables, and $Y$ a binary or continuous outcome. Let $A^d$ be a _modified_ or _shifted_ version of $A$. Let $\mathcal{B}$ be a conditioning set. Define the target parameter of interest as $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = A^d, W] | A \in \mathcal{B}]$. 

Note that $\theta$ is a linear functional of $\mathbb{E}[Y \mid A = A^d, W]$. Therefore, by the Riesz Representation Theorem, there exists a function $\alpha$ such that, for all $f$ with $\mathbb{E}[f(A, W)^2] < \infty$,

$$
\mathbb{E}[f(A^d, W)] = \mathbb{E}[\alpha(A, W) f(A, W)].
$$

This function $\alpha$ is called the _Riesz Representer_ of $\theta$. 

This package estimates the Riesz Representer $\alpha$ by solving the minimization problem

$$
\hat{\alpha} = \arg\min_{\alpha \in \mathcal{A}} \frac{1}{n} \sum_{i=1}^n \alpha(A_i, W_i)^2-2\alpha(A_i^d, W_i). 
$$

For more details, [_RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests_](https://arxiv.org/abs/2110.03031) by Chernozhukov et al. (2022) has a nice overview of the theory. 

## Usage
The key function is `super_riesz`. The argument `m` allows for customization of the causal parameter of interest for which the Riesz Representer is estimated. 

Examples:
- Mean counterfactual outcome, no conditioning: $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = A^d, W]]$. Set `m <- \\(natural, shifted, conditional, conditional_mean) shifted`.
- Mean counterfactual outcome, with conditioning: $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = A^d, W] | A \in \mathcal{B}]$. Set `m <- \\(natural, shifted, conditional, conditional_mean) shifted * conditional`.

## Example
Estimate the Riesz Representer for $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = 1, W]]$ and $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = 1, W] | A = 0]$ . 
```
# Simulate data
library(tidyverse)
N <- 1e3

set.seed(152)
data <- tibble(
  W = runif(N, -1, 1),
  A = rbinom(N, 1, plogis(W)),
  Y = rnorm(N, mean = W + A, sd = 0.5)
)

data_shifted <- mutate(data, A = 1)

# Without conditioning
m <- \(natural, shifted, conditional_indicator, conditional_mean) shifted
fit <- super_riesz(
  data[, c("A", "W")], 
  data_shifted[, c("A", "W")], 
  library = c("glm", "torch"), 
  m = m
)
predict(fit, data_shifted)

# With conditioning
m <- \(natural, shifted, conditional_indicator, conditional_mean) shifted * conditional_indicator
conditional_indicator <- matrix(data$A == 0, ncol = 1)
fit <- super_riesz(
  data[, c("W", "A")], 
  data_shifted[, c("W", "A")], 
  conditional_indicator = conditional_indicator,
  library = c("glm", "torch"), 
  m = m
)
predict(fit, data_shifted)

```


## Learners

### `torch`
Neural network learner based on `torch`.

Parameters:
- `hidden`: number of neurons in each hidden layer (integer)
- `learning_rate`: training learning rate (positive float)
- `epochs`: number of training epochs (integer)
- `dropout`: dropout in hidden layer during training (float between 0 and 1)
- `seed`: torch random seed (optional)

### `glm`
Linear learner based on `optim`. 
