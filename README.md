# SuperRiesz
R package for ensemble estimation of Riesz Representers for a family of causal functionals.

## Installation
To install `SuperRiesz` directly from Github:
```
remotes::install_github("herbps10/SuperRiesz")
```

## Mathematical Background
Suppose we observe data $O = (Y, X)$ where $Y$ is an outcome and $X$ a vector. Define the target parameter of interest as $\theta = \mathbb{E}[m(O, \bar{Q})]$ where $m$ is an arbitrary function and $\bar{Q}(X) = \mathbb{E}[Y \mid X]$. 

If $\theta$ is a continuous linear functional of $\bar{Q}$, then by the Riesz Representation Theorem there exists a random variable $\alpha(X)$ such that, for all $f$ with $\mathbb{E}[f(X)^2] < \infty$,

$$
\mathbb{E}[m(O, f)] = \mathbb{E}[\alpha(X) f(X)].
$$

The random variable $\alpha(X)$ is called the _Riesz Representer_ of $\theta$. This package estimates the Riesz Representer $\alpha$ by solving the minimization problem

$$
\hat{\alpha} = \arg\min_{\alpha \in \mathcal{A}} \frac{1}{n} \sum_{i=1}^n \alpha(X_i)^2-2 m(O_i, \alpha).
$$

For more details, [_RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests_](https://proceedings.mlr.press/v162/chernozhukov22a/chernozhukov22a.pdf) by Chernozhukov et al. (2022) has a nice overview of the theory. See also [_Automatic Debiased Machine Learning via Riesz Regression_](https://arxiv.org/abs/2104.14737) by Chernozhukov et al. (2021).

## Usage
The primary function is `super_riesz`. 

Arguments:
- `data`: data frame containing all observations.
- `alternatives`: named list containing any alternative versions of the dataset. 
- `library`: vector or list of candidate learners.
- `m`: function defining the parameter mapping.
- `discrete`: boolean indicating whether to estimate a discrete (`TRUE`) or continuous (`FALSE`) Super Learner.

## Example
Estimate the Riesz Representer for the Average Treatment Effect: $\theta = \mathbb{E}[\mathbb{E}[Y \mid A = 1, W]] - \mathbb{E}[Y \mid A = 0, W]]$. Define $\bar{Q}(A, W) = \mathbb{E}[Y \mid A, W]$. Then $m(O, \bar{Q}) = \bar{Q}(1, W) - \bar{Q}(0, W)$.  
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

vars <- c("W", "A")
m <- \(alpha, data) alpha(data("treatment")) - alpha(data("control"))
fit <- super_riesz(
  data[, vars], 
  list(
    "control" = mutate(data[, vars], A = 0),
    "treatment" = mutate(data[, vars], A = 1)
  ),
  library = c("glm", "torch"), 
  m = m
)

predict(fit, data[, vars])

```


## Learners

### `torch`
Simple neural network learner based on `torch`.

Parameters:
- `hidden`: number of neurons in each hidden layer (integer)
- `learning_rate`: training learning rate (positive float)
- `epochs`: number of training epochs (integer)
- `dropout`: dropout in hidden layer during training (float between 0 and 1)
- `constrain_positive`: whether to constrain predictions to be positive (true or false)
- `seed`: torch random seed (optional)

### `glm`
Linear learner based on `optim`. 

Parameters:
- `constrain_positive`: whether to constrain predictions to be positive (true or false)
