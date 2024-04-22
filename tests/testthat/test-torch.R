set.seed(1)
N <- 1e2
data <- data.frame(W = runif(N, -1, 1))
data$A = rbinom(N, 1, plogis(data$W))

m <- \(alpha, data) alpha(data())

test_that("custom architectures work", {
  architecture <- function(input_dimension) {
    torch::nn_sequential(
      torch::nn_linear(input_dimension, 10),
      torch::nn_elu(),
      torch::nn_linear(10, 1),
      torch::nn_softplus()
    )
  }

  expect_no_error(super_riesz(data, list(list("torch", architecture = architecture, epochs = 20))))
})
