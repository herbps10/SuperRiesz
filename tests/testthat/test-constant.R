set.seed(1)
N <- 1e2
data <- data.frame(W = runif(N, -1, 1))
data$A = rbinom(N, 1, plogis(data$W))

m <- \(alpha, data) alpha(data())

test_that("constant learner works", {
  expect_no_error(fit <- super_riesz(data, c("constant")))
  expect_equal(predict(fit, data), rep(1, nrow(data)))

  expect_no_error(fit <- super_riesz(data, list(list("constant", constant = 2))))
  expect_equal(predict(fit, data), rep(2, nrow(data)))
})

