set.seed(1)
N <- 1e2
data <- data.frame(W = runif(N, -1, 1))
data$A = rbinom(N, 1, plogis(data$W))
data$Y = rnorm(N, mean = data$W + data$A, sd = 0.5)
data0 <- data1 <- data
data0$A <- 0
data1$A <- 1

m <- \(alpha, data) alpha(data("treatment")) - alpha(data("control"))

test_that("argument checking works", {
  expect_error(super_riesz())
  expect_error(super_riesz(data, list(), m = "test"))
  expect_error(super_riesz(data, list(), library = c()))
  expect_error(super_riesz(data, list(), library = "torch", m = m, folds = "A"))
  expect_error(super_riesz(data, list(), library = "torch", m = m, folds = -1))
  expect_error(super_riesz(data, c("A"), library = "torch", m = m, folds = 5))
})

test_that("risks have learner names", {
  set.seed(5)
  expect_no_error(fit <- super_riesz(data, library = list(list("glm", constrain_positive = FALSE), list("constant", constant = 0.1)), list(control = data0, treatment = data1), m = m, folds = 5))
  expect_equal(names(fit$risk), c("riesz.glm", "riesz.constant"))
})

test_that("spot check works", {
  set.seed(5)
  fit <- super_riesz(data[, c("W", "A")], library = list(list("glm", constrain_positive = FALSE)), list(control = data0, treatment = data1), m = m, folds = 5)
  pred <- predict(fit, data[, c("W", "A")])

  expect_equal(fit$weights, 1)
  expect_equal(unname(fit$risk), -4.0, tolerance = 1e-1)
  expect_equal(pred[1:5], c(2.7, -1.7, 2.1, -2.6, -1.4), tolerance = 1e-1)
})

