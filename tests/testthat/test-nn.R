set.seed(1)
N <- 1e2
data <- data.frame(W = runif(N, -1, 1))
data$A = rbinom(N, 1, plogis(data$W))

m <- \(alpha, data) alpha(data())

test_that("nn learner works", {
  expect_no_error(super_riesz(data, library = list(list("nn", epochs = 20, hidden = 1, verbose = FALSE))))
})
