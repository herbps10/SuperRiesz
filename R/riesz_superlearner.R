riesz_superlearner_weights <- function(learners, task) {
  risks <- lapply(learners, function(x) {
    x$loss(task)
  })

  unlist(risks)
}

#' @import mlr3
#' @export
super_riesz <- function(data, data_shifted, library, conditional_indicator = matrix(ncol = 1, rep(1, nrow(data))), m = \(data, data_shifted, conditional_indicator, conditional_mean) data_shifted, discrete = TRUE) {
  task <- TaskRiesz$new(id = "superriesz", data, data_shifted, conditional_indicator, m)

  if(is.list(library)) {
    learners <- lapply(library, \(learner) {
      args <- learner[-1]
      args$.key <- paste0("riesz.", learner[[1]])
      do.call(lrn, args)
    })
  }
  else {
    learners <- lapply(library, \(learner) {
      lrn(paste0("riesz.", learner))
    })
  }

  # Train learners
  lapply(learners, \(learner) learner$train(task))
  risks <- riesz_superlearner_weights(learners, task)

  if(length(library) == 1 || discrete) {
    weights <- numeric(length(library))
    weights[which.min(risks)] <- 1
  }

  sl <- list(learners = learners, weights = weights, m = m, risk = risks)
  class(sl) <- "super_riesz"
  sl
}
