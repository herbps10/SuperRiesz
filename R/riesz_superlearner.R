riesz_superlearner_weights <- function(learners, task, folds) {
  resampling <- mlr3::rsmp("cv", folds = folds)
  resampling$instantiate(task)

  cv_risks <- matrix(nrow = folds, ncol = length(learners))
  preds <- matrix(nrow = task$nrow, ncol = length(learners))
  for(fold in 1:folds) {
    cv_learners <- lapply(learners, \(x) x$clone()$train(task$clone()$filter(resampling$train_set(fold))))
    cv_risks[fold, ] <- unlist(lapply(cv_learners, \(x) x$loss(task$clone()$filter(resampling$test_set(fold)))))
    preds[resampling$test_set(fold), ] <- unlist(lapply(cv_learners, \(x) x$predict(task$clone()$filter(resampling$test_set(fold)))$response))
  }

  risks <- colMeans(cv_risks)
  names(risks) <- unlist(lapply(learners, \(x) x$id))

  list(
    risks = risks,
    preds = preds
  )
}

#' Ensemble estimation of Riesz Representers
#'
#' @param data data frame containing observations as originally observed
#' @param data_shifted data frame containing observations after treatment intervention has been performed
#' @param library vector or list specifying learners to be included in the ensemble
#' @param conditional_indicator matrix indicating which observations are included in conditioning set for causal parameter of interest
#' @param m functional used to define causal parameter of interest for which Riesz Representer is to be estimated
#' @param folds number of cross-fitting folds
#'
#' @import mlr3
#' @export
super_riesz <- function(data, library, alternatives = list(), m = \(alpha, data) alpha(data()), folds = 5) {
  checkmate::assert_vector(library, min.len = 1)
  checkmate::assert_function(m)
  checkmate::assert_int(folds, lower = 1)
  checkmate::assert_list(alternatives)

  discrete <- TRUE
  checkmate::assert_logical(discrete)

  task <- TaskRiesz$new(id = "superriesz", backend = data, alternatives = alternatives, m = m)

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

  if(is.null(folds)) folds = 5

  if(folds > 1) {
    cv <- riesz_superlearner_weights(learners, task, folds)
    cv_risks <- cv$risks
    cv_preds <- cv$preds
  }

  # Train learners
  lapply(learners, \(learner) learner$train(task))

  if(folds == 1) {
    cv_risks <- unlist(lapply(learners, \(learner) learner$loss(task)))
    names(cv_risks) <- unlist(lapply(learners, \(learner) learner$id))
    cv_preds <- matrix(ncol = length(learners), nrow = task$nrow, unlist(lapply(learners, \(learner) learner$predict(task)$response)))
  }

  if(length(library) == 1 || discrete) {
    weights <- numeric(length(library))
    weights[which.min(cv_risks)] <- 1
  }
  else {
    #par <- rep(0, length(library))
    #optim(par, \(weights) {
    #
    #})
  }

  sl <- list(learners = learners, weights = weights, m = m, risk = cv_risks)
  class(sl) <- "super_riesz"
  sl
}
