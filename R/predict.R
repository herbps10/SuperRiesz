#' @export
predict.super_riesz <- function(object, newdata, ...) {
  task <- TaskRiesz$new(id = "superriesz", newdata, newdata, matrix(1, ncol = 1, nrow = nrow(newdata)), object$m)

  # Generate predictions
  preds <- matrix(
    unlist(lapply(object$learners, \(learner) learner$predict(task)$response)),
    ncol = length(object$learners),
    byrow = FALSE
  )

  (preds %*% object$weights)[, 1]
}

