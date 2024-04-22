glm_estimate_representer <- function(natural,
  shifted,
  conditional_indicator,
  m,
  hidden = 20,
  epochs = 500,
  learning_rate = 1e-3,
  dropout = 0.1) {
  pred <- function(beta, x) {
    exp(as.matrix(cbind(1, x)) %*% beta)
  }

  loss <- function(beta) {
    mean(pred(beta, natural)^2 - 2 * m(pred(beta, natural), pred(beta, shifted), conditional_indicator, mean(conditional_indicator)))
  }

  pars <- numeric(ncol(natural) + 1)
  x <- optim(pars, fn = loss, lower = -20, upper = 20, method = "L-BFGS-B")
  beta <- x$par
  beta
}

#' @importFrom R6 R6Class
#' @import mlr3
LearnerRieszGLM <- R6::R6Class(
  "LearnerRieszGLM",
  inherit = mlr3::Learner,
  public = list(
    initialize = function() {
      super$initialize(
        id = "riesz.torch",
        task_type = "riesz",
        predict_types = c("response"),
        feature_types = c("logical", "integer", "numeric")
      )
    },
    loss = function(task) {
      natural <- task$natural()
      shifted <- task$shifted()
      m_natural <- as.matrix(cbind(1, natural))
      m_shifted <- as.matrix(cbind(1, shifted))
      conditional_indicator <- task$conditional_indicator()
      l <- mean(
        exp(m_natural %*% self$model)^2 - 2 * task$m(exp(m_natural %*% self$model), exp(m_shifted %*% self$model), conditional_indicator, mean(conditional_indicator))
      )
      l
    }
  ),
  private = list(
    .model = NULL,
    .train = function(task, epochs) {
      pv <- self$param_set$get_values(tags = "train")

      self$model <-
        mlr3misc::invoke(
          glm_estimate_representer,
          natural = task$natural(),
          shifted = task$shifted(),
          m = task$m,
          conditional_indicator = task$conditional_indicator(),
          .args = pv
        )
    },
    .predict = function(task) {
      data <- task$natural()
      response <- exp(as.matrix(cbind(1, data)) %*% self$model)
      list(response = response)
    }
  )
)
