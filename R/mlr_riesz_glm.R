#' @importFrom stats optim
glm_estimate_representer <- function(data,
  m,
  constrain_positive = TRUE) {
  if(constrain_positive == TRUE) {
    transform <- exp
  }
  else {
    transform <- \(x) x
  }

  pred <- function(beta, x) {
    transform(as.matrix(cbind(1, x)) %*% beta)
  }

  loss <- function(beta) {
    alpha <- function(x) pred(beta, x)
    y <- m(alpha, data)
    if(is.data.frame(y) || is.data.table(y)) y <- y[[1]]
    mean(alpha(data())^2 - 2 * y)
  }

  pars <- numeric(ncol(data()) + 1)
  x <- optim(pars, fn = loss)
  beta <- x$par
  beta
}

#' @importFrom R6 R6Class
#' @import mlr3
LearnerRieszGLM <- R6::R6Class(
  "LearnerRieszGLM",
  inherit = mlr3::Learner,
  public = list(
    #' @importFrom paradox ps p_lgl
    initialize = function() {
      params <- ps(
        constrain_positive = p_lgl(default = TRUE, tags = "train")
      )

      super$initialize(
        id = "riesz.glm",
        task_type = "riesz",
        predict_types = c("response"),
        feature_types = c("logical", "integer", "numeric"),
        param_set = params
      )
    },
    loss = function(task) {
      mean(
        self$alpha(task$data())^2 - 2 * task$m(self$alpha, task$data)[[1]]
      )
    },
    alpha = function(x) {
      pv <- self$param_set$get_values(tags = "train")
      constrain_positive <- TRUE
      if(!is.null(pv$constrain_positive)) constrain_positive <- pv$constrain_positive
      transform <- \(x) x
      if(constrain_positive == TRUE) transform <- exp
      transform(as.matrix(cbind(1, x)) %*% self$model)
    }
  ),
  private = list(
    .model = NULL,

    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$model <-
        mlr3misc::invoke(
          glm_estimate_representer,
          data = task$data,
          m = task$m,
          .args = pv
        )
    },
    .predict = function(task) {
      response <- self$alpha(task$data())
      list(response = response)
    }
  )
)
