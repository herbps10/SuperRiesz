#' @importFrom stats optim
glm_estimate_representer <- function(data,
  m,
  lambda = 0,
  constrain_positive = TRUE) {
  if(constrain_positive == TRUE) {
    transform <- exp
  }
  else {
    transform <- \(x) x
  }

  pred <- function(beta, x) {
    transform(as.matrix(x) %*% beta)
  }

  loss <- function(beta) {
    alpha <- function(x) pred(beta, x)
    y <- m(alpha, data)
    if(is.data.frame(y) || is.data.table(y)) y <- y[[1]]
    mean((alpha(data())^2 - 2 * y)[, 1]) + lambda * exp(sum(beta^2))
  }

  pars <- numeric(ncol(data()))
  x <- optim(pars, fn = loss, method = "BFGS")

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
        constrain_positive = p_lgl(default = TRUE, tags = "train"),
        lambda = p_dbl(default = 0, tags = "train"),
        interactions = p_int(1L, default = 1L, tags = "data")
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
      data <- private$.glm_data(task)
      mean(
        self$alpha(data())^2 - 2 * task$m(self$alpha, data)[[1]]
      )
    },
    alpha = function(x) {
      pv <- self$param_set$get_values(tags = c("train", "data"))
      constrain_positive <- TRUE
      if(!is.null(pv$constrain_positive)) constrain_positive <- pv$constrain_positive
      transform <- \(x) x
      if(constrain_positive == TRUE) transform <- exp
      transform(as.matrix(x) %*% self$model)
    }
  ),
  private = list(
    .model = NULL,

    .glm_data = function(task) {
      # Convert natural data to torch tensor
      pv <- self$param_set$get_values(tags = "data")

      data <- add_interactions(task$data(), pv$interactions)

      # Convert all alternative versions of the data
      alternatives = lapply(names(task$alternatives), \(x) add_interactions(task$data(x), pv$interactions))
      names(alternatives) <- names(task$alternatives)

      function(key = NA) {
        if(is.na(key)) return(data)
        if(key %in% names(task$extra)) return(task$data(key))
        return(alternatives[[key]])
      }
    },

    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$model <-
        mlr3misc::invoke(
          glm_estimate_representer,
          data = private$.glm_data(task),
          m = task$m,
          .args = pv
        )
    },
    .predict = function(task) {
      response <- self$alpha(private$.glm_data(task)())
      list(response = response)
    }
  )
)
