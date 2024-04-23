#' @importFrom R6 R6Class
#' @import mlr3
LearnerRieszConstant <- R6::R6Class(
  "LearnerRieszConstant",
  inherit = mlr3::Learner,
  public = list(
    #' @importFrom paradox ps p_lgl
    initialize = function() {
      params <- ps(
        constant = p_dbl(default = 1, tags = "train")
      )

      super$initialize(
        id = "riesz.constant",
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
      constant <- 1
      if(!is.null(pv$constant)) constant <- pv$constant
      matrix(constant, ncol = 1, nrow = nrow(x))
    }
  ),
  private = list(
    .model = NULL,
    .train = function(task) {
      TRUE
    },
    .predict = function(task) {
      response <- self$alpha(task$data())
      list(response = response)
    }
  )
)
