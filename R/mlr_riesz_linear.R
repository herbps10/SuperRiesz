#' @import torch
linear_nn_architecture <-
  function(data,
           epochs = 500,
           learning_rate = 1e-3,
           seed = 1,
           lambda = 0,
           constrain_positive = TRUE,
           verbose = FALSE,
           m = \(learner, data) learner(data())) {
    d_in <- ncol(data())
    print(d_in)
    d_out <- 1

    if(constrain_positive == TRUE) {
      architecture <- \(d_in) torch::nn_sequential(
        torch::nn_linear(d_in, d_out, bias = FALSE),
        torch::nn_softplus()
      )
    }
    else {
      architecture <- \(d_in) torch::nn_sequential(
        torch::nn_linear(d_in, d_out, bias = FALSE),
      )
    }

    architecture
  }

#' @import mlr3
LearnerRieszLinear <- R6::R6Class(
  "LearnerRieszLinear",
  inherit = LearnerRieszTorch,
  public = list(
    #' @importFrom paradox ps p_int p_dbl p_lgl
    initialize = function() {
      super$initialize(
        architecture = NULL,
        constrain_positive = p_lgl(default = TRUE, tags = "train")
      )
    }
  ),
  private = list(
    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$architecture <-
        mlr3misc::invoke(
          linear_nn_architecture,
          data = private$.torch_data(task),
          m = task$m,
          .args = pv
        )

      super$.train(task)
    }
  )
)

