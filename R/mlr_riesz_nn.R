#' @import torch
nn_architecture <-
  function(data,
           layers = 1,
           hidden = 20,
           epochs = 500,
           learning_rate = 1e-3,
           seed = 1,
           constrain_positive = TRUE,
           verbose = FALSE,
           m = \(learner, data) learner(data()),
           dropout = 0.1) {
  d_in <- ncol(data())
  d_out <- 1

  middle_layers <- lapply(1:layers, \(x) torch::nn_sequential(torch::nn_linear(hidden, hidden), torch::nn_elu()))

  if(constrain_positive == TRUE) {
    architecture <- \(d_in) torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_elu(),
      do.call(torch::nn_sequential, middle_layers),
      torch::nn_linear(hidden, d_out),
      torch::nn_dropout(dropout),
      torch::nn_softplus()
    )
  }
  else {
    architecture <- \(d_in) torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_elu(),
      do.call(torch::nn_sequential, middle_layers),
      torch::nn_linear(hidden, d_out),
      torch::nn_dropout(dropout)
    )
  }

  architecture
}

#' @import mlr3
LearnerRieszNN <- R6::R6Class(
  "LearnerRieszNN",
  inherit = LearnerRieszTorch,
  public = list(
    #' @importFrom paradox ps p_int p_dbl p_lgl
    initialize = function() {
      super$initialize(
        architecture = NULL,
        hidden = p_int(1L, default = 20L, tags = "train"),
        layers = p_int(1L, default = 1L, tags = "train"),
        dropout = p_dbl(0, 1, default = 0.1, tags = "train"),
        constrain_positive = p_lgl(default = TRUE, tags = "train")
      )
    }
  ),
  private = list(
    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$architecture <-
        mlr3misc::invoke(
          nn_architecture,
          data = private$.torch_data(task),
          m = task$m,
          .args = pv
        )

      super$.train(task)
    }
  )
)

