torch_estimate_representer <-
  function(natural,
           shifted,
           conditional_indicator,
           hidden = 20,
           epochs = 500,
           learning_rate = 1e-3,
           m = \(natural, shifted) shifted,
           dropout = 0.1) {
    d_in <- ncol(natural)
    d_out <- 1

    natural <- torch::torch_tensor(as.matrix(natural), dtype = torch::torch_float())
    shifted <- torch::torch_tensor(as.matrix(shifted), dtype = torch::torch_float())

    riesz <- torch::nn_sequential(
      torch::nn_linear(d_in, hidden),
      torch::nn_elu(),
      torch::nn_linear(hidden, hidden),
      torch::nn_elu(),
      torch::nn_dropout(dropout),
      torch::nn_linear(hidden, d_out),
      torch::nn_softplus()
    )

    Map(\(x) torch::nn_init_normal_(x, 0, 0.1), riesz$parameters)

    learner <- function(x) {
      riesz(x)[, 1]
    }

    optimizer <- torch::optim_adam(
      params = c(riesz$parameters),
      lr = learning_rate,
      weight_decay = 0
    )

    conditional_indicator <- torch::torch_tensor(conditional_indicator)
    conditional_mean <- conditional_indicator$mean(dtype = torch::torch_float())

    scheduler <- torch::lr_one_cycle(optimizer, max_lr = learning_rate, total_steps = epochs)
    for (epoch in 1:epochs) {
      rr <- learner(natural)
      rr_shifted <- learner(shifted)

      # Regression loss
      loss <- (rr$pow(2))$mean(dtype = torch::torch_float()) - (2 * m(rr, rr_shifted, conditional_indicator, conditional_mean))$mean(dtype = torch::torch_float())

      if (epoch %% 20 == 0) {
        cat("Epoch: ", epoch, " Loss: ", loss$item(), "\n")
      }

      optimizer$zero_grad()
      loss$backward()

      optimizer$step()
      scheduler$step()
    }

    riesz$eval()
    riesz
  }

#' @import mlr3
LearnerRieszTorch <- R6::R6Class(
  "LearnerRieszTorch",
  inherit = mlr3::Learner,
  public = list(
    #' @importFrom paradox ps p_int p_dbl
    initialize = function() {
      params <- ps(
        hidden = p_int(1L, default = 20L, tags = "train"),
        epochs = p_int(1L, default = 20L, tags = "train"),
        dropout = p_dbl(0, 1, default = 0.1, tags = "train"),
        learning_rate = p_dbl(default = 1e3, tags = "train")
      )

      super$initialize(
        id = "riesz.torch",
        task_type = "riesz",
        predict_types = c("response"),
        feature_types = c("logical", "integer", "numeric"),
        param_set = params
      )
    },
    loss = function(task) {
      natural <- self$model(torch::torch_tensor(as.matrix(task$natural()), dtype = torch::torch_float()))
      shifted <- self$model(torch::torch_tensor(as.matrix(task$shifted()), dtype = torch::torch_float()))

      conditional_indicator <- torch::torch_tensor(task$conditional_indicator())
      conditional_mean      <- conditional_indicator$mean(dtype = torch::torch_float())

      loss <- (natural$pow(2))$mean(dtype = torch::torch_float()) - 2 * (task$m(natural, shifted, conditional_indicator, conditional_mean))$mean(dtype = torch::torch_float())
      torch::as_array(loss)
    }
  ),
  private = list(
    .model = NULL,
    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$model <-
        mlr3misc::invoke(
          torch_estimate_representer,
          natural = task$natural(),
          shifted = task$shifted(),
          m = task$m,
          conditional_indicator = task$conditional_indicator(),
          .args = pv
        )
    },
    .predict = function(task) {
      #shifted <- self$model(torch::torch_tensor(as.matrix(task$shifted()), dtype = torch::torch_float()))
      #conditional_mean <- torch::torch_tensor(task$conditional_indicator())$mean(dtype = torch::torch_float())
      #list(response = torch::as_array(natural * conditional_mean)[, 1])

      natural <- self$model(torch::torch_tensor(as.matrix(task$natural()), dtype = torch::torch_float()))
      list(response = torch::as_array(natural)[, 1])
    }
  )
)

