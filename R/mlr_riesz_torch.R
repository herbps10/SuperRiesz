torch_estimate_representer <-
  function(data,
           architecture,
           epochs = 500,
           learning_rate = 1e-3,
           verbose = FALSE,
           seed = 1,
           m = \(learner, data) learner(data()),
           ...) {
    d_in <- ncol(data())
    d_out <- 1

    torch::torch_manual_seed(seed)

    riesz <- architecture(d_in)

    Map(\(x) torch::nn_init_normal_(x, 0, 0.1), riesz$parameters)

    learner <- function(x) {
      riesz(x)[, 1]
    }

    optimizer <- torch::optim_adam(
      params = c(riesz$parameters),
      lr = learning_rate,
      weight_decay = 0
    )

    scheduler <- torch::lr_one_cycle(optimizer, max_lr = learning_rate, total_steps = epochs)
    for (epoch in 1:epochs) {
      # Regression loss
      loss <- (learner(data())$pow(2) - (2 * m(learner, data)))$mean(dtype = torch::torch_float())

      if (verbose == TRUE && (epoch %% 20 == 0)) {
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
    architecture = NULL,
    #' @importFrom paradox ps p_int p_dbl p_lgl
    initialize = function(architecture, ...) {
      params <- ps(
        epochs = p_int(1L, default = 20L, tags = "train"),
        learning_rate = p_dbl(default = 1e3, tags = "train"),
        seed = p_int(1L, default = 1L, tags = "train"),
        verbose = p_lgl(default = FALSE, tags = "train"),
        ...
      )

      self$architecture <- architecture

      super$initialize(
        id = "riesz.torch",
        task_type = "riesz",
        predict_types = c("response"),
        feature_types = c("logical", "integer", "numeric"),
        param_set = params
      )
    },
    loss = function(task) {
      data <- private$.torch_data(task)
      loss <- (self$alpha(data()$pow(2)) - 2 * task$m(self$alpha, data))$mean(dtype = torch::torch_float())
      torch::as_array(loss)
    },
    alpha = function(x) {
      self$model(torch::torch_tensor(as.matrix(x), dtype = torch::torch_float()))
    }
  ),
  private = list(
    .model = NULL,
    .torch_data = function(task) {
      # Convert natural data to torch tensor
      torch_data <- torch::torch_tensor(as.matrix(task$data()), dtype = torch::torch_float())

      # Convert all alternative versions of the data to torch tensors
      torch_alternatives = lapply(names(task$alternatives), \(x) torch::torch_tensor(as.matrix(task$data(x)), dtype = torch::torch_float()))
      names(torch_alternatives) <- names(task$alternatives)

      function(key = NA) {
        if(is.na(key)) return(torch_data)
        return(torch_alternatives[[key]])
      }
    },
    .train = function(task) {
      pv <- self$param_set$get_values(tags = "train")

      self$model <-
        mlr3misc::invoke(
          torch_estimate_representer,
          architecture = self$architecture,
          data = private$.torch_data(task),
          m = task$m,
          .args = pv
        )
    },
    .predict = function(task) {
      x <- self$alpha(task$data())
      list(response = torch::as_array(x)[, 1])
    }
  )
)

