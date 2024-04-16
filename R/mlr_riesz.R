TaskRiesz <- R6::R6Class(
  "TaskRiesz",
  inherit = Task,
  public = list(
    m = NULL,
    initialize = function(id,
                          backend,
                          backend_shifted,
                          conditional_indicator,
                          m,
                          label = NA_character_,
                          extra_args = list()) {
      super$initialize(
        id = id,
        task_type = "riesz",
        backend = backend,
        label = label,
        extra_args = extra_args
      )
      private$.backend_shifted <- as_data_backend(backend_shifted)
      private$.conditional_indicator <- conditional_indicator
      self$m <- m
    },
    natural = function() {
      self$data()
    },
    shifted = function() {
      rows <- private$.row_roles$use
      cols <- private$.col_roles$feature
      data_format = "data.table"
      private$.backend_shifted$data(rows, cols, data_format)
    },
    conditional_indicator = function() {
      private$.conditional_indicator
    }
  ),
  private = list(
    .conditional_indicator = NULL,
    .backend_shifted = NULL
  )
)
