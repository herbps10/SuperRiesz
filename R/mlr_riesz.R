TaskRiesz <- R6::R6Class(
  "TaskRiesz",
  inherit = Task,
  public = list(
    m = NULL,
    alternatives = NULL,
    initialize = function(id,
                          backend,
                          alternatives = list(),
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
      self$alternatives <- lapply(alternatives, as_data_backend)
      self$m <- m
    },
    data = function(key = NA, ...) {
      if(is.na(key)) {
        super$data(...)
      }
      else {
        rows <- private$.row_roles$use
        cols <- private$.col_roles$feature
        if(!all(cols %in% self$alternatives[[key]]$colnames)) {
          cols <- setdiff(self$alternatives[[key]]$colnames, "..row_id")
        }
        data_format = "data.table"
        self$alternatives[[key]]$data(rows, cols, data_format)
      }
    }
  )
)
