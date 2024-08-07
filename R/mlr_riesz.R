TaskRiesz <- R6::R6Class(
  "TaskRiesz",
  inherit = Task,
  public = list(
    m = NULL,
    alternatives = NULL,
    extra = NULL,
    initialize = function(id,
                          backend,
                          alternatives = list(),
                          extra = list(),
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
      self$extra <- lapply(extra, as_data_backend)
      self$m <- m
    },
    data = function(key = NA, ...) {
      if(is.na(key)) {
        super$data(...)
      }
      else {
        rows <- private$.row_roles$use
        cols <- private$.col_roles$feature

        df <- self$alternatives
        if(!(key %in% names(df))) df <- self$extra

        if(!all(cols %in% df[[key]]$colnames)) {
          cols <- setdiff(df[[key]]$colnames, "..row_id")
        }
        data_format = "data.table"
        df[[key]]$data(rows, cols, data_format)
      }
    }
  )
)
