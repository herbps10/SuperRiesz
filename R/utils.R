 add_interactions <- function(df, interactions = 1) {
  if(is.null(interactions)) interactions <- 1
  interactions <- min(interactions, ncol(df) - 1)
  if(interactions > 1) {
    for(j in 2:interactions) {
      vars <- t(combn(names(df), j))

      for(i in 1:nrow(vars)) {
        var <- vars[i, ]
        if(is.data.table(df)) {
          df[, paste0(var, collapse = "_")] <- prod(df[, ..var])
        }
        else {
          df[, paste0(var, collapse = "_")] <- prod(df[, var])
        }
      }
    }
  }
  df
}
