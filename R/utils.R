 add_interactions <- function(df, interactions = 1) {
  if(is.null(interactions)) interactions <- 1
  interactions <- min(interactions, ncol(df) - 1)
  if(interactions > 1) {
    for(j in 2:interactions) {
      vars <- t(combn(names(df), j))

      for(i in 1:nrow(vars)) {
        var <- vars[i, ]
        if(is.data.table(df)) {
          y <- apply(df[, ..var], 1, prod)
          df[, paste0(var, collapse = "_")] <- y
        }
        else {
          y <- apply(df[, var], 1, prod)
          df[, paste0(var, collapse = "_")] <- y
        }
      }
    }
  }
  df
}
