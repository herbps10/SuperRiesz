#' @import mlr3 mlr3misc data.table
.onLoad <- function(libname, pkgname){
  mlr_reflections$task_types <- data.table::setkeyv(rbind(mlr_reflections$task_types, rowwise_table(
    ~type, ~package, ~task, ~learner, ~prediction, ~prediction_data, ~measure,
    "riesz", "SuperRiesz", "TaskRiesz", "LearnerRiesz", "PredictionRegr", "PredictionDataRegr", "MeasureRegr"
  )), "type")

  mlr_reflections$task_col_roles$riesz = mlr3::mlr_reflections$task_col_roles$regr
  mlr_reflections$learner_predict_types$riesz <- list(response = "response")

  mlr_learners$add(key = "riesz.constant", LearnerRieszConstant)
  mlr_learners$add(key = "riesz.nn", LearnerRieszNN)
  mlr_learners$add(key = "riesz.linear", LearnerRieszLinear)
  mlr_learners$add(key = "riesz.torch", LearnerRieszTorch)
  mlr_learners$add(key = "riesz.glm", LearnerRieszGLM)
}
