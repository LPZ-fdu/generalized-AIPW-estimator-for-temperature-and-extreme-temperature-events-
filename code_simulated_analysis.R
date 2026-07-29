# ============================================================================
# File: 02_analyze_temperature_case_crossover_simulation_selected_metrics.R
# Purpose:
#   Analyze the non-optimal-temperature case-crossover simulation data and
#   compare four estimators using only the prespecified performance metrics
#   retained for the final manuscript table.
#
# Estimators:
#   1. Conventional conditional logistic regression.
#   2. Outcome-model estimating-equation estimator (f = 0).
#   3. Exposure-regression estimating-equation estimator (g = 0).
#   4. Proposed cross-fitted generalized AIPW estimator.
#
# Retained performance metrics:
#   - Scenario-specific mean absolute coefficient bias, averaged over the
#     heat- and cold-side log rate-ratio coefficients.
#   - Overall mean absolute coefficient bias across the three scenarios in
#     which at least one nuisance model is correctly specified.
#   - Worst-case absolute coefficient bias across those three scenarios.
#   - Overall mean absolute bias of the mean annual total attributable fraction.
#   - Mean empirical 95% confidence interval coverage.
#   - Numerical convergence rate.
#
# Final output:
#   A single transposed, formatted Excel table suitable for direct use as a
#   manuscript or supplementary-material table.
#
# Important:
#   - The original matched sets are the outer cross-fitting and inference units.
#   - The outcome nuisance model is fitted on the original sampled-day risk sets.
#   - The exposure nuisance model is fitted only among control-first ordered
#     pseudo-pairs (D = 0).
#   - Fixed prespecified XGBoost settings are used; no tuning or bootstrap is run.
# ============================================================================

options(stringsAsFactors = FALSE)

required_packages <- c("data.table", "survival", "xgboost", "openxlsx")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    "Missing required package(s): ",
    paste(missing_packages, collapse = ", "),
    ". Install them before running this script.",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(survival)
  library(xgboost)
  library(openxlsx)
})

# ----------------------------------------------------------------------------
# 1. Analysis settings and input files
# ----------------------------------------------------------------------------

ANALYSIS_SEED <- 20260713L

PROJECT_ROOT <- "/xxx/ccc"
INPUT_DIR <- file.path(PROJECT_ROOT, "simulated_data")
RESULT_DIR <- file.path(PROJECT_ROOT, "results")
CHECKPOINT_DIR <- file.path(RESULT_DIR, "selected_metric_checkpoints")

XGB_THREADS <- as.integer(
  Sys.getenv("SIM_XGB_THREADS", unset = "1")
)
if (!is.finite(XGB_THREADS) || XGB_THREADS < 1L) {
  XGB_THREADS <- 1L
}

RESUME_COMPLETED <- tolower(
  Sys.getenv("SIM_RESUME_COMPLETED", unset = "true")
) %in% c("true", "1", "yes", "y")

analysis_cfg <- list(
  analysis_seed = ANALYSIS_SEED,
  xgb_threads = XGB_THREADS,
  xgb_rounds = 180L,
  score_tolerance = 1e-8,
  parameter_tolerance = 1e-8,
  maximum_newton_iterations = 100L,
  maximum_absolute_eta = 700
)

dir.create(RESULT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

manifest_path <- file.path(INPUT_DIR, "simulation_manifest.csv")
configuration_path <- file.path(INPUT_DIR, "simulation_configuration.rds")
scenario_design_path <- file.path(INPUT_DIR, "scenario_design.csv")

required_input_files <- c(
  manifest_path,
  configuration_path,
  scenario_design_path
)

missing_input_files <- required_input_files[!file.exists(required_input_files)]
if (length(missing_input_files) > 0L) {
  stop(
    "Simulation input file(s) not found:\n",
    paste(missing_input_files, collapse = "\n"),
    "\nRun the data-generation script first.",
    call. = FALSE
  )
}

manifest <- fread(manifest_path)
simulation_configuration <- readRDS(configuration_path)
scenario_design <- fread(scenario_design_path)

scenarios_to_report <- c(
  "S1_both_correct",
  "S2_outcome_correct_exposure_misspecified",
  "S3_outcome_misspecified_exposure_correct",
  "S4_both_misspecified"
)

double_robustness_scenarios <- c(
  "S1_both_correct",
  "S2_outcome_correct_exposure_misspecified",
  "S3_outcome_misspecified_exposure_correct"
)

estimators_to_report <- c(
  "Conventional conditional logistic regression",
  "Outcome-model EE",
  "Exposure-regression EE",
  "Proposed generalized AIPW"
)

scenario_display_labels <- c(
  S1_both_correct =
    "Both nuisance models correct",
  S2_outcome_correct_exposure_misspecified =
    "Outcome correct; exposure misspecified",
  S3_outcome_misspecified_exposure_correct =
    "Outcome misspecified; exposure correct",
  S4_both_misspecified =
    "Both nuisance models misspecified"
)

manifest <- manifest[scenario_id %in% scenarios_to_report]
scenario_design <- scenario_design[scenario_id %in% scenarios_to_report]

manifest[, scenario_rank := match(scenario_id, scenarios_to_report)]
setorder(manifest, scenario_rank, replicate)
manifest[, scenario_rank := NULL]

if (nrow(manifest) != 4000L) {
  warning(
    "Expected 4000 simulated datasets, but the filtered manifest contains ",
    nrow(manifest),
    " datasets.",
    call. = FALSE
  )
}

if (anyDuplicated(manifest[, .(scenario_id, replicate)]) > 0L) {
  stop(
    "The manifest contains duplicated scenario_id-replicate combinations.",
    call. = FALSE
  )
}

set.seed(ANALYSIS_SEED)

# ----------------------------------------------------------------------------
# 2. General numerical helper functions
# ----------------------------------------------------------------------------

stable_matrix_inverse <- function(matrix_object, tolerance = 1e-10) {
  matrix_object <- as.matrix(matrix_object)
  
  if (any(!is.finite(matrix_object))) {
    return(matrix(
      NA_real_,
      nrow = nrow(matrix_object),
      ncol = ncol(matrix_object)
    ))
  }
  
  svd_object <- svd(matrix_object)
  
  if (length(svd_object$d) == 0L) {
    return(matrix(
      NA_real_,
      nrow = nrow(matrix_object),
      ncol = ncol(matrix_object)
    ))
  }
  
  cutoff <- tolerance * max(svd_object$d)
  inverse_values <- ifelse(svd_object$d > cutoff, 1 / svd_object$d, 0)
  
  svd_object$v %*% (inverse_values * t(svd_object$u))
}

softmax_by_group <- function(linear_predictor, group) {
  temporary <- data.table(
    row_order = seq_along(linear_predictor),
    group = group,
    linear_predictor = linear_predictor
  )
  
  temporary[, probability := {
    shifted <- linear_predictor - max(linear_predictor)
    exponential <- exp(shifted)
    exponential / sum(exponential)
  }, by = group]
  
  setorder(temporary, row_order)
  temporary$probability
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  mean(x)
}

safe_max <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  max(x)
}

# ----------------------------------------------------------------------------
# 3. Conditional logistic outcome nuisance model
# ----------------------------------------------------------------------------

add_outcome_basis <- function(day_data) {
  output <- copy(day_data)
  
  output[, `:=`(
    sin_x1 = sin(x1),
    x1_sq = x1^2,
    x1_x2 = x1 * x2
  )]
  
  output
}

outcome_covariate_names <- function(specification) {
  if (identical(specification, "correct")) {
    return(c(
      "sin_x1",
      "x1_sq",
      "x2",
      "x1_x2",
      "x3",
      "holiday"
    ))
  }
  
  if (identical(specification, "misspecified")) {
    return(c(
      "x1",
      "x2",
      "x3",
      "holiday"
    ))
  }
  
  stop(
    "Unknown outcome nuisance specification: ",
    specification,
    call. = FALSE
  )
}

make_clogit_formula <- function(specification) {
  covariates <- outcome_covariate_names(specification)
  
  as.formula(
    paste(
      "case ~ heat + cold +",
      paste(covariates, collapse = " + "),
      "+ strata(set_id)"
    )
  )
}

fit_conditional_logistic_model <- function(training_data, specification) {
  model_formula <- make_clogit_formula(specification)
  
  tryCatch(
    clogit(
      formula = model_formula,
      data = training_data,
      method = "efron",
      control = coxph.control(
        iter.max = 50L,
        eps = 1e-9
      )
    ),
    error = function(e) {
      structure(
        list(error_message = conditionMessage(e)),
        class = "failed_clogit"
      )
    }
  )
}

extract_named_coefficients <- function(model, coefficient_names) {
  output <- setNames(
    rep(0, length(coefficient_names)),
    coefficient_names
  )
  
  if (inherits(model, "failed_clogit")) {
    return(output)
  }
  
  estimated <- coef(model)
  common_names <- intersect(names(estimated), coefficient_names)
  
  output[common_names] <- estimated[common_names]
  output[!is.finite(output)] <- 0
  
  output
}

fit_crossfitted_outcome_nuisance <- function(day_data, specification) {
  working <- add_outcome_basis(day_data)
  
  working[, `:=`(
    h_hat = NA_real_,
    outcome_fold_beta_heat = NA_real_,
    outcome_fold_beta_cold = NA_real_
  )]
  
  folds <- sort(unique(working$fold))
  covariate_names <- outcome_covariate_names(specification)
  fold_records <- vector("list", length(folds))
  
  for (fold_index in seq_along(folds)) {
    held_out_fold <- folds[fold_index]
    
    training_data <- working[fold != held_out_fold]
    test_indices <- which(working$fold == held_out_fold)
    test_data <- working[test_indices]
    
    model <- fit_conditional_logistic_model(
      training_data = training_data,
      specification = specification
    )
    
    model_failed <- inherits(model, "failed_clogit")
    
    target_coefficients <- extract_named_coefficients(
      model,
      c("heat", "cold")
    )
    
    nuisance_coefficients <- extract_named_coefficients(
      model,
      covariate_names
    )
    
    test_matrix <- as.matrix(
      test_data[, ..covariate_names]
    )
    
    nuisance_linear_predictor <- as.numeric(
      test_matrix %*% nuisance_coefficients
    )
    
    working[test_indices, `:=`(
      h_hat = nuisance_linear_predictor,
      outcome_fold_beta_heat =
        unname(target_coefficients[["heat"]]),
      outcome_fold_beta_cold =
        unname(target_coefficients[["cold"]])
    )]
    
    fold_records[[fold_index]] <- data.table(
      fold = held_out_fold,
      converged = !model_failed,
      beta_heat = unname(target_coefficients[["heat"]]),
      beta_cold = unname(target_coefficients[["cold"]]),
      error_message = if (model_failed) {
        model$error_message
      } else {
        ""
      }
    )
  }
  
  if (any(!is.finite(working$h_hat))) {
    stop(
      "Cross-fitted outcome nuisance predictions contain non-finite values.",
      call. = FALSE
    )
  }
  
  list(
    day_data = working,
    fold_coefficients = rbindlist(fold_records, fill = TRUE)
  )
}

fit_full_conventional_clogit <- function(day_data, specification) {
  working <- add_outcome_basis(day_data)
  
  model <- fit_conditional_logistic_model(
    training_data = working,
    specification = specification
  )
  
  if (inherits(model, "failed_clogit")) {
    return(list(
      estimator =
        "Conventional conditional logistic regression",
      beta = c(heat = NA_real_, cold = NA_real_),
      se = c(heat = NA_real_, cold = NA_real_),
      converged = FALSE,
      error_message = model$error_message
    ))
  }
  
  coefficients <- coef(model)
  variance_matrix <- tryCatch(
    vcov(model),
    error = function(e) NULL
  )
  
  beta <- c(
    heat = if ("heat" %in% names(coefficients)) {
      unname(coefficients[["heat"]])
    } else {
      NA_real_
    },
    cold = if ("cold" %in% names(coefficients)) {
      unname(coefficients[["cold"]])
    } else {
      NA_real_
    }
  )
  
  se <- c(
    heat = NA_real_,
    cold = NA_real_
  )
  
  if (!is.null(variance_matrix)) {
    standard_errors <- sqrt(
      pmax(diag(variance_matrix), 0)
    )
    
    if ("heat" %in% names(standard_errors)) {
      se[["heat"]] <- unname(
        standard_errors[["heat"]]
      )
    }
    
    if ("cold" %in% names(standard_errors)) {
      se[["cold"]] <- unname(
        standard_errors[["cold"]]
      )
    }
  }
  
  list(
    estimator =
      "Conventional conditional logistic regression",
    beta = beta,
    se = se,
    converged =
      all(is.finite(beta)) &&
      all(is.finite(se)),
    error_message = ""
  )
}

# ----------------------------------------------------------------------------
# 4. Ordered case-control pseudo-pairs
# ----------------------------------------------------------------------------

build_ordered_pseudo_pairs <- function(
    day_data_with_outcome_predictions
) {
  member_columns <- c(
    "row_id",
    "heat",
    "cold",
    "x1",
    "x2",
    "x3",
    "holiday",
    "set_u1",
    "set_u2",
    "h_hat"
  )
  
  set_columns <- c(
    "set_id",
    "scenario_id",
    "scenario_label",
    "replicate",
    "year",
    "fold",
    "n_controls"
  )
  
  missing_member_columns <- setdiff(
    member_columns,
    names(day_data_with_outcome_predictions)
  )
  
  missing_set_columns <- setdiff(
    set_columns,
    names(day_data_with_outcome_predictions)
  )
  
  if (length(missing_member_columns) > 0L ||
      length(missing_set_columns) > 0L) {
    stop(
      "Required columns are missing before pseudo-pair construction. ",
      "Member columns: ",
      paste(missing_member_columns, collapse = ", "),
      "; set columns: ",
      paste(missing_set_columns, collapse = ", "),
      call. = FALSE
    )
  }
  
  controls <- day_data_with_outcome_predictions[
    case == 0L,
    c(set_columns, member_columns),
    with = FALSE
  ]
  
  cases <- day_data_with_outcome_predictions[
    case == 1L,
    c("set_id", member_columns),
    with = FALSE
  ]
  
  setnames(
    controls,
    member_columns,
    paste0(member_columns, "_control")
  )
  
  setnames(
    cases,
    member_columns,
    paste0(member_columns, "_case")
  )
  
  comparisons <- merge(
    controls,
    cases,
    by = "set_id",
    all = FALSE,
    sort = FALSE
  )
  
  comparisons[
    ,
    comparison_index := seq_len(.N),
    by = set_id
  ]
  
  make_orientation <- function(
    first_role,
    second_role,
    d_value
  ) {
    output <- copy(
      comparisons[, ..set_columns]
    )
    
    output[, comparison_index :=
             comparisons$comparison_index]
    output[, D := as.integer(d_value)]
    
    for (variable_name in member_columns) {
      output[[paste0(variable_name, "_first")]] <-
        comparisons[[paste0(variable_name, "_", first_role)]]
      
      output[[paste0(variable_name, "_second")]] <-
        comparisons[[paste0(variable_name, "_", second_role)]]
    }
    
    output
  }
  
  control_first <- make_orientation(
    first_role = "control",
    second_role = "case",
    d_value = 0L
  )
  
  case_first <- make_orientation(
    first_role = "case",
    second_role = "control",
    d_value = 1L
  )
  
  pairs <- rbindlist(
    list(control_first, case_first),
    use.names = TRUE
  )
  
  pairs[, `:=`(
    pair_id = sprintf(
      "%s_P%02d_O%d",
      set_id,
      comparison_index,
      D
    ),
    z_heat = heat_first - heat_second,
    z_cold = cold_first - cold_second,
    g_hat_pair = h_hat_first - h_hat_second,
    pair_weight = 1 / (2 * n_controls),
    exposure_training_weight = 1 / n_controls,
    year_centered = year - mean(2013:2019)
  )]
  
  pairs[, `:=`(
    x1_diff = x1_first - x1_second,
    x2_diff = x2_first - x2_second,
    x3_diff = x3_first - x3_second,
    holiday_diff =
      holiday_first - holiday_second,
    x1_mean =
      0.5 * (x1_first + x1_second),
    x2_mean =
      0.5 * (x2_first + x2_second),
    x3_mean =
      0.5 * (x3_first + x3_second),
    holiday_mean =
      0.5 * (holiday_first + holiday_second),
    x1_abs_diff =
      abs(x1_first - x1_second),
    x2_abs_diff =
      abs(x2_first - x2_second),
    x3_abs_diff =
      abs(x3_first - x3_second),
    year_sine =
      sin(2 * pi * (year - 2013) / 7),
    year_cosine =
      cos(2 * pi * (year - 2013) / 7)
  )]
  
  setorder(
    pairs,
    set_id,
    comparison_index,
    D
  )
  
  pairs[]
}

exposure_feature_names <- function(specification) {
  if (identical(specification, "correct")) {
    return(c(
      "x1_first",
      "x2_first",
      "x3_first",
      "holiday_first",
      "x1_second",
      "x2_second",
      "x3_second",
      "holiday_second",
      "set_u1_first",
      "set_u2_first",
      "set_u1_second",
      "set_u2_second",
      "x1_diff",
      "x2_diff",
      "x3_diff",
      "holiday_diff",
      "x1_mean",
      "x2_mean",
      "x3_mean",
      "holiday_mean",
      "x1_abs_diff",
      "x2_abs_diff",
      "x3_abs_diff",
      "year_centered",
      "year_sine",
      "year_cosine"
    ))
  }
  
  if (identical(specification, "misspecified")) {
    return(c(
      "x1_diff",
      "x2_diff",
      "holiday_diff"
    ))
  }
  
  stop(
    "Unknown exposure nuisance specification: ",
    specification,
    call. = FALSE
  )
}

# ----------------------------------------------------------------------------
# 5. Cross-fitted exposure conditional-mean models
# ----------------------------------------------------------------------------

fixed_xgb_configuration <- function(component, analysis_cfg) {
  row_subsample <- if (
    identical(component, "heat")
  ) {
    0.80
  } else {
    0.85
  }
  
  list(
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = 3L,
      min_child_weight = 5,
      eta = 0.05,
      subsample = row_subsample,
      colsample_bytree = 0.85,
      nthread = analysis_cfg$xgb_threads
    ),
    nrounds = analysis_cfg$xgb_rounds
  )
}

fit_one_xgb_model <- function(
    training_data,
    test_data,
    feature_names,
    outcome_name,
    model_configuration
) {
  training_matrix <- as.matrix(
    training_data[, ..feature_names]
  )
  
  test_matrix <- as.matrix(
    test_data[, ..feature_names]
  )
  
  dtrain <- xgb.DMatrix(
    data = training_matrix,
    label = training_data[[outcome_name]],
    weight =
      training_data$exposure_training_weight
  )
  
  model <- xgb.train(
    params = model_configuration$params,
    data = dtrain,
    nrounds = model_configuration$nrounds,
    verbose = 0
  )
  
  as.numeric(
    predict(model, test_matrix)
  )
}

fit_one_misspecified_linear_exposure_model <- function(
    training_data,
    test_data,
    feature_names,
    outcome_name
) {
  formula_object <- as.formula(
    paste(
      outcome_name,
      "~",
      paste(feature_names, collapse = " + ")
    )
  )
  
  model <- lm(
    formula = formula_object,
    data = training_data,
    weights = exposure_training_weight
  )
  
  as.numeric(
    predict(model, newdata = test_data)
  )
}

fit_crossfitted_exposure_nuisance <- function(
    pairs,
    specification,
    analysis_cfg
) {
  working <- copy(pairs)
  
  working[, `:=`(
    f_hat_heat = NA_real_,
    f_hat_cold = NA_real_
  )]
  
  feature_names <- exposure_feature_names(
    specification
  )
  
  folds <- sort(unique(working$fold))
  
  for (held_out_fold in folds) {
    training_data <- working[
      fold != held_out_fold & D == 0L
    ]
    
    test_indices <- which(
      working$fold == held_out_fold
    )
    
    test_data <- working[test_indices]
    
    if (nrow(training_data) == 0L ||
        nrow(test_data) == 0L) {
      stop(
        "An outer fold has no eligible exposure-model ",
        "training or test records.",
        call. = FALSE
      )
    }
    
    for (component in c("heat", "cold")) {
      outcome_name <- paste0("z_", component)
      
      if (identical(specification, "correct")) {
        predictions <- fit_one_xgb_model(
          training_data = training_data,
          test_data = test_data,
          feature_names = feature_names,
          outcome_name = outcome_name,
          model_configuration =
            fixed_xgb_configuration(
              component = component,
              analysis_cfg = analysis_cfg
            )
        )
      } else {
        predictions <-
          fit_one_misspecified_linear_exposure_model(
            training_data = training_data,
            test_data = test_data,
            feature_names = feature_names,
            outcome_name = outcome_name
          )
      }
      
      prediction_column <- paste0(
        "f_hat_",
        component
      )
      
      working[
        test_indices,
        (prediction_column) := predictions
      ]
    }
  }
  
  if (any(!is.finite(working$f_hat_heat)) ||
      any(!is.finite(working$f_hat_cold))) {
    stop(
      "Cross-fitted exposure nuisance predictions ",
      "contain non-finite values.",
      call. = FALSE
    )
  }
  
  working[]
}

# ----------------------------------------------------------------------------
# 6. Generalized estimating equations and clustered sandwich variance
# ----------------------------------------------------------------------------

prepare_ee_arrays <- function(
    pairs,
    g_values,
    f_matrix
) {
  list(
    Z = cbind(
      pairs$z_heat,
      pairs$z_cold
    ),
    D = as.numeric(pairs$D),
    g = as.numeric(g_values),
    f = as.matrix(f_matrix),
    weights =
      as.numeric(pairs$pair_weight),
    set_factor =
      factor(pairs$set_id),
    n_sets =
      uniqueN(pairs$set_id)
  )
}

compute_ee_moments <- function(
    beta,
    arrays,
    return_rows = FALSE
) {
  eta <- as.numeric(
    arrays$Z %*% beta + arrays$g
  )
  
  negative_eta <- pmin(
    pmax(-eta, -700),
    700
  )
  
  multiplier <- ifelse(
    arrays$D == 1,
    exp(negative_eta),
    -1
  )
  
  residualized_exposure <-
    arrays$Z - arrays$f
  
  row_score <-
    residualized_exposure *
    multiplier *
    arrays$weights
  
  mean_score <-
    colSums(row_score) /
    arrays$n_sets
  
  derivative_weight <-
    -arrays$D *
    exp(negative_eta) *
    arrays$weights
  
  jacobian <- matrix(
    0,
    nrow = 2L,
    ncol = 2L
  )
  
  for (a in 1:2) {
    for (b in 1:2) {
      jacobian[a, b] <- sum(
        residualized_exposure[, a] *
          arrays$Z[, b] *
          derivative_weight
      ) / arrays$n_sets
    }
  }
  
  output <- list(
    mean_score = mean_score,
    jacobian = jacobian
  )
  
  if (isTRUE(return_rows)) {
    output$row_score <- row_score
  }
  
  output
}

newton_solve_one_start <- function(
    starting_beta,
    arrays,
    analysis_cfg
) {
  beta <- as.numeric(starting_beta)
  
  current <- compute_ee_moments(
    beta = beta,
    arrays = arrays
  )
  
  current_norm <- sqrt(
    sum(current$mean_score^2)
  )
  
  converged <- FALSE
  iteration <- 0L
  
  for (iteration in seq_len(
    analysis_cfg$maximum_newton_iterations
  )) {
    if (!all(is.finite(current$mean_score)) ||
        !all(is.finite(current$jacobian))) {
      break
    }
    
    if (current_norm <=
        analysis_cfg$score_tolerance) {
      converged <- TRUE
      break
    }
    
    ridge_sequence <- c(
      0,
      1e-10,
      1e-8,
      1e-6,
      1e-4
    )
    
    step <- NULL
    
    for (ridge in ridge_sequence) {
      candidate_step <- tryCatch(
        solve(
          current$jacobian -
            diag(ridge, 2L),
          current$mean_score
        ),
        error = function(e) NULL
      )
      
      if (!is.null(candidate_step) &&
          all(is.finite(candidate_step))) {
        step <- as.numeric(candidate_step)
        break
      }
    }
    
    if (is.null(step)) {
      generalized_inverse <-
        stable_matrix_inverse(
          current$jacobian
        )
      
      step <- as.numeric(
        generalized_inverse %*%
          current$mean_score
      )
    }
    
    if (!all(is.finite(step))) {
      break
    }
    
    accepted <- FALSE
    best_beta <- beta
    best_moments <- current
    best_norm <- current_norm
    
    for (damping in 2^-(0:15)) {
      candidate_beta <-
        beta - damping * step
      
      candidate_moments <-
        compute_ee_moments(
          beta = candidate_beta,
          arrays = arrays
        )
      
      candidate_norm <- sqrt(
        sum(candidate_moments$mean_score^2)
      )
      
      if (is.finite(candidate_norm) &&
          candidate_norm < best_norm) {
        best_beta <- candidate_beta
        best_moments <- candidate_moments
        best_norm <- candidate_norm
        accepted <- TRUE
        break
      }
    }
    
    if (!accepted) {
      break
    }
    
    parameter_change <- max(
      abs(best_beta - beta)
    )
    
    beta <- best_beta
    current <- best_moments
    current_norm <- best_norm
    
    if (current_norm <=
        analysis_cfg$score_tolerance &&
        parameter_change <=
        analysis_cfg$parameter_tolerance) {
      converged <- TRUE
      break
    }
  }
  
  list(
    beta = beta,
    converged =
      converged ||
      current_norm <=
      analysis_cfg$score_tolerance,
    iterations = iteration,
    score_norm = current_norm
  )
}

solve_generalized_ee <- function(
    arrays,
    starting_values,
    analysis_cfg
) {
  unique_starts <- unique(
    do.call(
      rbind,
      lapply(starting_values, as.numeric)
    )
  )
  
  solutions <- lapply(
    seq_len(nrow(unique_starts)),
    function(index) {
      newton_solve_one_start(
        starting_beta =
          unique_starts[index, ],
        arrays = arrays,
        analysis_cfg = analysis_cfg
      )
    }
  )
  
  convergence_indicator <- vapply(
    solutions,
    `[[`,
    logical(1),
    "converged"
  )
  
  score_norms <- vapply(
    solutions,
    `[[`,
    numeric(1),
    "score_norm"
  )
  
  if (any(convergence_indicator)) {
    candidate_indices <- which(
      convergence_indicator
    )
    
    best_index <- candidate_indices[
      which.min(
        score_norms[candidate_indices]
      )
    ]
  } else {
    best_index <- which.min(score_norms)
  }
  
  solutions[[best_index]]
}

compute_clustered_sandwich <- function(
    beta,
    arrays
) {
  moments <- compute_ee_moments(
    beta = beta,
    arrays = arrays,
    return_rows = TRUE
  )
  
  set_scores <- rowsum(
    moments$row_score,
    arrays$set_factor,
    reorder = FALSE
  )
  
  meat <- crossprod(set_scores) /
    arrays$n_sets
  
  bread_inverse <-
    stable_matrix_inverse(
      moments$jacobian
    )
  
  covariance <-
    bread_inverse %*%
    meat %*%
    t(bread_inverse) /
    arrays$n_sets
  
  standard_errors <- sqrt(
    pmax(diag(covariance), 0)
  )
  
  list(
    covariance = covariance,
    standard_errors = standard_errors
  )
}

fit_one_ee_estimator <- function(
    estimator_name,
    pairs,
    g_values,
    f_matrix,
    starting_values,
    analysis_cfg
) {
  arrays <- prepare_ee_arrays(
    pairs = pairs,
    g_values = g_values,
    f_matrix = f_matrix
  )
  
  solution <- solve_generalized_ee(
    arrays = arrays,
    starting_values = starting_values,
    analysis_cfg = analysis_cfg
  )
  
  if (!all(is.finite(solution$beta))) {
    return(list(
      estimator = estimator_name,
      beta = c(
        heat = NA_real_,
        cold = NA_real_
      ),
      se = c(
        heat = NA_real_,
        cold = NA_real_
      ),
      converged = FALSE,
      iterations = solution$iterations,
      score_norm = solution$score_norm
    ))
  }
  
  sandwich <- compute_clustered_sandwich(
    beta = solution$beta,
    arrays = arrays
  )
  
  standard_errors <-
    sandwich$standard_errors
  
  list(
    estimator = estimator_name,
    beta = c(
      heat = unname(solution$beta[1L]),
      cold = unname(solution$beta[2L])
    ),
    se = c(
      heat = unname(standard_errors[1L]),
      cold = unname(standard_errors[2L])
    ),
    converged =
      isTRUE(solution$converged) &&
      all(is.finite(solution$beta)) &&
      all(is.finite(standard_errors)),
    iterations = solution$iterations,
    score_norm = solution$score_norm
  )
}

# ----------------------------------------------------------------------------
# 7. Mean annual total attributable fractions
# ----------------------------------------------------------------------------

compute_mean_annual_total_af <- function(
    day_data,
    estimator_name,
    beta_heat,
    beta_cold
) {
  cases <- day_data[case == 1L]
  
  if (nrow(cases) == 0L) {
    return(data.table(
      estimator = estimator_name,
      estimated_af = NA_real_,
      true_af = NA_real_
    ))
  }
  
  estimated_log_rate <-
    beta_heat * cases$heat +
    beta_cold * cases$cold
  
  true_log_rate <-
    cases$true_beta_heat *
    cases$heat +
    cases$true_beta_cold *
    cases$cold +
    cases$true_gamma_heat2 *
    cases$heat^2 +
    cases$true_gamma_cold2 *
    cases$cold^2 +
    cases$true_gamma_heat_cold *
    cases$heat *
    cases$cold
  
  cases[, estimated_total_af :=
          1 - exp(-estimated_log_rate)]
  
  cases[, true_total_af :=
          1 - exp(-true_log_rate)]
  
  annual <- cases[, .(
    estimated_af = mean(
      estimated_total_af
    ),
    true_af = mean(
      true_total_af
    )
  ), by = year]
  
  data.table(
    estimator = estimator_name,
    estimated_af =
      mean(annual$estimated_af),
    true_af =
      mean(annual$true_af)
  )
}

# ----------------------------------------------------------------------------
# 8. One-dataset analysis
# ----------------------------------------------------------------------------

make_estimator_result_rows <- function(
    fit_object,
    scenario_id,
    replicate_id,
    true_beta_heat,
    true_beta_cold
) {
  parameters <- c("heat", "cold")
  
  true_values <- c(
    heat = true_beta_heat,
    cold = true_beta_cold
  )
  
  rbindlist(
    lapply(parameters, function(parameter_name) {
      estimate <-
        unname(
          fit_object$beta[[parameter_name]]
        )
      
      standard_error <-
        unname(
          fit_object$se[[parameter_name]]
        )
      
      lower <- estimate -
        qnorm(0.975) * standard_error
      
      upper <- estimate +
        qnorm(0.975) * standard_error
      
      coverage_indicator <- if (
        all(is.finite(c(
          lower,
          upper,
          true_values[[parameter_name]]
        )))
      ) {
        as.integer(
          true_values[[parameter_name]] >=
            lower &&
            true_values[[parameter_name]] <=
            upper
        )
      } else {
        NA_integer_
      }
      
      data.table(
        scenario_id = scenario_id,
        replicate = replicate_id,
        estimator = fit_object$estimator,
        parameter = parameter_name,
        true_beta =
          true_values[[parameter_name]],
        estimate_beta = estimate,
        standard_error = standard_error,
        covered_beta = coverage_indicator,
        converged =
          isTRUE(fit_object$converged)
      )
    }),
    fill = TRUE
  )
}

analyze_one_simulated_dataset <- function(
    day_data,
    scenario_row,
    analysis_cfg
) {
  scenario_id <- unique(day_data$scenario_id)
  replicate_id <- unique(day_data$replicate)
  
  if (length(scenario_id) != 1L ||
      length(replicate_id) != 1L) {
    stop(
      "Each input file must contain exactly one ",
      "scenario and one replicate.",
      call. = FALSE
    )
  }
  
  outcome_specification <-
    scenario_row$outcome_nuisance_spec
  
  exposure_specification <-
    scenario_row$exposure_nuisance_spec
  
  outcome_fit <-
    fit_crossfitted_outcome_nuisance(
      day_data = day_data,
      specification =
        outcome_specification
    )
  
  conventional_fit <-
    fit_full_conventional_clogit(
      day_data = day_data,
      specification =
        outcome_specification
    )
  
  pairs <- build_ordered_pseudo_pairs(
    outcome_fit$day_data
  )
  
  pairs <- fit_crossfitted_exposure_nuisance(
    pairs = pairs,
    specification =
      exposure_specification,
    analysis_cfg = analysis_cfg
  )
  
  conventional_beta <-
    conventional_fit$beta
  
  fold_beta_start <- c(
    heat = mean(
      outcome_fit$fold_coefficients$beta_heat,
      na.rm = TRUE
    ),
    cold = mean(
      outcome_fit$fold_coefficients$beta_cold,
      na.rm = TRUE
    )
  )
  
  fold_beta_start[
    !is.finite(fold_beta_start)
  ] <- 0
  
  conventional_beta[
    !is.finite(conventional_beta)
  ] <- fold_beta_start[
    !is.finite(conventional_beta)
  ]
  
  conventional_beta[
    !is.finite(conventional_beta)
  ] <- 0
  
  starting_values <- list(
    conventional_beta,
    fold_beta_start,
    c(0, 0),
    conventional_beta + c(0.02, -0.02),
    conventional_beta + c(-0.02, 0.02)
  )
  
  zero_f <- matrix(
    0,
    nrow = nrow(pairs),
    ncol = 2L
  )
  
  fitted_f <- cbind(
    pairs$f_hat_heat,
    pairs$f_hat_cold
  )
  
  outcome_only <- fit_one_ee_estimator(
    estimator_name =
      "Outcome-model EE",
    pairs = pairs,
    g_values = pairs$g_hat_pair,
    f_matrix = zero_f,
    starting_values = starting_values,
    analysis_cfg = analysis_cfg
  )
  
  exposure_only <- fit_one_ee_estimator(
    estimator_name =
      "Exposure-regression EE",
    pairs = pairs,
    g_values = rep(0, nrow(pairs)),
    f_matrix = fitted_f,
    starting_values = starting_values,
    analysis_cfg = analysis_cfg
  )
  
  proposed_aipw <- fit_one_ee_estimator(
    estimator_name =
      "Proposed generalized AIPW",
    pairs = pairs,
    g_values = pairs$g_hat_pair,
    f_matrix = fitted_f,
    starting_values = starting_values,
    analysis_cfg = analysis_cfg
  )
  
  estimator_fits <- list(
    conventional_fit,
    outcome_only,
    exposure_only,
    proposed_aipw
  )
  
  true_beta_heat <- unique(
    day_data$true_beta_heat
  )
  
  true_beta_cold <- unique(
    day_data$true_beta_cold
  )
  
  if (length(true_beta_heat) != 1L ||
      length(true_beta_cold) != 1L) {
    stop(
      "The true heat- and cold-side coefficients ",
      "must be constant within each simulated dataset.",
      call. = FALSE
    )
  }
  
  coefficient_rows <- rbindlist(
    lapply(estimator_fits, function(fit_object) {
      make_estimator_result_rows(
        fit_object = fit_object,
        scenario_id = scenario_id,
        replicate_id = replicate_id,
        true_beta_heat = true_beta_heat,
        true_beta_cold = true_beta_cold
      )
    }),
    fill = TRUE
  )
  
  mean_annual_af_rows <- rbindlist(
    lapply(estimator_fits, function(fit_object) {
      result <- compute_mean_annual_total_af(
        day_data = day_data,
        estimator_name =
          fit_object$estimator,
        beta_heat =
          fit_object$beta[["heat"]],
        beta_cold =
          fit_object$beta[["cold"]]
      )
      
      result[, `:=`(
        scenario_id = scenario_id,
        replicate = replicate_id
      )]
      
      result
    }),
    fill = TRUE
  )
  
  list(
    coefficient_rows = coefficient_rows,
    mean_annual_af_rows =
      mean_annual_af_rows
  )
}

make_failed_dataset_result <- function(
    day_data,
    scenario_id,
    replicate_id,
    error_message
) {
  true_beta_heat <- if (
    "true_beta_heat" %in% names(day_data)
  ) {
    unique(day_data$true_beta_heat)[1L]
  } else {
    NA_real_
  }
  
  true_beta_cold <- if (
    "true_beta_cold" %in% names(day_data)
  ) {
    unique(day_data$true_beta_cold)[1L]
  } else {
    NA_real_
  }
  
  coefficient_rows <- CJ(
    estimator = estimators_to_report,
    parameter = c("heat", "cold"),
    unique = TRUE
  )
  
  coefficient_rows[, `:=`(
    scenario_id = scenario_id,
    replicate = replicate_id,
    true_beta = fifelse(
      parameter == "heat",
      true_beta_heat,
      true_beta_cold
    ),
    estimate_beta = NA_real_,
    standard_error = NA_real_,
    covered_beta = NA_integer_,
    converged = FALSE
  )]
  
  mean_annual_af_rows <- data.table(
    scenario_id = scenario_id,
    replicate = replicate_id,
    estimator = estimators_to_report,
    estimated_af = NA_real_,
    true_af = NA_real_
  )
  
  list(
    coefficient_rows = coefficient_rows,
    mean_annual_af_rows =
      mean_annual_af_rows,
    error_message = error_message
  )
}

# ----------------------------------------------------------------------------
# 9. Main loop with restartable dataset-level checkpoints
# ----------------------------------------------------------------------------

message("Input datasets: ", nrow(manifest))
message("XGBoost threads per model: ", XGB_THREADS)
message("Resume completed datasets: ", RESUME_COMPLETED)
message(
  "Results directory: ",
  normalizePath(
    RESULT_DIR,
    mustWork = FALSE
  )
)

coefficient_result_list <- vector(
  "list",
  nrow(manifest)
)

af_result_list <- vector(
  "list",
  nrow(manifest)
)

error_records <- list()
error_index <- 0L

for (manifest_index in seq_len(nrow(manifest))) {
  manifest_row <- manifest[manifest_index]
  
  checkpoint_path <- file.path(
    CHECKPOINT_DIR,
    sprintf(
      "%s_rep%04d_selected_metrics.rds",
      manifest_row$scenario_id,
      manifest_row$replicate
    )
  )
  
  if (RESUME_COMPLETED &&
      file.exists(checkpoint_path)) {
    checkpoint <- readRDS(checkpoint_path)
    
    coefficient_result_list[[manifest_index]] <- checkpoint$coefficient_rows
    
    af_result_list[[manifest_index]] <- checkpoint$mean_annual_af_rows
    
    next
  }
  
  message(
    "Analyzing ",
    manifest_row$scenario_id,
    ", replicate ",
    manifest_row$replicate,
    " (",
    manifest_index,
    "/",
    nrow(manifest),
    ")"
  )
  
  day_data <- readRDS(
    manifest_row$file_path
  )
  
  scenario_row <- as.list(
    scenario_design[
      scenario_id ==
        manifest_row$scenario_id
    ][1L]
  )
  
  analysis_result <- tryCatch(
    analyze_one_simulated_dataset(
      day_data = day_data,
      scenario_row = scenario_row,
      analysis_cfg = analysis_cfg
    ),
    error = function(e) {
      make_failed_dataset_result(
        day_data = day_data,
        scenario_id =
          manifest_row$scenario_id,
        replicate_id =
          manifest_row$replicate,
        error_message =
          conditionMessage(e)
      )
    }
  )
  
  coefficient_result_list[[manifest_index]] <- analysis_result$coefficient_rows
  
  af_result_list[[manifest_index]] <- analysis_result$mean_annual_af_rows
  
  if (!is.null(
    analysis_result$error_message
  )) {
    error_index <- error_index + 1L
    
    error_records[[error_index]] <- data.table(
      scenario_id =
        manifest_row$scenario_id,
      replicate =
        manifest_row$replicate,
      file_path =
        manifest_row$file_path,
      error_message =
        analysis_result$error_message
    )
  }
  
  saveRDS(
    list(
      coefficient_rows =
        analysis_result$coefficient_rows,
      mean_annual_af_rows =
        analysis_result$mean_annual_af_rows
    ),
    checkpoint_path
  )
}

# ----------------------------------------------------------------------------
# 10. Combine minimal replicate-level results
# ----------------------------------------------------------------------------

coefficient_results <- rbindlist(
  coefficient_result_list,
  fill = TRUE
)

mean_annual_af_results <- rbindlist(
  af_result_list,
  fill = TRUE
)

if (length(error_records) == 0L) {
  analysis_errors <- data.table(
    scenario_id = character(),
    replicate = integer(),
    file_path = character(),
    error_message = character()
  )
} else {
  analysis_errors <- rbindlist(
    error_records,
    fill = TRUE
  )
}

minimal_results_path <- file.path(
  RESULT_DIR,
  "temperature_aipw_selected_metric_replicate_results.rds"
)

saveRDS(
  list(
    coefficient_results =
      coefficient_results,
    mean_annual_af_results =
      mean_annual_af_results,
    analysis_errors =
      analysis_errors,
    analysis_configuration =
      analysis_cfg,
    simulation_configuration =
      simulation_configuration
  ),
  minimal_results_path
)

if (nrow(analysis_errors) > 0L) {
  fwrite(
    analysis_errors,
    file.path(
      RESULT_DIR,
      "temperature_aipw_selected_metric_analysis_errors.csv"
    )
  )
  
  warning(
    nrow(analysis_errors),
    " simulated dataset(s) failed. ",
    "See temperature_aipw_selected_metric_analysis_errors.csv.",
    call. = FALSE
  )
}

# ----------------------------------------------------------------------------
# 11. Calculate only the retained performance metrics
# ----------------------------------------------------------------------------

coefficient_performance <- coefficient_results[
  ,
  {
    valid_estimate <-
      is.finite(estimate_beta) &
      is.finite(true_beta)
    
    valid_coverage <-
      !is.na(covered_beta)
    
    list(
      n_replicates = .N,
      empirical_bias = if (
        any(valid_estimate)
      ) {
        mean(
          estimate_beta[valid_estimate] -
            true_beta[valid_estimate]
        )
      } else {
        NA_real_
      },
      empirical_coverage_95 = if (
        any(valid_coverage)
      ) {
        mean(
          covered_beta[valid_coverage]
        )
      } else {
        NA_real_
      },
      convergence_rate = mean(
        converged &
          is.finite(estimate_beta)
      )
    )
  },
  by = .(
    scenario_id,
    estimator,
    parameter
  )
]

mean_annual_af_performance <-
  mean_annual_af_results[
    ,
    {
      valid <- is.finite(estimated_af) &
        is.finite(true_af)
      
      list(
        n_replicates = .N,
        empirical_bias = if (
          any(valid)
        ) {
          mean(
            estimated_af[valid] -
              true_af[valid]
          )
        } else {
          NA_real_
        }
      )
    },
    by = .(
      scenario_id,
      estimator
    )
  ]

heat_summary <- coefficient_performance[
  parameter == "heat",
  .(
    scenario_id,
    estimator,
    heat_empirical_bias =
      empirical_bias,
    heat_absolute_bias =
      abs(empirical_bias),
    heat_coverage_percent =
      100 * empirical_coverage_95,
    heat_convergence_percent =
      100 * convergence_rate
  )
]

cold_summary <- coefficient_performance[
  parameter == "cold",
  .(
    scenario_id,
    estimator,
    cold_empirical_bias =
      empirical_bias,
    cold_absolute_bias =
      abs(empirical_bias),
    cold_coverage_percent =
      100 * empirical_coverage_95,
    cold_convergence_percent =
      100 * convergence_rate
  )
]

scenario_performance <- merge(
  heat_summary,
  cold_summary,
  by = c(
    "scenario_id",
    "estimator"
  ),
  all = TRUE
)

scenario_performance <- merge(
  scenario_performance,
  mean_annual_af_performance[
    ,
    .(
      scenario_id,
      estimator,
      mean_annual_total_af_bias =
        empirical_bias,
      mean_annual_total_af_absolute_bias =
        abs(empirical_bias)
    )
  ],
  by = c(
    "scenario_id",
    "estimator"
  ),
  all = TRUE
)

scenario_performance[, `:=`(
  mean_absolute_coefficient_bias =
    rowMeans(
      cbind(
        heat_absolute_bias,
        cold_absolute_bias
      ),
      na.rm = FALSE
    ),
  mean_coverage_percent =
    rowMeans(
      cbind(
        heat_coverage_percent,
        cold_coverage_percent
      ),
      na.rm = FALSE
    ),
  convergence_percent =
    pmin(
      heat_convergence_percent,
      cold_convergence_percent,
      na.rm = FALSE
    )
)]

scenario_performance[
  ,
  scenario_rank :=
    match(
      scenario_id,
      scenarios_to_report
    )
]

scenario_performance[
  ,
  estimator_rank :=
    match(
      estimator,
      estimators_to_report
    )
]

setorder(
  scenario_performance,
  scenario_rank,
  estimator_rank
)

scenario_performance[
  ,
  c(
    "scenario_rank",
    "estimator_rank"
  ) := NULL
]

primary_overall_summary <-
  scenario_performance[
    scenario_id %in%
      double_robustness_scenarios,
    .(
      overall_mean_absolute_coefficient_bias =
        safe_mean(
          mean_absolute_coefficient_bias
        ),
      worst_case_absolute_coefficient_bias =
        safe_max(
          mean_absolute_coefficient_bias
        ),
      overall_mean_absolute_mean_annual_af_bias =
        safe_mean(
          mean_annual_total_af_absolute_bias
        ),
      mean_coverage_percent =
        safe_mean(
          mean_coverage_percent
        )
    ),
    by = estimator
  ]

all_scenario_convergence <-
  scenario_performance[
    ,
    .(
      minimum_convergence_percent =
        if (
          all(
            !is.finite(
              convergence_percent
            )
          )
        ) {
          NA_real_
        } else {
          min(
            convergence_percent,
            na.rm = TRUE
          )
        }
    ),
    by = estimator
  ]

primary_overall_summary <- merge(
  primary_overall_summary,
  all_scenario_convergence,
  by = "estimator",
  all = TRUE
)

# ----------------------------------------------------------------------------
# 12. Construct the transposed manuscript-ready table
# ----------------------------------------------------------------------------

scenario_metric_rows <- rbindlist(
  lapply(
    scenarios_to_report,
    function(current_scenario) {
      current_values <- scenario_performance[
        scenario_id == current_scenario
      ]
      
      value_vector <- setNames(
        current_values$
          mean_absolute_coefficient_bias,
        current_values$estimator
      )
      
      data.table(
        performance_metric = paste0(
          scenario_display_labels[[current_scenario]],
          ": mean absolute coefficient bias \u2193"
        ),
        `Conventional conditional logistic regression` =
          unname(
            value_vector[[
              "Conventional conditional logistic regression"
            ]]
          ),
        `Outcome-model EE` =
          unname(
            value_vector[["Outcome-model EE"]]
          ),
        `Exposure-regression EE` =
          unname(
            value_vector[["Exposure-regression EE"]]
          ),
        `Proposed generalized AIPW` =
          unname(
            value_vector[["Proposed generalized AIPW"]]
          )
      )
    }
  ),
  fill = TRUE
)

overall_value <- function(
    metric_column,
    estimator_name
) {
  primary_overall_summary[
    estimator == estimator_name,
    get(metric_column)
  ][1L]
}

summary_metric_rows <- data.table(
  performance_metric = c(
    paste0(
      "Overall mean absolute coefficient bias across ",
      "the three double-robustness scenarios \u2193"
    ),
    paste0(
      "Worst-case absolute coefficient bias across ",
      "the three double-robustness scenarios \u2193"
    ),
    paste0(
      "Overall mean absolute mean annual AF bias across ",
      "the three double-robustness scenarios \u2193"
    ),
    paste0(
      "Mean 95% confidence interval coverage across ",
      "the three double-robustness scenarios (%) \u2192 95"
    ),
    "Minimum convergence across all four scenarios (%) \u2191"
  )
)

for (estimator_name in estimators_to_report) {
  summary_metric_rows[
    ,
    (estimator_name) := c(
      overall_value(
        "overall_mean_absolute_coefficient_bias",
        estimator_name
      ),
      overall_value(
        "worst_case_absolute_coefficient_bias",
        estimator_name
      ),
      overall_value(
        "overall_mean_absolute_mean_annual_af_bias",
        estimator_name
      ),
      overall_value(
        "mean_coverage_percent",
        estimator_name
      ),
      overall_value(
        "minimum_convergence_percent",
        estimator_name
      )
    )
  ]
}

transposed_summary_table <- rbind(
  scenario_metric_rows,
  summary_metric_rows,
  fill = TRUE
)

setcolorder(
  transposed_summary_table,
  c(
    "performance_metric",
    estimators_to_report
  )
)

# ----------------------------------------------------------------------------
# 13. Export the single formatted Excel summary table
# ----------------------------------------------------------------------------

workbook <- createWorkbook()

sheet_name <- "Simulation Summary"

addWorksheet(
  workbook,
  sheet_name,
  gridLines = FALSE
)

mergeCells(
  workbook,
  sheet = sheet_name,
  cols = 1:5,
  rows = 1
)

writeData(
  workbook,
  sheet = sheet_name,
  x = paste0(
    "Simulation performance across four nuisance-model ",
    "specification scenarios"
  ),
  startRow = 1,
  startCol = 1
)

mergeCells(
  workbook,
  sheet = sheet_name,
  cols = 1:5,
  rows = 2
)

writeData(
  workbook,
  sheet = sheet_name,
  x = paste0(
    "Scenario-specific results are shown for all four scenarios; ",
    "overall summaries use the three scenarios in which at least ",
    "one nuisance model was correctly specified."
  ),
  startRow = 2,
  startCol = 1
)

display_table <- copy(
  transposed_summary_table
)

setnames(
  display_table,
  "performance_metric",
  "Performance metric"
)

writeDataTable(
  workbook,
  sheet = sheet_name,
  x = as.data.frame(display_table),
  startRow = 4,
  startCol = 1,
  tableStyle = "TableStyleMedium2",
  withFilter = FALSE
)

title_style <- createStyle(
  fontSize = 14,
  fontColour = "#FFFFFF",
  fgFill = "#1F4E78",
  textDecoration = "bold",
  halign = "center",
  valign = "center"
)

subtitle_style <- createStyle(
  fontSize = 9,
  fontColour = "#404040",
  textDecoration = "italic",
  halign = "center",
  valign = "center",
  wrapText = TRUE
)

header_style <- createStyle(
  fontColour = "#FFFFFF",
  fgFill = "#5B9BD5",
  textDecoration = "bold",
  halign = "center",
  valign = "center",
  wrapText = TRUE,
  border = "TopBottomLeftRight",
  borderColour = "#D9E2F3"
)

body_style <- createStyle(
  halign = "center",
  valign = "center",
  wrapText = TRUE,
  border = "TopBottomLeftRight",
  borderColour = "#D9D9D9"
)

metric_style <- createStyle(
  halign = "left",
  valign = "center",
  wrapText = TRUE,
  border = "TopBottomLeftRight",
  borderColour = "#D9D9D9"
)

summary_metric_style <- createStyle(
  fgFill = "#D9EAF7",
  textDecoration = "bold",
  halign = "left",
  valign = "center",
  wrapText = TRUE,
  border = "TopBottomLeftRight",
  borderColour = "#A6A6A6"
)

aipw_style <- createStyle(
  fgFill = "#E2F0D9",
  textDecoration = "bold",
  halign = "center",
  valign = "center",
  border = "TopBottomLeftRight",
  borderColour = "#70AD47"
)

bias_number_style <- createStyle(
  numFmt = "0.0000",
  halign = "center",
  valign = "center"
)

percent_number_style <- createStyle(
  numFmt = "0.00",
  halign = "center",
  valign = "center"
)

note_header_style <- createStyle(
  textDecoration = "bold",
  fontSize = 10,
  halign = "left",
  valign = "top"
)

note_style <- createStyle(
  fontSize = 9,
  fontColour = "#595959",
  halign = "left",
  valign = "top",
  wrapText = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = title_style,
  rows = 1,
  cols = 1:5,
  gridExpand = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = subtitle_style,
  rows = 2,
  cols = 1:5,
  gridExpand = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = header_style,
  rows = 4,
  cols = 1:5,
  gridExpand = TRUE
)

data_start_row <- 5L
data_end_row <-
  data_start_row +
  nrow(display_table) -
  1L

addStyle(
  workbook,
  sheet = sheet_name,
  style = body_style,
  rows = data_start_row:data_end_row,
  cols = 1:5,
  gridExpand = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = metric_style,
  rows = data_start_row:data_end_row,
  cols = 1,
  gridExpand = TRUE,
  stack = TRUE
)

summary_start_row <-
  data_start_row +
  length(scenarios_to_report)

addStyle(
  workbook,
  sheet = sheet_name,
  style = summary_metric_style,
  rows = summary_start_row:data_end_row,
  cols = 1,
  gridExpand = TRUE,
  stack = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = aipw_style,
  rows = data_start_row:data_end_row,
  cols = 5,
  gridExpand = TRUE,
  stack = TRUE
)

bias_rows <- data_start_row:(
  summary_start_row + 2L
)

coverage_and_convergence_rows <- (
  summary_start_row + 3L
):data_end_row

addStyle(
  workbook,
  sheet = sheet_name,
  style = bias_number_style,
  rows = bias_rows,
  cols = 2:5,
  gridExpand = TRUE,
  stack = TRUE
)

addStyle(
  workbook,
  sheet = sheet_name,
  style = percent_number_style,
  rows = coverage_and_convergence_rows,
  cols = 2:5,
  gridExpand = TRUE,
  stack = TRUE
)

notes <- c(
  "Notes:",
  paste0(
    "The first four rows show scenario-specific mean absolute ",
    "coefficient bias, calculated as the average of the absolute ",
    "empirical biases for the heat- and cold-side log rate-ratio ",
    "coefficients."
  ),
  paste0(
    "The overall coefficient-bias, worst-case-bias, attributable-",
    "fraction-bias, and confidence-interval-coverage summaries are ",
    "restricted to the three scenarios in which at least one ",
    "nuisance model was correctly specified."
  ),
  paste0(
    "The both-misspecified scenario is displayed to show the ",
    "boundary of the double-robustness property but is not pooled ",
    "into the primary overall summaries."
  ),
  paste0(
    "Arrows indicate the preferred direction: \u2193 lower is better; ",
    "\u2191 higher is better; confidence interval coverage is best ",
    "when close to 95%."
  ),
  paste0(
    "AF, attributable fraction; AIPW, augmented inverse probability ",
    "weighting; EE, estimating equation."
  )
)

note_start_row <- data_end_row + 2L

for (note_index in seq_along(notes)) {
  current_row <-
    note_start_row +
    note_index -
    1L
  
  mergeCells(
    workbook,
    sheet = sheet_name,
    cols = 1:5,
    rows = current_row
  )
  
  writeData(
    workbook,
    sheet = sheet_name,
    x = notes[note_index],
    startRow = current_row,
    startCol = 1
  )
  
  addStyle(
    workbook,
    sheet = sheet_name,
    style = if (note_index == 1L) {
      note_header_style
    } else {
      note_style
    },
    rows = current_row,
    cols = 1:5,
    gridExpand = TRUE
  )
}

setColWidths(
  workbook,
  sheet = sheet_name,
  cols = 1,
  widths = 52
)

setColWidths(
  workbook,
  sheet = sheet_name,
  cols = 2:5,
  widths = 22
)

setRowHeights(
  workbook,
  sheet = sheet_name,
  rows = 1,
  heights = 30
)

setRowHeights(
  workbook,
  sheet = sheet_name,
  rows = 2,
  heights = 34
)

setRowHeights(
  workbook,
  sheet = sheet_name,
  rows = 4,
  heights = 52
)

setRowHeights(
  workbook,
  sheet = sheet_name,
  rows = data_start_row:data_end_row,
  heights = 34
)

freezePane(
  workbook,
  sheet = sheet_name,
  firstActiveRow = 5,
  firstActiveCol = 2
)

workbook_path <- file.path(
  RESULT_DIR,
  "temperature_aipw_simulation_summary_transposed.xlsx"
)

saveWorkbook(
  workbook,
  workbook_path,
  overwrite = TRUE
)

# Save a simple CSV version of the same transposed table for checking.
fwrite(
  transposed_summary_table,
  file.path(
    RESULT_DIR,
    "temperature_aipw_simulation_summary_transposed.csv"
  )
)

message("Simulation analysis completed successfully.")
message("Selected-metric replicate results: ", minimal_results_path)
message("Final transposed Excel table: ", workbook_path)