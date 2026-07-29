####################################################################################################
# REVIEWER-READY ANALYSIS CODE
# Cross-fitted generalized augmented inverse probability weighting estimator for conditional
# mortality rate ratios in a time-stratified case-crossover design
####################################################################################################
#
# OVERVIEW
# --------
# This script implements the outcome-specific primary analysis described in the accompanying
# methodological appendix. It estimates the conditional mortality rate ratios associated with
# heat-side and cold-side temperature deviations in a time-stratified case-crossover design.
#
# The implementation has four principal components:
#
#   1. Exposure construction
#      A_hot  = max(mean temperature over HOT_LAGS  - MMT, 0)
#      A_cold = max(MMT - mean temperature over COLD_LAGS, 0)
#
#   2. Prospective outcome nuisance model
#      A cross-fitted conditional logistic regression is fitted to the original matched risk sets.
#      XGBoost is not used for the outcome model.
#
#   3. Retrospective exposure nuisance model
#      Ordered case-control pseudo-pairs are constructed. Separate cross-fitted XGBoost regressions
#      estimate E(Z_hot | D = 0, X) and E(Z_cold | D = 0, X).
#
#   4. Generalized AIPW target estimator
#      The heat-side and cold-side log mortality rate-ratio coefficients are estimated jointly by
#      solving the cross-fitted generalized AIPW estimating equation.
#
# The script also calculates case-specific attributable fractions, annual attributable fractions,
# the equal-year mean annual attributable fraction, bootstrap confidence intervals, and the model
# diagnostics specified in the methodological appendix.
#
# REQUIRED INPUT
# --------------
# DATA_RDS must be an RDS file containing one row per sampled day. It must contain:
#
#   - ID_COL: matched-set identifier (default: "id")
#   - CASE_COL: case indicator coded 1 for the case day and 0 for control days (default: "case")
#   - YEAR_COL or DATE_COL: calendar information used to identify the case year
#   - temp_lag0, temp_lag1, ... through the largest requested temperature lag
#   - rh_lag0, rh_lag1, ... through the largest requested humidity lag
#   - HOLIDAY_COL, if available; if absent, the holiday indicator is set to zero
#   - any variables named in OUTCOME_EXTRA_VARS or EXPOSURE_EXTRA_VARS
#
# Each valid matched set must contain exactly one case day and at least one control day. Additional
# variables must be numerically encoded before analysis. Exposure-model predictors must not reveal
# which sampled day is the observed case day.
#
# TYPICAL USE
# -----------
# result <- case_crossover_aipw(
#   DATA_RDS    = "/path/to/COPD.rds",
#   MMT         = 23.1,
#   OUTCOME_NAME = "COPD",
#   OUT_DIR     = "/path/to/results",
#   BOOT_B      = 500L
# )
#
# Use a small BOOT_B only for code testing. Final inference should use a sufficiently large number
# of bootstrap replicates.
#
# MAIN OUTPUTS
# ------------
# *_RR_results.csv
#   Heat-side and cold-side RR estimates, bootstrap confidence intervals, and outcome-only
#   comparator estimates.
#
# *_AF_annual_mean_results.csv
#   Annual and equal-year mean annual attributable fractions, including total, standalone, and
#   Shapley-decomposed components, with percentile bootstrap confidence intervals.
#
# *_case_specific_AF.csv
#   Case-specific total, standalone, and Shapley-decomposed attributable fractions.
#
# *_diagnostic_outcome_folds.csv
# *_diagnostic_outcome_calibration.csv
# *_diagnostic_outcome_residual_scores.csv
#   Conditional logistic convergence, prediction, calibration, residual-score, and fold-stability
#   diagnostics.
#
# *_diagnostic_exposure_folds.csv
# *_diagnostic_exposure_calibration.csv
# *_diagnostic_exposure_residual_associations.csv
# *_diagnostic_exposure_quantiles.csv
# *_diagnostic_exposure_seasons.csv
#   XGBoost exposure conditional-mean diagnostics.
#
# *_diagnostic_AIPW_summary.csv
# *_diagnostic_AIPW_score_quantiles.csv
# *_diagnostic_AIPW_influential_sets.csv
# *_diagnostic_AIPW_leave_one_fold_out.csv
# *_diagnostic_AIPW_multiple_starts.csv
# *_diagnostic_estimator_comparison.csv
#   Overlap, inverse-probability, Jacobian, estimating-equation, influence, fold-deletion,
#   multiple-start, and estimator-comparison diagnostics.
#
# *_bootstrap_RR_trace.csv and *_bootstrap_AF_trace.csv
#   Replicate-specific results for reproducibility and later aggregation across mutually exclusive
#   outcomes.
#
# *_compact_results.rds and *_session_info.txt
#   Compact machine-readable results and the software environment.
#
# IMPORTANT IMPLEMENTATION NOTES
# ------------------------------
# - All eligible matched sets enter the joint heat/cold model. Matched sets are not selected by the
#   temperature observed on the case day.
# - No Gaussian exposure-density model, conditional variance model, density ratio, or exposure-shift
#   estimator is used.
# - Conditional case-day probabilities are not absolute mortality probabilities.
# - Analytic sandwich standard errors are diagnostic. Primary confidence intervals use the matched-
#   set bootstrap.
# - The bootstrap holds the original MMT, lag windows, fold allocation, outcome-basis specifications,
#   and fold-specific XGBoost hyperparameters fixed, while refitting all nuisance and target models.
# - This script does not implement the separate simulation study.
# - The implementation is memory-conscious: raw lag columns are discarded after exposure
#   construction, full conditional-logistic fit objects are not retained across folds, pseudo-pair
#   features are generated only when needed, and bootstrap replicates use reduced diagnostics.
####################################################################################################

options(stringsAsFactors = FALSE)

.required_packages <- c("data.table", "survival", "splines", "xgboost")
.missing_packages <- .required_packages[
  !vapply(.required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(.missing_packages) > 0L) {
  stop(
    sprintf(
      "Install the following R packages before running this script: %s",
      paste(.missing_packages, collapse = ", ")
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(survival)
  library(splines)
  library(xgboost)
})

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

case_crossover_aipw <- function(
    DATA_RDS,
    MMT,
    OUT_DIR = file.path(getwd(), "case_crossover_AIPW_results"),
    OUTCOME_NAME = NULL,
    BOOT_B = 100L,
    K_fold = 3L,
    ANALYSIS_YEARS = 2013:2019,
    HOT_LAGS = 0:3,
    COLD_LAGS = 0:14,
    RH_LAGS = 0:3,
    RH_SPLINE_DF = 3L,
    ID_COL = "id",
    CASE_COL = "case",
    YEAR_COL = "year",
    DATE_COL = NULL,
    MONTH_COL = NULL,
    HOLIDAY_COL = "holiday",
    OUTCOME_EXTRA_VARS = character(0),
    EXPOSURE_EXTRA_VARS = character(0),
    CLOGIT_METHOD = "efron",
    CLOGIT_MAX_ITER = 50L,
    CLOGIT_EPS = 1e-9,
    FIT_FULL_OUTCOME_COMPARATOR = FALSE,
    tune_frac = 0.10,
    tune_try_random = 10L,
    tune_inner_folds = 3L,
    xgb_nthread = 4L,
    xgb_nrounds_grid = seq(200L, 600L, by = 50L),
    xgb_early_stopping_rounds = 20L,
    XGB_PREDICT_CHUNK_SIZE = 250000L,
    AIPW_CHUNK_SIZE = 500000L,
    DIAGNOSTIC_GROUPS = 10L,
    NEWTON_MAX_ITER = 100L,
    NEWTON_TOL_SCORE = 1e-8,
    NEWTON_TOL_BETA = 1e-8,
    NEWTON_MAX_STEP = 1,
    CHECK_MULTIPLE_STARTS = TRUE,
    START_PERTURBATION = 0.05,
    MULTISTART_ROOT_TOL = 1e-5,
    TOP_INFLUENTIAL_SETS = 100L,
    BOOT_MIN_SUCCESS_FRAC = 0.70,
    SAVE_CASE_SPECIFIC = TRUE,
    SAVE_COMPACT_RDS = TRUE,
    SEED_MASTER = 20260712L
) {
  
  # ==============================================================================================
  # 0. Argument validation and general utilities
  # ==============================================================================================
  
  stopifnot(is.character(DATA_RDS), length(DATA_RDS) == 1L, nzchar(DATA_RDS))
  if (!file.exists(DATA_RDS)) stop(sprintf("Input file does not exist: %s", DATA_RDS))
  stopifnot(is.numeric(MMT), length(MMT) == 1L, is.finite(MMT))
  stopifnot(K_fold >= 2L)
  stopifnot(length(ANALYSIS_YEARS) >= 1L)
  stopifnot(length(HOT_LAGS) >= 1L, length(COLD_LAGS) >= 1L, length(RH_LAGS) >= 1L)
  stopifnot(RH_SPLINE_DF >= 1L)
  stopifnot(BOOT_B >= 0L)
  if (BOOT_B == 1L) stop("BOOT_B must be 0 or at least 2.")
  if (!CLOGIT_METHOD %in% c("exact", "efron", "breslow")) {
    stop("CLOGIT_METHOD must be one of: exact, efron, breslow.")
  }
  if (tune_frac <= 0 || tune_frac > 1) stop("tune_frac must be in (0, 1].")
  if (DIAGNOSTIC_GROUPS < 2L) stop("DIAGNOSTIC_GROUPS must be at least 2.")
  if (XGB_PREDICT_CHUNK_SIZE < 1000L || AIPW_CHUNK_SIZE < 1000L) {
    stop("Chunk sizes must be at least 1,000 rows.")
  }
  
  stable_string_seed <- function(x) {
    ints <- utf8ToInt(as.character(x)[1L])
    if (length(ints) == 0L) return(0L)
    as.integer(sum((seq_along(ints) * ints) %% 100000L) %% 100000L)
  }
  
  DATA_TAG <- tools::file_path_sans_ext(basename(DATA_RDS))
  OUTCOME_NAME <- as.character(OUTCOME_NAME %||% DATA_TAG)[1L]
  OUTCOME_SEED_OFFSET <- stable_string_seed(OUTCOME_NAME)
  ANALYSIS_SEED <- as.integer(SEED_MASTER + OUTCOME_SEED_OFFSET)
  
  set.seed(ANALYSIS_SEED)
  Sys.setenv(OMP_NUM_THREADS = as.character(xgb_nthread))
  
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  
  BOOT_SEED <- as.integer(ANALYSIS_SEED + 50000L)
  BOOT_MIN_OK <- if (BOOT_B > 0L) {
    min(BOOT_B, max(2L, ceiling(BOOT_MIN_SUCCESS_FRAC * BOOT_B)))
  } else {
    0L
  }
  
  tic <- function() proc.time()[[3L]]
  
  toc <- function(t0, label, quiet = FALSE) {
    if (!quiet) message(sprintf("[%.2fs] %s", proc.time()[[3L]] - t0, label))
    invisible(NULL)
  }
  
  safe_error_message <- function(e) {
    out <- tryCatch(conditionMessage(e), error = function(...) as.character(e))
    if (length(out) == 0L || is.na(out[1L]) || !nzchar(out[1L])) "unknown error" else out[1L]
  }
  
  clamp_machine_exp <- function(x) {
    if (any(!is.finite(x))) stop("The AIPW linear predictor contains non-finite values.")
    if (any(x > 700)) {
      stop(
        "The AIPW estimating equation would overflow because exp(eta) exceeds machine range; inspect overlap diagnostics."
      )
    }
    exp(x)
  }
  
  expit <- function(x) stats::plogis(x)
  
  numericize <- function(x, name) {
    if (is.logical(x)) return(as.numeric(x))
    if (is.numeric(x) || is.integer(x)) return(as.numeric(x))
    z <- suppressWarnings(as.numeric(as.character(x)))
    if (all(is.na(z) == is.na(x))) return(z)
    stop(sprintf("Variable '%s' must be numerically encoded before analysis.", name))
  }
  
  coerce_to_date <- function(x, name) {
    if (inherits(x, "Date")) return(x)
    if (inherits(x, c("POSIXct", "POSIXlt"))) return(as.Date(x))
    out <- suppressWarnings(as.Date(x))
    if (all(is.na(out))) stop(sprintf("Column '%s' cannot be converted to Date.", name))
    out
  }
  
  month_to_season <- function(month) {
    out <- rep(NA_character_, length(month))
    out[month %in% c(12L, 1L, 2L)] <- "Winter"
    out[month %in% c(3L, 4L, 5L)] <- "Spring"
    out[month %in% c(6L, 7L, 8L)] <- "Summer"
    out[month %in% c(9L, 10L, 11L)] <- "Autumn"
    out
  }
  
  rmse_from_sums <- function(sw, sse) {
    if (!is.finite(sw) || sw <= 0 || !is.finite(sse)) return(NA_real_)
    sqrt(sse / sw)
  }
  
  r2_from_sums <- function(sw, sy, sy2, sse) {
    if (!is.finite(sw) || sw <= 0) return(NA_real_)
    denom <- sy2 - sy^2 / sw
    if (!is.finite(denom) || denom <= 0) return(NA_real_)
    1 - sse / denom
  }
  
  weighted_regression_from_sums <- function(sw, sy, sp, sp2, syp) {
    if (!is.finite(sw) || sw <= 0) return(c(intercept = NA_real_, slope = NA_real_))
    var_p <- sp2 - sp^2 / sw
    cov_py <- syp - sp * sy / sw
    if (!is.finite(var_p) || var_p <= 0) return(c(intercept = NA_real_, slope = NA_real_))
    slope <- cov_py / var_p
    intercept <- sy / sw - slope * sp / sw
    c(intercept = intercept, slope = slope)
  }
  
  weighted_quantile <- function(x, probs, w = NULL) {
    ok <- is.finite(x)
    x <- x[ok]
    if (is.null(w)) {
      if (length(x) == 0L) return(rep(NA_real_, length(probs)))
      return(as.numeric(stats::quantile(x, probs = probs, names = FALSE, na.rm = TRUE)))
    }
    w <- w[ok]
    ok2 <- is.finite(w) & w >= 0
    x <- x[ok2]
    w <- w[ok2]
    if (length(x) == 0L || sum(w) <= 0) return(rep(NA_real_, length(probs)))
    ord <- order(x)
    x <- x[ord]
    w <- w[ord]
    cw <- cumsum(w) / sum(w)
    vapply(probs, function(p) x[which(cw >= p)[1L]], numeric(1L))
  }
  
  softmax_by_set <- function(eta, set_index) {
    tmp <- data.table(set_index = set_index, eta = as.numeric(eta))
    tmp[, p := {
      z <- eta - max(eta)
      ez <- exp(z)
      ez / sum(ez)
    }, by = set_index]
    tmp$p
  }
  
  add_grouped_values <- function(target, group, values) {
    if (length(group) == 0L) return(target)
    ag <- rowsum(values, group = group, reorder = FALSE)
    ids <- as.integer(rownames(ag))
    target[ids] <- target[ids] + ag[, 1L]
    target
  }
  
  matrix_condition_number <- function(M) {
    if (is.null(M) || length(M) == 0L || any(!is.finite(M))) return(NA_real_)
    sv <- tryCatch(base::svd(M, nu = 0L, nv = 0L)$d, error = function(e) NA_real_)
    if (length(sv) == 0L || any(!is.finite(sv)) || min(sv) <= 0) return(Inf)
    max(sv) / min(sv)
  }
  
  matrix_min_abs_eigen <- function(M) {
    if (is.null(M) || length(M) == 0L || any(!is.finite(M))) return(NA_real_)
    ev <- tryCatch(eigen(M, symmetric = FALSE, only.values = TRUE)$values, error = function(e) NA_complex_)
    ev <- ev[is.finite(ev)]
    if (length(ev) == 0L) return(NA_real_)
    min(abs(ev))
  }
  
  predict_xgb_two_models <- function(model_1, model_2, X, chunk_size = XGB_PREDICT_CHUNK_SIZE) {
    n <- nrow(X)
    if (n == 0L) return(list(first = numeric(0L), second = numeric(0L)))
    pred_1 <- numeric(n)
    pred_2 <- numeric(n)
    starts <- seq.int(1L, n, by = chunk_size)
    for (st in starts) {
      en <- min(n, st + chunk_size - 1L)
      dm <- xgboost::xgb.DMatrix(data = X[st:en, , drop = FALSE])
      pred_1[st:en] <- stats::predict(model_1, dm)
      pred_2[st:en] <- stats::predict(model_2, dm)
      rm(dm)
    }
    list(first = pred_1, second = pred_2)
  }
  
  make_group_folds <- function(group_id, K, seed) {
    uid <- unique(as.integer(group_id))
    if (length(uid) < 2L) stop("Too few matched sets for grouped cross-validation.")
    K <- min(max(2L, as.integer(K)), length(uid))
    set.seed(seed)
    data.table(
      group_id = uid,
      fold = sample(rep(seq_len(K), length.out = length(uid)))
    )
  }
  
  make_grouped_cv_indices <- function(group_id, K, seed) {
    fmap <- make_group_folds(group_id, K, seed)
    row_fold <- fmap[data.table(group_id = as.integer(group_id)), on = "group_id", fold]
    lapply(sort(unique(row_fold)), function(k) which(row_fold == k))
  }
  
  flatten_tunes <- function(tunes) {
    rows <- list()
    z <- 0L
    for (side in names(tunes)) {
      for (fold_name in names(tunes[[side]])) {
        tt <- tunes[[side]][[fold_name]]
        if (is.null(tt)) next
        base <- data.table(
          Model = side,
          Fold = as.integer(fold_name),
          CV_RMSE = as.numeric(tt$rmse %||% NA_real_),
          Nrounds = as.integer(tt$nrounds %||% NA_integer_)
        )
        pp <- tt$params %||% list()
        pp$seed <- NULL
        for (nm in names(pp)) base[, (nm) := as.character(pp[[nm]])]
        z <- z + 1L
        rows[[z]] <- base
      }
    }
    if (length(rows) == 0L) return(data.table())
    rbindlist(rows, fill = TRUE)
  }
  
  message("================================================================================================")
  message("Case-crossover generalized AIPW analysis for non-optimal temperature")
  message("================================================================================================")
  message(sprintf("Outcome:   %s", OUTCOME_NAME))
  message(sprintf("Input:     %s", DATA_RDS))
  message(sprintf("Output:    %s", OUT_DIR))
  message(sprintf("MMT:       %.4f", MMT))
  message(sprintf("Years:     %s", paste(ANALYSIS_YEARS, collapse = ",")))
  message(sprintf("Outer folds: %d | Bootstrap replicates: %d", K_fold, BOOT_B))
  
  # ==============================================================================================
  # 1. Read and prepare a compact sampled-day data set
  # ==============================================================================================
  
  prepare_data <- function(path) {
    t0 <- tic()
    raw <- readRDS(path)
    setDT(raw)
    raw <- copy(raw)
    
    required_base <- c(ID_COL, CASE_COL)
    missing_base <- setdiff(required_base, names(raw))
    if (length(missing_base) > 0L) {
      stop(sprintf("Missing required columns: %s", paste(missing_base, collapse = ", ")))
    }
    
    temp_hot_cols <- sprintf("temp_lag%d", HOT_LAGS)
    temp_cold_cols <- sprintf("temp_lag%d", COLD_LAGS)
    rh_cols <- sprintf("rh_lag%d", RH_LAGS)
    required_lag <- unique(c(temp_hot_cols, temp_cold_cols, rh_cols))
    missing_lag <- setdiff(required_lag, names(raw))
    if (length(missing_lag) > 0L) {
      stop(sprintf("Missing required lag columns: %s", paste(missing_lag, collapse = ", ")))
    }
    
    raw[, set_id_original := as.character(get(ID_COL))]
    raw[, case_internal := numericize(get(CASE_COL), CASE_COL)]
    if (!all(raw$case_internal %in% c(0, 1))) stop("The case indicator must contain only 0 and 1.")
    
    for (nm in required_lag) raw[, (nm) := numericize(get(nm), nm)]
    
    date_name <- DATE_COL
    if (!is.null(YEAR_COL) && YEAR_COL %in% names(raw)) {
      raw[, row_year_internal := as.integer(numericize(get(YEAR_COL), YEAR_COL))]
    } else {
      if (is.null(date_name)) {
        auto_dates <- c("date", "death_date", "case_date")
        date_name <- auto_dates[auto_dates %in% names(raw)][1L]
      }
      if (is.null(date_name) || is.na(date_name) || !date_name %in% names(raw)) {
        stop("No usable year column was found. Supply YEAR_COL or DATE_COL.")
      }
      dd <- coerce_to_date(raw[[date_name]], date_name)
      raw[, row_year_internal := as.integer(format(dd, "%Y"))]
    }
    
    if (!is.null(MONTH_COL) && MONTH_COL %in% names(raw)) {
      raw[, row_month_internal := as.integer(numericize(get(MONTH_COL), MONTH_COL))]
    } else if (!is.null(date_name) && !is.na(date_name) && date_name %in% names(raw)) {
      dd <- coerce_to_date(raw[[date_name]], date_name)
      raw[, row_month_internal := as.integer(format(dd, "%m"))]
    } else {
      raw[, row_month_internal := NA_integer_]
    }
    
    if (!is.null(HOLIDAY_COL) && HOLIDAY_COL %in% names(raw)) {
      raw[, holiday_internal := numericize(get(HOLIDAY_COL), HOLIDAY_COL)]
    } else {
      raw[, holiday_internal := 0]
    }
    
    all_extra <- unique(c(OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS))
    missing_extra <- setdiff(all_extra, names(raw))
    if (length(missing_extra) > 0L) {
      stop(sprintf("Missing additional covariates: %s", paste(missing_extra, collapse = ", ")))
    }
    for (nm in all_extra) raw[, (nm) := numericize(get(nm), nm)]
    
    raw[, T_hot_internal := rowMeans(.SD), .SDcols = temp_hot_cols]
    raw[, T_cold_internal := rowMeans(.SD), .SDcols = temp_cold_cols]
    raw[, rh_summary_internal := rowMeans(.SD), .SDcols = rh_cols]
    raw[, A_hot := pmax(T_hot_internal - MMT, 0)]
    raw[, A_cold := pmax(MMT - T_cold_internal, 0)]
    
    complete_vars <- unique(c(
      "case_internal", "row_year_internal", "A_hot", "A_cold", "rh_summary_internal",
      "holiday_internal", OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
    ))
    raw[, analysis_complete := stats::complete.cases(.SD), .SDcols = complete_vars]
    
    set_check <- raw[, .(
      n_rows = .N,
      n_cases = sum(case_internal == 1),
      all_complete = all(analysis_complete),
      case_year = if (sum(case_internal == 1) == 1L) {
        row_year_internal[case_internal == 1][1L]
      } else {
        NA_integer_
      }
    ), by = set_id_original]
    
    valid_sets <- set_check[
      n_cases == 1L & n_rows >= 2L & all_complete & case_year %in% ANALYSIS_YEARS,
      .(set_id_original, case_year)
    ]
    
    n_dropped <- nrow(set_check) - nrow(valid_sets)
    if (nrow(valid_sets) < max(10L, K_fold * 2L)) {
      stop(sprintf("Too few valid matched sets remain: %d.", nrow(valid_sets)))
    }
    
    set.seed(ANALYSIS_SEED + 1L)
    valid_sets[, fold := sample(rep(seq_len(min(K_fold, .N)), length.out = .N)), by = case_year]
    valid_sets[, set_index := .I]
    
    raw_compact_names <- unique(c(
      "set_id_original", "case_internal", "row_month_internal",
      "A_hot", "A_cold", "rh_summary_internal", "holiday_internal",
      OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
    ))
    raw_compact <- raw[, ..raw_compact_names]
    rm(raw)
    gc(FALSE)
    
    joined <- valid_sets[
      raw_compact,
      on = "set_id_original",
      nomatch = 0L,
      allow.cartesian = TRUE
    ]
    
    compact_names <- unique(c(
      "set_index", "case_internal", "case_year", "fold", "row_month_internal",
      "A_hot", "A_cold", "rh_summary_internal", "holiday_internal",
      OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
    ))
    rows <- joined[, ..compact_names]
    setorder(rows, set_index, -case_internal)
    rows[, row_uid := .I]
    
    set_map <- valid_sets[, .(set_index, set_id_original, case_year, fold)]
    setorder(set_map, set_index)
    
    rm(raw_compact, joined, set_check, valid_sets)
    gc(FALSE)
    
    message(sprintf(
      "Prepared compact data: %d matched sets and %d sampled-day rows; dropped %d invalid or out-of-period sets.",
      nrow(set_map), nrow(rows), n_dropped
    ))
    toc(t0, "Data preparation completed")
    
    list(rows = rows, set_map = set_map)
  }
  
  prepared <- prepare_data(DATA_RDS)
  data_main <- prepared$rows
  set_map_main <- prepared$set_map
  rm(prepared)
  gc(FALSE)
  
  # ==============================================================================================
  # 2. Cross-fitted conditional logistic outcome nuisance model
  # ==============================================================================================
  
  derive_outcome_basis_spec <- function(rows, train_idx) {
    x <- rows$rh_summary_internal[train_idx]
    x <- x[is.finite(x)]
    if (length(x) == 0L) stop("No finite relative-humidity values are available in an outcome training fold.")
    
    use_spline <- length(unique(x)) >= max(5L, RH_SPLINE_DF + 1L) && RH_SPLINE_DF >= 2L
    if (use_spline) {
      n_internal <- max(0L, RH_SPLINE_DF - 1L)
      boundary <- range(x)
      if (n_internal > 0L) {
        probs <- seq(0, 1, length.out = n_internal + 2L)[-c(1L, n_internal + 2L)]
        knots <- unique(as.numeric(stats::quantile(x, probs = probs, names = FALSE, type = 7L)))
        knots <- knots[knots > boundary[1L] & knots < boundary[2L]]
      } else {
        knots <- numeric(0L)
      }
      spec <- list(
        rh_type = "natural_spline",
        knots = knots,
        boundary_knots = boundary,
        requested_df = as.integer(RH_SPLINE_DF),
        active_columns = NULL
      )
    } else {
      spec <- list(
        rh_type = "linear",
        knots = numeric(0L),
        boundary_knots = range(x),
        requested_df = 1L,
        active_columns = NULL
      )
    }
    spec
  }
  
  build_outcome_basis <- function(rows, idx, spec, determine_active = FALSE) {
    x <- rows$rh_summary_internal[idx]
    
    if (identical(spec$rh_type, "natural_spline")) {
      rh_basis <- splines::ns(
        x,
        knots = spec$knots,
        Boundary.knots = spec$boundary_knots,
        intercept = FALSE
      )
      colnames(rh_basis) <- paste0("rh_ns", seq_len(ncol(rh_basis)))
    } else {
      rh_basis <- matrix(x, ncol = 1L)
      colnames(rh_basis) <- "rh_linear"
    }
    
    pieces <- list(rh_basis, matrix(rows$holiday_internal[idx], ncol = 1L))
    names_extra <- c(colnames(rh_basis), "holiday_internal")
    
    if (length(OUTCOME_EXTRA_VARS) > 0L) {
      extra_matrix <- do.call(
        cbind,
        lapply(OUTCOME_EXTRA_VARS, function(nm) rows[[nm]][idx])
      )
      if (is.null(dim(extra_matrix))) extra_matrix <- matrix(extra_matrix, ncol = 1L)
      colnames(extra_matrix) <- OUTCOME_EXTRA_VARS
      pieces[[length(pieces) + 1L]] <- extra_matrix
      names_extra <- c(names_extra, OUTCOME_EXTRA_VARS)
    }
    
    B <- do.call(cbind, pieces)
    colnames(B) <- names_extra
    storage.mode(B) <- "double"
    B[!is.finite(B)] <- 0
    
    if (determine_active || is.null(spec$active_columns)) {
      active <- vapply(seq_len(ncol(B)), function(j) {
        v <- B[, j]
        length(v) > 1L && is.finite(stats::sd(v)) && stats::sd(v) > 0
      }, logical(1L))
      spec$active_columns <- colnames(B)[active]
    }
    
    active_columns <- intersect(spec$active_columns %||% character(0L), colnames(B))
    if (length(active_columns) == 0L) {
      B <- matrix(numeric(0L), nrow = length(idx), ncol = 0L)
    } else {
      B <- B[, active_columns, drop = FALSE]
    }
    
    list(matrix = B, spec = spec)
  }
  
  compact_clogit_fit <- function(fit, nuisance_names, fold_id) {
    cf <- stats::coef(fit)
    if (!all(c("A_hot", "A_cold") %in% names(cf))) {
      stop(sprintf("Outcome fold %s did not return both target coefficients.", fold_id))
    }
    if (any(!is.finite(cf[c("A_hot", "A_cold")]))) {
      stop(sprintf("Outcome fold %s returned a non-finite target coefficient.", fold_id))
    }
    
    gamma <- setNames(rep(0, length(nuisance_names)), nuisance_names)
    common <- intersect(nuisance_names, names(cf))
    if (length(common) > 0L) {
      gamma[common] <- cf[common]
      gamma[!is.finite(gamma)] <- 0
    }
    
    vc <- tryCatch(stats::vcov(fit), error = function(e) NULL)
    coefficient_se <- setNames(rep(NA_real_, length(cf)), names(cf))
    if (!is.null(vc) && nrow(vc) == length(cf)) {
      coefficient_se <- sqrt(pmax(diag(vc), 0))
      names(coefficient_se) <- names(cf)
    }
    
    information_condition_number <- if (!is.null(vc) && all(is.finite(vc))) {
      matrix_condition_number(vc)
    } else {
      NA_real_
    }
    
    fail_text <- as.character(fit$fail %||% "")
    converged <- length(fail_text) == 0L || !nzchar(fail_text[1L])
    
    list(
      beta = c(A_hot = as.numeric(cf["A_hot"]), A_cold = as.numeric(cf["A_cold"])),
      gamma = gamma,
      coefficient_se = coefficient_se,
      loglik = if (!is.null(fit$loglik)) as.numeric(tail(fit$loglik, 1L)) else NA_real_,
      iterations = as.integer((fit$iter %||% NA_integer_)[1L]),
      information_condition_number = information_condition_number,
      maximum_absolute_coefficient = max(abs(cf), na.rm = TRUE),
      maximum_coefficient_se = if (all(is.na(coefficient_se))) NA_real_ else max(coefficient_se, na.rm = TRUE),
      non_estimable_nuisance_coefficients = sum(!is.finite(cf[intersect(nuisance_names, names(cf))])),
      converged = converged,
      coefficient_table = data.table(
        Term = names(cf),
        Estimate = as.numeric(cf),
        Standard_error = as.numeric(coefficient_se[names(cf)])
      )
    )
  }
  
  fit_one_clogit_fold <- function(
    rows,
    train_idx,
    test_idx,
    fold_id,
    basis_spec = NULL,
    collect_diagnostics = TRUE
  ) {
    if (is.null(basis_spec)) basis_spec <- derive_outcome_basis_spec(rows, train_idx)
    
    basis_train <- build_outcome_basis(
      rows = rows,
      idx = train_idx,
      spec = basis_spec,
      determine_active = is.null(basis_spec$active_columns)
    )
    Btr <- basis_train$matrix
    basis_spec <- basis_train$spec
    nuisance_names <- colnames(Btr) %||% character(0L)
    
    fit_df <- data.frame(
      case_internal = rows$case_internal[train_idx],
      set_index = rows$set_index[train_idx],
      A_hot = rows$A_hot[train_idx],
      A_cold = rows$A_cold[train_idx],
      check.names = FALSE
    )
    if (ncol(Btr) > 0L) {
      for (j in seq_len(ncol(Btr))) fit_df[[colnames(Btr)[j]]] <- Btr[, j]
    }
    rm(Btr, basis_train)
    gc(FALSE)
    
    rhs <- c("A_hot", "A_cold", nuisance_names, "strata(set_index)")
    fml <- stats::as.formula(paste("case_internal ~", paste(rhs, collapse = " + ")))
    
    fit <- survival::clogit(
      formula = fml,
      data = fit_df,
      method = CLOGIT_METHOD,
      model = FALSE,
      x = FALSE,
      y = FALSE,
      control = survival::coxph.control(iter.max = CLOGIT_MAX_ITER, eps = CLOGIT_EPS)
    )
    
    compact <- compact_clogit_fit(fit, nuisance_names, fold_id)
    
    rm(fit, fit_df)
    gc(FALSE)
    
    basis_test <- build_outcome_basis(
      rows = rows,
      idx = test_idx,
      spec = basis_spec,
      determine_active = FALSE
    )
    Bte <- basis_test$matrix
    
    nuisance_lp_test <- if (ncol(Bte) > 0L) {
      drop(Bte %*% compact$gamma[colnames(Bte)])
    } else {
      rep(0, length(test_idx))
    }
    if (any(!is.finite(nuisance_lp_test))) {
      stop(sprintf("Outcome fold %s produced non-finite nuisance predictions.", fold_id))
    }
    
    fold_diag <- data.table(
      Fold = as.integer(fold_id),
      N_train_sets = uniqueN(rows$set_index[train_idx]),
      N_test_sets = uniqueN(rows$set_index[test_idx]),
      N_train_rows = length(train_idx),
      N_test_rows = length(test_idx),
      beta_hot_outcome = compact$beta[1L],
      beta_cold_outcome = compact$beta[2L],
      SE_beta_hot_outcome = as.numeric(compact$coefficient_se["A_hot"]),
      SE_beta_cold_outcome = as.numeric(compact$coefficient_se["A_cold"]),
      RR_hot_outcome = exp(compact$beta[1L]),
      RR_cold_outcome = exp(compact$beta[2L]),
      Conditional_loglik_train = compact$loglik,
      Model_iterations = compact$iterations,
      Information_condition_number = compact$information_condition_number,
      Maximum_absolute_coefficient = compact$maximum_absolute_coefficient,
      Maximum_coefficient_SE = compact$maximum_coefficient_se,
      Non_estimable_nuisance_coefficients = compact$non_estimable_nuisance_coefficients,
      Converged = compact$converged
    )
    
    calibration <- data.table()
    residual_scores <- data.table()
    
    if (collect_diagnostics) {
      eta_test <- compact$beta[1L] * rows$A_hot[test_idx] +
        compact$beta[2L] * rows$A_cold[test_idx] + nuisance_lp_test
      set_test <- rows$set_index[test_idx]
      y_test <- rows$case_internal[test_idx]
      p_test <- softmax_by_set(eta_test, set_test)
      
      if (any(!is.finite(p_test)) || any(p_test < 0) || any(p_test > 1)) {
        stop(sprintf("Outcome fold %s produced invalid conditional probabilities.", fold_id))
      }
      
      set_lengths <- rle(set_test)$lengths
      probability_sums <- rowsum(p_test, set_test, reorder = FALSE)[, 1L]
      maximum_sum_error <- max(abs(probability_sums - 1), na.rm = TRUE)
      p_case <- p_test[y_test == 1]
      if (length(p_case) != length(set_lengths)) {
        stop(sprintf("Outcome fold %s has an invalid number of observed case probabilities.", fold_id))
      }
      
      probability_quantiles <- as.numeric(stats::quantile(
        p_test,
        probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
        names = FALSE,
        na.rm = TRUE
      ))
      case_probability_quantiles <- as.numeric(stats::quantile(
        p_case,
        probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
        names = FALSE,
        na.rm = TRUE
      ))
      
      heldout_nll <- -mean(log(pmax(p_case, .Machine$double.xmin)))
      squared_error <- (y_test - p_test)^2
      set_brier <- rowsum(squared_error, set_test, reorder = FALSE)[, 1L] / set_lengths
      heldout_brier <- mean(set_brier)
      
      row_set_weight <- rep(1 / set_lengths, times = set_lengths)
      cuts <- unique(as.numeric(stats::quantile(
        p_test,
        probs = seq(0, 1, length.out = DIAGNOSTIC_GROUPS + 1L),
        names = FALSE,
        na.rm = TRUE
      )))
      if (length(cuts) >= 3L) {
        probability_group <- findInterval(p_test, cuts, all.inside = TRUE)
      } else {
        probability_group <- rep(1L, length(p_test))
      }
      cal_work <- data.table(
        group = probability_group,
        observed = y_test,
        predicted = p_test,
        weight = row_set_weight
      )
      calibration <- cal_work[, .(
        N_rows = .N,
        Total_set_weight = sum(weight),
        Mean_predicted_probability = sum(weight * predicted) / sum(weight),
        Observed_case_proportion = sum(weight * observed) / sum(weight)
      ), by = group]
      calibration[, `:=`(
        Fold = as.integer(fold_id),
        Absolute_calibration_error = abs(Observed_case_proportion - Mean_predicted_probability)
      )]
      setcolorder(calibration, c(
        "Fold", "group", "N_rows", "Total_set_weight", "Mean_predicted_probability",
        "Observed_case_proportion", "Absolute_calibration_error"
      ))
      
      residual <- y_test - p_test
      if (ncol(Bte) > 0L) {
        score_values <- drop(crossprod(Bte, residual)) / length(set_lengths)
        residual_scores <- data.table(
          Fold = as.integer(fold_id),
          Basis_term = colnames(Bte),
          Residual_score = as.numeric(score_values),
          Absolute_residual_score = abs(as.numeric(score_values))
        )
        residual_mean_abs <- mean(abs(score_values))
        residual_max_abs <- max(abs(score_values))
      } else {
        residual_mean_abs <- NA_real_
        residual_max_abs <- NA_real_
      }
      
      fold_diag[, `:=`(
        Heldout_negative_loglik_per_set = heldout_nll,
        Heldout_set_weighted_Brier = heldout_brier,
        Heldout_calibration_mean_abs_error = if (nrow(calibration)) mean(calibration$Absolute_calibration_error) else NA_real_,
        Heldout_calibration_max_abs_error = if (nrow(calibration)) max(calibration$Absolute_calibration_error) else NA_real_,
        Conditional_probability_min = probability_quantiles[1L],
        Conditional_probability_p01 = probability_quantiles[2L],
        Conditional_probability_p05 = probability_quantiles[3L],
        Conditional_probability_median = probability_quantiles[4L],
        Conditional_probability_p95 = probability_quantiles[5L],
        Conditional_probability_p99 = probability_quantiles[6L],
        Conditional_probability_max = probability_quantiles[7L],
        Case_probability_min = case_probability_quantiles[1L],
        Case_probability_p01 = case_probability_quantiles[2L],
        Case_probability_p05 = case_probability_quantiles[3L],
        Case_probability_median = case_probability_quantiles[4L],
        Case_probability_p95 = case_probability_quantiles[5L],
        Case_probability_p99 = case_probability_quantiles[6L],
        Case_probability_max = case_probability_quantiles[7L],
        Proportion_case_probability_lt_0_01 = mean(p_case < 0.01),
        Proportion_case_probability_lt_0_05 = mean(p_case < 0.05),
        Max_set_probability_sum_error = maximum_sum_error,
        Heldout_residual_score_mean_abs = residual_mean_abs,
        Heldout_residual_score_max_abs = residual_max_abs
      )]
      
      rm(eta_test, set_test, y_test, p_test, set_lengths, probability_sums, p_case,
         squared_error, set_brier, row_set_weight, cuts, probability_group, cal_work, residual)
      gc(FALSE)
    }
    
    rm(Bte, basis_test)
    gc(FALSE)
    
    list(
      nuisance_lp_test = nuisance_lp_test,
      beta = compact$beta,
      gamma = compact$gamma,
      basis_spec = basis_spec,
      diagnostics = fold_diag,
      coefficient_table = compact$coefficient_table[, Fold := as.integer(fold_id)][],
      calibration = calibration,
      residual_scores = residual_scores
    )
  }
  
  fit_full_outcome_comparator <- function(rows) {
    idx <- seq_len(nrow(rows))
    spec <- derive_outcome_basis_spec(rows, idx)
    basis <- build_outcome_basis(rows, idx, spec, determine_active = TRUE)
    B <- basis$matrix
    nuisance_names <- colnames(B) %||% character(0L)
    
    fit_df <- data.frame(
      case_internal = rows$case_internal,
      set_index = rows$set_index,
      A_hot = rows$A_hot,
      A_cold = rows$A_cold,
      check.names = FALSE
    )
    if (ncol(B) > 0L) {
      for (j in seq_len(ncol(B))) fit_df[[colnames(B)[j]]] <- B[, j]
    }
    rm(B, basis)
    gc(FALSE)
    rhs <- c("A_hot", "A_cold", nuisance_names, "strata(set_index)")
    fml <- stats::as.formula(paste("case_internal ~", paste(rhs, collapse = " + ")))
    fit <- survival::clogit(
      formula = fml,
      data = fit_df,
      method = CLOGIT_METHOD,
      model = FALSE,
      x = FALSE,
      y = FALSE,
      control = survival::coxph.control(iter.max = CLOGIT_MAX_ITER, eps = CLOGIT_EPS)
    )
    compact <- compact_clogit_fit(fit, nuisance_names, "full")
    rm(fit, fit_df)
    gc(FALSE)
    compact$beta
  }
  
  fit_outcome_crossfit <- function(
    rows,
    fixed_basis_specs = NULL,
    collect_diagnostics = TRUE,
    fit_full_comparator = FALSE,
    quiet = FALSE
  ) {
    folds <- sort(unique(rows$fold))
    nuisance_oof <- rep(NA_real_, nrow(rows))
    fold_diagnostics <- list()
    coefficient_rows <- list()
    calibration_rows <- list()
    residual_rows <- list()
    fold_betas <- list()
    basis_specs <- fixed_basis_specs %||% list()
    
    for (ii in seq_along(folds)) {
      k <- folds[ii]
      train_idx <- which(rows$fold != k)
      test_idx <- which(rows$fold == k)
      if (length(train_idx) == 0L || length(test_idx) == 0L) {
        stop(sprintf("Outcome outer fold %d is empty.", k))
      }
      
      spec_k <- basis_specs[[as.character(k)]] %||% NULL
      one <- fit_one_clogit_fold(
        rows = rows,
        train_idx = train_idx,
        test_idx = test_idx,
        fold_id = k,
        basis_spec = spec_k,
        collect_diagnostics = collect_diagnostics
      )
      
      nuisance_oof[test_idx] <- one$nuisance_lp_test
      basis_specs[[as.character(k)]] <- one$basis_spec
      fold_diagnostics[[ii]] <- one$diagnostics
      coefficient_rows[[ii]] <- one$coefficient_table
      calibration_rows[[ii]] <- one$calibration
      residual_rows[[ii]] <- one$residual_scores
      fold_betas[[ii]] <- data.table(
        Fold = k,
        N_test_sets = uniqueN(rows$set_index[test_idx]),
        beta_hot = one$beta[1L],
        beta_cold = one$beta[2L]
      )
      
      if (!quiet) {
        if (collect_diagnostics) {
          message(sprintf(
            "Outcome fold %d/%d: beta_H=%.6f, beta_C=%.6f, held-out NLL=%.4f, Brier=%.6f",
            ii, length(folds), one$beta[1L], one$beta[2L],
            one$diagnostics$Heldout_negative_loglik_per_set,
            one$diagnostics$Heldout_set_weighted_Brier
          ))
        } else {
          message(sprintf(
            "Outcome fold %d/%d: beta_H=%.6f, beta_C=%.6f",
            ii, length(folds), one$beta[1L], one$beta[2L]
          ))
        }
      }
      
      rm(one, train_idx, test_idx)
      gc(FALSE)
    }
    
    if (any(!is.finite(nuisance_oof))) {
      stop("At least one outcome nuisance out-of-fold prediction is missing or non-finite.")
    }
    
    beta_table <- rbindlist(fold_betas)
    beta_start <- c(
      A_hot = stats::weighted.mean(beta_table$beta_hot, beta_table$N_test_sets),
      A_cold = stats::weighted.mean(beta_table$beta_cold, beta_table$N_test_sets)
    )
    
    beta_stability_summary <- rbindlist(list(
      data.table(
        Parameter = "Heat",
        Mean = mean(beta_table$beta_hot),
        Standard_deviation = stats::sd(beta_table$beta_hot),
        Minimum = min(beta_table$beta_hot),
        Maximum = max(beta_table$beta_hot)
      ),
      data.table(
        Parameter = "Cold",
        Mean = mean(beta_table$beta_cold),
        Standard_deviation = stats::sd(beta_table$beta_cold),
        Minimum = min(beta_table$beta_cold),
        Maximum = max(beta_table$beta_cold)
      )
    ))
    
    comparator_beta <- beta_start
    comparator_label <- "Cross-fitted fold-average conditional logistic regression"
    if (fit_full_comparator) {
      comparator_beta <- fit_full_outcome_comparator(rows)
      comparator_label <- "Full-data conditional logistic regression"
    }
    
    list(
      nuisance_oof = nuisance_oof,
      basis_specs = basis_specs,
      diagnostics = rbindlist(fold_diagnostics, fill = TRUE),
      coefficients = rbindlist(coefficient_rows, fill = TRUE),
      calibration = rbindlist(calibration_rows, fill = TRUE),
      residual_scores = rbindlist(residual_rows, fill = TRUE),
      fold_betas = beta_table,
      fold_stability_summary = beta_stability_summary,
      beta_start = beta_start,
      comparator_beta = comparator_beta,
      comparator_label = comparator_label
    )
  }
  
  # ==============================================================================================
  # 3. Ordered case-control pseudo-pairs and on-demand pair-context features
  # ==============================================================================================
  
  build_pair_index <- function(rows, nuisance_oof) {
    n_sets <- max(rows$set_index)
    case_rows <- which(rows$case_internal == 1)
    control_rows <- which(rows$case_internal == 0)
    
    case_row_by_set <- integer(n_sets)
    case_row_by_set[rows$set_index[case_rows]] <- case_rows
    control_set <- rows$set_index[control_rows]
    paired_case_rows <- case_row_by_set[control_set]
    if (any(paired_case_rows <= 0L)) stop("At least one control row has no corresponding case row.")
    
    controls_per_set <- tabulate(control_set, nbins = n_sets)
    if (any(controls_per_set <= 0L)) stop("At least one matched set has no control day.")
    
    pairs <- data.table(
      set_index = as.integer(control_set),
      fold = as.integer(rows$fold[control_rows]),
      control_row = as.integer(control_rows),
      case_row = as.integer(paired_case_rows),
      M_s = as.integer(controls_per_set[control_set]),
      Z_hot_0 = rows$A_hot[control_rows] - rows$A_hot[paired_case_rows],
      Z_cold_0 = rows$A_cold[control_rows] - rows$A_cold[paired_case_rows],
      g_0 = nuisance_oof[control_rows] - nuisance_oof[paired_case_rows]
    )
    
    if (any(!is.finite(pairs$Z_hot_0)) || any(!is.finite(pairs$Z_cold_0)) || any(!is.finite(pairs$g_0))) {
      stop("The ordered pseudo-pair contrasts contain non-finite values.")
    }
    
    context_names <- unique(c("rh_summary_internal", "holiday_internal", EXPOSURE_EXTRA_VARS))
    feature_names <- unlist(lapply(context_names, function(nm) {
      c(
        paste0(nm, "_first"),
        paste0(nm, "_second"),
        paste0(nm, "_difference"),
        paste0(nm, "_mean")
      )
    }), use.names = FALSE)
    feature_names <- c(feature_names, "n_controls")
    
    list(
      pairs = pairs,
      n_sets = n_sets,
      context_names = context_names,
      feature_names = feature_names
    )
  }
  
  build_pair_feature_matrix <- function(rows, pair_index, idx, orientation = 0L) {
    pairs <- pair_index$pairs
    context_names <- pair_index$context_names
    n <- length(idx)
    p <- length(context_names)
    X <- matrix(0, nrow = n, ncol = 4L * p + 1L)
    colnames(X) <- pair_index$feature_names
    
    if (orientation == 0L) {
      first_rows <- pairs$control_row[idx]
      second_rows <- pairs$case_row[idx]
    } else if (orientation == 1L) {
      first_rows <- pairs$case_row[idx]
      second_rows <- pairs$control_row[idx]
    } else {
      stop("orientation must be 0 or 1.")
    }
    
    col_pos <- 0L
    for (nm in context_names) {
      first <- rows[[nm]][first_rows]
      second <- rows[[nm]][second_rows]
      X[, col_pos + 1L] <- first
      X[, col_pos + 2L] <- second
      X[, col_pos + 3L] <- first - second
      X[, col_pos + 4L] <- (first + second) / 2
      col_pos <- col_pos + 4L
    }
    X[, ncol(X)] <- pairs$M_s[idx]
    X[!is.finite(X)] <- 0
    storage.mode(X) <- "double"
    X
  }
  
  # ==============================================================================================
  # 4. Cross-fitted XGBoost exposure conditional-mean models
  # ==============================================================================================
  
  tune_xgb_regression <- function(
    rows,
    pair_index,
    outcome_vector,
    outer_train_idx,
    seed,
    tag
  ) {
    pairs <- pair_index$pairs
    training_sets <- unique(pairs$set_index[outer_train_idx])
    n_tune_sets <- max(tune_inner_folds * 2L, ceiling(tune_frac * length(training_sets)))
    n_tune_sets <- min(length(training_sets), max(2L, n_tune_sets))
    
    set.seed(seed)
    selected_sets <- sample(training_sets, n_tune_sets, replace = FALSE)
    tune_idx <- outer_train_idx[pairs$set_index[outer_train_idx] %in% selected_sets]
    
    if (length(tune_idx) < 20L || length(unique(pairs$set_index[tune_idx])) < 2L) {
      warning(sprintf("%s has insufficient tuning data; prespecified XGBoost parameters are used.", tag))
      return(list(
        params = list(
          max_depth = 4L,
          eta = 0.05,
          subsample = 0.85,
          colsample_bytree = 0.85,
          min_child_weight = 5,
          tree_method = "hist",
          objective = "reg:squarederror",
          eval_metric = "rmse",
          nthread = xgb_nthread,
          verbosity = 0
        ),
        nrounds = 300L,
        rmse = NA_real_
      ))
    }
    
    X_tune <- build_pair_feature_matrix(rows, pair_index, tune_idx, orientation = 0L)
    y_tune <- outcome_vector[tune_idx]
    w_tune <- 1 / pairs$M_s[tune_idx]
    groups_tune <- pairs$set_index[tune_idx]
    
    dtrain <- xgboost::xgb.DMatrix(data = X_tune, label = y_tune, weight = w_tune)
    cv_folds <- make_grouped_cv_indices(groups_tune, tune_inner_folds, seed + 1000L)
    
    best <- NULL
    best_rmse <- Inf
    set.seed(seed)
    
    for (i in seq_len(tune_try_random)) {
      params <- list(
        max_depth = sample(3:8, 1L),
        eta = stats::runif(1L, 0.02, 0.20),
        subsample = stats::runif(1L, 0.70, 1.00),
        colsample_bytree = stats::runif(1L, 0.70, 1.00),
        min_child_weight = sample(1:10, 1L),
        tree_method = "hist",
        objective = "reg:squarederror",
        eval_metric = "rmse",
        nthread = xgb_nthread,
        verbosity = 0
      )
      nround_cap <- sample(xgb_nrounds_grid, 1L)
      
      set.seed(seed + i)
      cv <- xgboost::xgb.cv(
        params = params,
        data = dtrain,
        nrounds = nround_cap,
        folds = cv_folds,
        early_stopping_rounds = xgb_early_stopping_rounds,
        verbose = 0,
        maximize = FALSE,
        showsd = TRUE
      )
      
      best_iteration <- cv$best_iteration %||% which.min(cv$evaluation_log$test_rmse_mean)
      this_rmse <- cv$evaluation_log$test_rmse_mean[best_iteration]
      if (is.finite(this_rmse) && this_rmse < best_rmse) {
        best_rmse <- this_rmse
        best <- list(
          params = params,
          nrounds = as.integer(best_iteration),
          rmse = as.numeric(this_rmse)
        )
      }
      rm(cv)
      gc(FALSE)
    }
    
    rm(dtrain, X_tune, y_tune, w_tune, groups_tune)
    gc(FALSE)
    
    if (is.null(best)) stop(sprintf("%s did not produce a valid XGBoost tuning result.", tag))
    best$params$seed <- NULL
    best
  }
  
  train_one_xgb_model <- function(X, y, w, tune, seed) {
    params <- tune$params
    params$seed <- NULL
    dm <- xgboost::xgb.DMatrix(data = X, label = y, weight = w)
    set.seed(seed)
    model <- xgboost::xgb.train(
      params = params,
      data = dm,
      nrounds = tune$nrounds,
      verbose = 0
    )
    train_prediction <- stats::predict(model, dm)
    rm(dm)
    list(model = model, prediction = train_prediction)
  }
  
  initialize_metric_sums <- function() {
    c(
      sw = 0, sy = 0, sy2 = 0, sp = 0, sp2 = 0, syp = 0,
      sse = 0, sae = 0, sr = 0, sr2 = 0
    )
  }
  
  update_metric_sums <- function(sums, y, pred, w) {
    residual <- y - pred
    sums["sw"] <- sums["sw"] + sum(w)
    sums["sy"] <- sums["sy"] + sum(w * y)
    sums["sy2"] <- sums["sy2"] + sum(w * y^2)
    sums["sp"] <- sums["sp"] + sum(w * pred)
    sums["sp2"] <- sums["sp2"] + sum(w * pred^2)
    sums["syp"] <- sums["syp"] + sum(w * y * pred)
    sums["sse"] <- sums["sse"] + sum(w * residual^2)
    sums["sae"] <- sums["sae"] + sum(w * abs(residual))
    sums["sr"] <- sums["sr"] + sum(w * residual)
    sums["sr2"] <- sums["sr2"] + sum(w * residual^2)
    sums
  }
  
  metric_table_from_sums <- function(train_sums, test_sums, tune, fold, component, n_train, n_test) {
    cal <- weighted_regression_from_sums(
      test_sums["sw"], test_sums["sy"], test_sums["sp"], test_sums["sp2"], test_sums["syp"]
    )
    rmse_train <- rmse_from_sums(train_sums["sw"], train_sums["sse"])
    rmse_test <- rmse_from_sums(test_sums["sw"], test_sums["sse"])
    r2_train <- r2_from_sums(train_sums["sw"], train_sums["sy"], train_sums["sy2"], train_sums["sse"])
    r2_test <- r2_from_sums(test_sums["sw"], test_sums["sy"], test_sums["sy2"], test_sums["sse"])
    
    data.table(
      Fold = as.integer(fold),
      Component = component,
      N_train_pairs = as.integer(n_train),
      N_test_pairs = as.integer(n_test),
      RMSE_train = rmse_train,
      RMSE_test = rmse_test,
      MAE_train = train_sums["sae"] / train_sums["sw"],
      MAE_test = test_sums["sae"] / test_sums["sw"],
      R2_train = r2_train,
      R2_test = r2_test,
      Delta_RMSE_percent = 100 * (rmse_test - rmse_train) / max(abs(rmse_train), 1e-12),
      Delta_R2_percent = 100 * (r2_test - r2_train) / max(abs(r2_train), 1e-12),
      Residual_mean_test = test_sums["sr"] / test_sums["sw"],
      Calibration_intercept_test = cal["intercept"],
      Calibration_slope_test = cal["slope"],
      CV_RMSE = as.numeric(tune$rmse %||% NA_real_),
      Nrounds = as.integer(tune$nrounds)
    )
  }
  
  fit_exposure_crossfit <- function(
    rows,
    pair_index,
    fixed_tunes = NULL,
    collect_diagnostics = TRUE,
    run_seed,
    quiet = FALSE
  ) {
    pairs <- pair_index$pairs
    n_pairs <- nrow(pairs)
    n_sets <- pair_index$n_sets
    folds <- sort(unique(pairs$fold))
    
    rH1 <- rep(NA_real_, n_pairs)
    rC1 <- rep(NA_real_, n_pairs)
    base_H0_set <- numeric(n_sets)
    base_C0_set <- numeric(n_sets)
    
    tunes <- list(heat = list(), cold = list())
    fold_diagnostics <- list()
    calibration_rows <- list()
    residual_association_rows <- list()
    quantile_rows <- list()
    season_rows <- list()
    diag_counter <- 0L
    
    residual_moments <- c(sw = 0, sH = 0, sC = 0, sHH = 0, sCC = 0, sHC = 0)
    
    for (fold_position in seq_along(folds)) {
      k <- folds[fold_position]
      train_idx <- which(pairs$fold != k)
      test_idx <- which(pairs$fold == k)
      if (length(train_idx) < 20L || length(test_idx) < 1L) {
        stop(sprintf("Exposure outer fold %d has insufficient training or held-out pairs.", k))
      }
      
      tune_H <- if (!is.null(fixed_tunes)) {
        fixed_tunes$heat[[as.character(k)]]
      } else {
        tune_xgb_regression(
          rows = rows,
          pair_index = pair_index,
          outcome_vector = pairs$Z_hot_0,
          outer_train_idx = train_idx,
          seed = run_seed + 10000L + k,
          tag = sprintf("Heat exposure model, fold %d", k)
        )
      }
      tune_C <- if (!is.null(fixed_tunes)) {
        fixed_tunes$cold[[as.character(k)]]
      } else {
        tune_xgb_regression(
          rows = rows,
          pair_index = pair_index,
          outcome_vector = pairs$Z_cold_0,
          outer_train_idx = train_idx,
          seed = run_seed + 20000L + k,
          tag = sprintf("Cold exposure model, fold %d", k)
        )
      }
      if (is.null(tune_H) || is.null(tune_C)) {
        stop(sprintf("Exposure outer fold %d is missing fixed XGBoost hyperparameters.", k))
      }
      tune_H$params$seed <- NULL
      tune_C$params$seed <- NULL
      tunes$heat[[as.character(k)]] <- tune_H
      tunes$cold[[as.character(k)]] <- tune_C
      
      X_train <- build_pair_feature_matrix(rows, pair_index, train_idx, orientation = 0L)
      w_train <- 1 / pairs$M_s[train_idx]
      
      heat_train <- train_one_xgb_model(
        X = X_train,
        y = pairs$Z_hot_0[train_idx],
        w = w_train,
        tune = tune_H,
        seed = run_seed + 30000L + k
      )
      heat_train_sums <- update_metric_sums(
        initialize_metric_sums(),
        pairs$Z_hot_0[train_idx],
        heat_train$prediction,
        w_train
      )
      model_H <- heat_train$model
      rm(heat_train)
      gc(FALSE)
      
      cold_train <- train_one_xgb_model(
        X = X_train,
        y = pairs$Z_cold_0[train_idx],
        w = w_train,
        tune = tune_C,
        seed = run_seed + 40000L + k
      )
      cold_train_sums <- update_metric_sums(
        initialize_metric_sums(),
        pairs$Z_cold_0[train_idx],
        cold_train$prediction,
        w_train
      )
      model_C <- cold_train$model
      rm(cold_train, X_train, w_train)
      gc(FALSE)
      
      heat_test_sums <- initialize_metric_sums()
      cold_test_sums <- initialize_metric_sums()
      
      feature_names <- pair_index$feature_names
      heat_assoc <- data.table(
        Feature = feature_names,
        sw = 0, sr = 0, sr2 = 0, sx = 0, sx2 = 0, srx = 0
      )
      cold_assoc <- copy(heat_assoc)
      
      yH_test_all <- pairs$Z_hot_0[test_idx]
      yC_test_all <- pairs$Z_cold_0[test_idx]
      w_test_all <- 1 / pairs$M_s[test_idx]
      heat_breaks <- unique(weighted_quantile(
        yH_test_all,
        probs = seq(0, 1, length.out = DIAGNOSTIC_GROUPS + 1L),
        w = w_test_all
      ))
      cold_breaks <- unique(weighted_quantile(
        yC_test_all,
        probs = seq(0, 1, length.out = DIAGNOSTIC_GROUPS + 1L),
        w = w_test_all
      ))
      
      heat_quantile_acc <- data.table(group = integer(0L), sw = numeric(0L), sr = numeric(0L), sr2 = numeric(0L))
      cold_quantile_acc <- copy(heat_quantile_acc)
      heat_season_acc <- data.table(season = character(0L), sw = numeric(0L), sr = numeric(0L), sr2 = numeric(0L))
      cold_season_acc <- copy(heat_season_acc)
      
      starts <- seq.int(1L, length(test_idx), by = XGB_PREDICT_CHUNK_SIZE)
      for (st in starts) {
        en <- min(length(test_idx), st + XGB_PREDICT_CHUNK_SIZE - 1L)
        local_pos <- st:en
        idx <- test_idx[local_pos]
        w <- 1 / pairs$M_s[idx]
        pair_weight <- w / 2
        
        X0 <- build_pair_feature_matrix(rows, pair_index, idx, orientation = 0L)
        pred0 <- predict_xgb_two_models(model_H, model_C, X0)
        predH0 <- pred0$first
        predC0 <- pred0$second
        rm(pred0)
        
        yH <- pairs$Z_hot_0[idx]
        yC <- pairs$Z_cold_0[idx]
        residH <- yH - predH0
        residC <- yC - predC0
        
        heat_test_sums <- update_metric_sums(heat_test_sums, yH, predH0, w)
        cold_test_sums <- update_metric_sums(cold_test_sums, yC, predC0, w)
        
        base_H0_set <- add_grouped_values(
          base_H0_set,
          pairs$set_index[idx],
          pair_weight * (-(yH - predH0))
        )
        base_C0_set <- add_grouped_values(
          base_C0_set,
          pairs$set_index[idx],
          pair_weight * (-(yC - predC0))
        )
        
        residual_moments["sw"] <- residual_moments["sw"] + sum(w)
        residual_moments["sH"] <- residual_moments["sH"] + sum(w * residH)
        residual_moments["sC"] <- residual_moments["sC"] + sum(w * residC)
        residual_moments["sHH"] <- residual_moments["sHH"] + sum(w * residH^2)
        residual_moments["sCC"] <- residual_moments["sCC"] + sum(w * residC^2)
        residual_moments["sHC"] <- residual_moments["sHC"] + sum(w * residH * residC)
        
        if (collect_diagnostics) {
          for (j in seq_along(feature_names)) {
            x <- X0[, j]
            heat_assoc[j, `:=`(
              sw = sw + sum(w),
              sr = sr + sum(w * residH),
              sr2 = sr2 + sum(w * residH^2),
              sx = sx + sum(w * x),
              sx2 = sx2 + sum(w * x^2),
              srx = srx + sum(w * residH * x)
            )]
            cold_assoc[j, `:=`(
              sw = sw + sum(w),
              sr = sr + sum(w * residC),
              sr2 = sr2 + sum(w * residC^2),
              sx = sx + sum(w * x),
              sx2 = sx2 + sum(w * x^2),
              srx = srx + sum(w * residC * x)
            )]
          }
          
          hgroup <- if (length(heat_breaks) >= 3L) findInterval(yH, heat_breaks, all.inside = TRUE) else rep(1L, length(yH))
          cgroup <- if (length(cold_breaks) >= 3L) findInterval(yC, cold_breaks, all.inside = TRUE) else rep(1L, length(yC))
          
          hq <- data.table(group = hgroup, w = w, r = residH)[, .(
            sw = sum(w), sr = sum(w * r), sr2 = sum(w * r^2)
          ), by = group]
          cq <- data.table(group = cgroup, w = w, r = residC)[, .(
            sw = sum(w), sr = sum(w * r), sr2 = sum(w * r^2)
          ), by = group]
          heat_quantile_acc <- rbindlist(list(heat_quantile_acc, hq), fill = TRUE)[, .(
            sw = sum(sw), sr = sum(sr), sr2 = sum(sr2)
          ), by = group]
          cold_quantile_acc <- rbindlist(list(cold_quantile_acc, cq), fill = TRUE)[, .(
            sw = sum(sw), sr = sum(sr), sr2 = sum(sr2)
          ), by = group]
          
          case_month <- rows$row_month_internal[pairs$case_row[idx]]
          season <- month_to_season(case_month)
          valid_season <- !is.na(season)
          if (any(valid_season)) {
            hs <- data.table(season = season[valid_season], w = w[valid_season], r = residH[valid_season])[, .(
              sw = sum(w), sr = sum(w * r), sr2 = sum(w * r^2)
            ), by = season]
            cs <- data.table(season = season[valid_season], w = w[valid_season], r = residC[valid_season])[, .(
              sw = sum(w), sr = sum(w * r), sr2 = sum(w * r^2)
            ), by = season]
            heat_season_acc <- rbindlist(list(heat_season_acc, hs), fill = TRUE)[, .(
              sw = sum(sw), sr = sum(sr), sr2 = sum(sr2)
            ), by = season]
            cold_season_acc <- rbindlist(list(cold_season_acc, cs), fill = TRUE)[, .(
              sw = sum(sw), sr = sum(sr), sr2 = sum(sr2)
            ), by = season]
          }
        }
        
        rm(X0, predH0, predC0)
        gc(FALSE)
        
        X1 <- build_pair_feature_matrix(rows, pair_index, idx, orientation = 1L)
        pred1 <- predict_xgb_two_models(model_H, model_C, X1)
        predH1 <- pred1$first
        predC1 <- pred1$second
        rm(pred1)
        rH1[idx] <- -yH - predH1
        rC1[idx] <- -yC - predC1
        rm(X1, predH1, predC1, yH, yC, residH, residC)
        gc(FALSE)
      }
      
      diag_counter <- diag_counter + 1L
      fold_diagnostics[[diag_counter]] <- metric_table_from_sums(
        heat_train_sums, heat_test_sums, tune_H, k, "Heat",
        length(train_idx), length(test_idx)
      )
      diag_counter <- diag_counter + 1L
      fold_diagnostics[[diag_counter]] <- metric_table_from_sums(
        cold_train_sums, cold_test_sums, tune_C, k, "Cold",
        length(train_idx), length(test_idx)
      )
      
      if (collect_diagnostics) {
        heat_cal <- weighted_regression_from_sums(
          heat_test_sums["sw"], heat_test_sums["sy"], heat_test_sums["sp"],
          heat_test_sums["sp2"], heat_test_sums["syp"]
        )
        cold_cal <- weighted_regression_from_sums(
          cold_test_sums["sw"], cold_test_sums["sy"], cold_test_sums["sp"],
          cold_test_sums["sp2"], cold_test_sums["syp"]
        )
        calibration_rows[[length(calibration_rows) + 1L]] <- data.table(
          Fold = k,
          Component = c("Heat", "Cold"),
          Calibration_intercept = c(heat_cal["intercept"], cold_cal["intercept"]),
          Calibration_slope = c(heat_cal["slope"], cold_cal["slope"]),
          Mean_observed = c(heat_test_sums["sy"] / heat_test_sums["sw"], cold_test_sums["sy"] / cold_test_sums["sw"]),
          Mean_predicted = c(heat_test_sums["sp"] / heat_test_sums["sw"], cold_test_sums["sp"] / cold_test_sums["sw"])
        )
        
        weighted_cor_from_row <- function(dt) {
          cov_rx <- dt$srx - dt$sr * dt$sx / dt$sw
          var_r <- dt$sr2 - dt$sr^2 / dt$sw
          var_x <- dt$sx2 - dt$sx^2 / dt$sw
          ifelse(var_r > 0 & var_x > 0, cov_rx / sqrt(var_r * var_x), NA_real_)
        }
        heat_assoc[, `:=`(
          Fold = k,
          Component = "Heat",
          Residual_feature_correlation = weighted_cor_from_row(heat_assoc)
        )]
        cold_assoc[, `:=`(
          Fold = k,
          Component = "Cold",
          Residual_feature_correlation = weighted_cor_from_row(cold_assoc)
        )]
        residual_association_rows[[length(residual_association_rows) + 1L]] <- rbindlist(list(
          heat_assoc[, .(Fold, Component, Feature, Residual_feature_correlation)],
          cold_assoc[, .(Fold, Component, Feature, Residual_feature_correlation)]
        ))
        
        if (nrow(heat_quantile_acc)) {
          heat_quantile_acc[, `:=`(
            Fold = k,
            Component = "Heat",
            Residual_mean = sr / sw,
            Residual_RMSE = sqrt(sr2 / sw)
          )]
          quantile_rows[[length(quantile_rows) + 1L]] <- heat_quantile_acc[, .(
            Fold, Component, Quantile_group = group, Residual_mean, Residual_RMSE, Total_weight = sw
          )]
        }
        if (nrow(cold_quantile_acc)) {
          cold_quantile_acc[, `:=`(
            Fold = k,
            Component = "Cold",
            Residual_mean = sr / sw,
            Residual_RMSE = sqrt(sr2 / sw)
          )]
          quantile_rows[[length(quantile_rows) + 1L]] <- cold_quantile_acc[, .(
            Fold, Component, Quantile_group = group, Residual_mean, Residual_RMSE, Total_weight = sw
          )]
        }
        if (nrow(heat_season_acc)) {
          heat_season_acc[, `:=`(
            Fold = k,
            Component = "Heat",
            Residual_mean = sr / sw,
            Residual_RMSE = sqrt(sr2 / sw)
          )]
          season_rows[[length(season_rows) + 1L]] <- heat_season_acc[, .(
            Fold, Component, Season = season, Residual_mean, Residual_RMSE, Total_weight = sw
          )]
        }
        if (nrow(cold_season_acc)) {
          cold_season_acc[, `:=`(
            Fold = k,
            Component = "Cold",
            Residual_mean = sr / sw,
            Residual_RMSE = sqrt(sr2 / sw)
          )]
          season_rows[[length(season_rows) + 1L]] <- cold_season_acc[, .(
            Fold, Component, Season = season, Residual_mean, Residual_RMSE, Total_weight = sw
          )]
        }
      }
      
      if (!quiet) {
        latest_heat <- fold_diagnostics[[diag_counter - 1L]]
        latest_cold <- fold_diagnostics[[diag_counter]]
        message(sprintf(
          "Exposure fold %d/%d: heat held-out RMSE=%.4f, cold held-out RMSE=%.4f",
          fold_position, length(folds), latest_heat$RMSE_test, latest_cold$RMSE_test
        ))
      }
      
      rm(model_H, model_C, train_idx, test_idx, yH_test_all, yC_test_all, w_test_all,
         heat_assoc, cold_assoc, heat_quantile_acc, cold_quantile_acc,
         heat_season_acc, cold_season_acc)
      gc(FALSE)
    }
    
    if (any(!is.finite(rH1)) || any(!is.finite(rC1))) {
      stop("At least one retrospective exposure nuisance prediction is missing or non-finite.")
    }
    
    sw <- residual_moments["sw"]
    mean_H <- residual_moments["sH"] / sw
    mean_C <- residual_moments["sC"] / sw
    residual_covariance <- matrix(c(
      residual_moments["sHH"] / sw - mean_H^2,
      residual_moments["sHC"] / sw - mean_H * mean_C,
      residual_moments["sHC"] / sw - mean_H * mean_C,
      residual_moments["sCC"] / sw - mean_C^2
    ), nrow = 2L, byrow = TRUE)
    dimnames(residual_covariance) <- list(c("Heat", "Cold"), c("Heat", "Cold"))
    
    list(
      rH1 = rH1,
      rC1 = rC1,
      base_H0_set = base_H0_set,
      base_C0_set = base_C0_set,
      tunes = tunes,
      diagnostics = rbindlist(fold_diagnostics, fill = TRUE),
      calibration = rbindlist(calibration_rows, fill = TRUE),
      residual_associations = rbindlist(residual_association_rows, fill = TRUE),
      quantile_diagnostics = rbindlist(quantile_rows, fill = TRUE),
      season_diagnostics = rbindlist(season_rows, fill = TRUE),
      residual_covariance = residual_covariance,
      residual_covariance_min_eigen = min(eigen(residual_covariance, symmetric = TRUE, only.values = TRUE)$values)
    )
  }
  
  # ==============================================================================================
  # 5. Generalized AIPW estimating equation and numerical solution
  # ==============================================================================================
  
  evaluate_aipw <- function(
    beta,
    pair_index,
    exposure_fit,
    pair_idx = NULL,
    included_sets = NULL,
    use_zero_outcome_nuisance = FALSE,
    return_set_components = FALSE
  ) {
    pairs <- pair_index$pairs
    n_pairs <- nrow(pairs)
    n_sets_total <- pair_index$n_sets
    
    use_all_pairs <- is.null(pair_idx)
    n_evaluated_pairs <- if (use_all_pairs) n_pairs else length(pair_idx)
    if (is.null(included_sets)) {
      if (use_all_pairs) {
        included_sets <- seq_len(n_sets_total)
      } else {
        included_sets <- sort(unique(pairs$set_index[pair_idx]))
      }
    }
    S <- length(included_sets)
    if (S < 1L) stop("No matched sets are available for the AIPW estimating equation.")
    
    score_sum_H <- sum(exposure_fit$base_H0_set[included_sets])
    score_sum_C <- sum(exposure_fit$base_C0_set[included_sets])
    J_sum <- matrix(0, nrow = 2L, ncol = 2L)
    
    if (return_set_components) {
      set_score_H <- numeric(n_sets_total)
      set_score_C <- numeric(n_sets_total)
      set_score_H[included_sets] <- exposure_fit$base_H0_set[included_sets]
      set_score_C[included_sets] <- exposure_fit$base_C0_set[included_sets]
      set_ipw_H <- numeric(n_sets_total)
      set_ipw_C <- numeric(n_sets_total)
    }
    
    starts <- seq.int(1L, n_evaluated_pairs, by = AIPW_CHUNK_SIZE)
    for (st in starts) {
      en <- min(n_evaluated_pairs, st + AIPW_CHUNK_SIZE - 1L)
      idx <- if (use_all_pairs) st:en else pair_idx[st:en]
      zH <- pairs$Z_hot_0[idx]
      zC <- pairs$Z_cold_0[idx]
      g <- if (use_zero_outcome_nuisance) 0 else pairs$g_0[idx]
      eta0 <- beta[1L] * zH + beta[2L] * zC + g
      ee <- clamp_machine_exp(eta0)
      w <- 1 / (2 * pairs$M_s[idx])
      
      d1_H <- w * exposure_fit$rH1[idx] * ee
      d1_C <- w * exposure_fit$rC1[idx] * ee
      score_sum_H <- score_sum_H + sum(d1_H)
      score_sum_C <- score_sum_C + sum(d1_C)
      
      J_sum[1L, 1L] <- J_sum[1L, 1L] + sum(w * exposure_fit$rH1[idx] * ee * zH)
      J_sum[1L, 2L] <- J_sum[1L, 2L] + sum(w * exposure_fit$rH1[idx] * ee * zC)
      J_sum[2L, 1L] <- J_sum[2L, 1L] + sum(w * exposure_fit$rC1[idx] * ee * zH)
      J_sum[2L, 2L] <- J_sum[2L, 2L] + sum(w * exposure_fit$rC1[idx] * ee * zC)
      
      if (return_set_components) {
        group <- pairs$set_index[idx]
        set_score_H <- add_grouped_values(set_score_H, group, d1_H)
        set_score_C <- add_grouped_values(set_score_C, group, d1_C)
        ipw_H <- w * (-zH - zH * ee)
        ipw_C <- w * (-zC - zC * ee)
        set_ipw_H <- add_grouped_values(set_ipw_H, group, ipw_H)
        set_ipw_C <- add_grouped_values(set_ipw_C, group, ipw_C)
      }
    }
    
    out <- list(
      score = c(score_sum_H / S, score_sum_C / S),
      J = J_sum / S,
      n_sets = S,
      included_sets = included_sets
    )
    
    if (return_set_components) {
      out$set_components <- data.table(
        set_index = included_sets,
        score_hot = set_score_H[included_sets],
        score_cold = set_score_C[included_sets],
        ipw_hot = set_ipw_H[included_sets],
        ipw_cold = set_ipw_C[included_sets]
      )
      out$set_components[, `:=`(
        augmentation_hot = score_hot - ipw_hot,
        augmentation_cold = score_cold - ipw_cold
      )]
    }
    out
  }
  
  solve_aipw_core <- function(
    pair_index,
    exposure_fit,
    beta_start,
    pair_idx = NULL,
    included_sets = NULL,
    use_zero_outcome_nuisance = FALSE,
    max_iter = NEWTON_MAX_ITER
  ) {
    beta <- as.numeric(beta_start)
    if (length(beta) != 2L || any(!is.finite(beta))) beta <- c(0, 0)
    converged <- FALSE
    used_bfgs <- FALSE
    history <- list()
    
    for (iter in seq_len(max_iter)) {
      ev <- evaluate_aipw(
        beta,
        pair_index,
        exposure_fit,
        pair_idx = pair_idx,
        included_sets = included_sets,
        use_zero_outcome_nuisance = use_zero_outcome_nuisance,
        return_set_components = FALSE
      )
      score_norm <- max(abs(ev$score))
      if (score_norm < NEWTON_TOL_SCORE) {
        converged <- TRUE
        break
      }
      
      step <- tryCatch(
        qr.solve(ev$J, ev$score, tol = 1e-12),
        error = function(e) rep(NA_real_, 2L)
      )
      if (any(!is.finite(step))) {
        step <- tryCatch(
          qr.solve(ev$J + diag(1e-6, 2L), ev$score),
          error = function(e) rep(NA_real_, 2L)
        )
      }
      if (any(!is.finite(step))) break
      if (max(abs(step)) > NEWTON_MAX_STEP) step <- step * NEWTON_MAX_STEP / max(abs(step))
      
      old_objective <- sum(ev$score^2)
      alpha <- 1
      accepted <- FALSE
      for (line_search in seq_len(25L)) {
        beta_try <- beta - alpha * step
        ev_try <- tryCatch(
          evaluate_aipw(
            beta_try,
            pair_index,
            exposure_fit,
            pair_idx = pair_idx,
            included_sets = included_sets,
            use_zero_outcome_nuisance = use_zero_outcome_nuisance,
            return_set_components = FALSE
          ),
          error = function(e) NULL
        )
        if (!is.null(ev_try) && all(is.finite(ev_try$score)) && sum(ev_try$score^2) < old_objective) {
          accepted <- TRUE
          break
        }
        alpha <- alpha / 2
      }
      if (!accepted) break
      
      beta_new <- beta_try
      history[[iter]] <- data.table(
        Iteration = iter,
        beta_hot = beta_new[1L],
        beta_cold = beta_new[2L],
        Score_norm = max(abs(ev_try$score)),
        Step_scale = alpha
      )
      if (
        max(abs(beta_new - beta)) < NEWTON_TOL_BETA &&
        max(abs(ev_try$score)) < NEWTON_TOL_SCORE * 10
      ) {
        beta <- beta_new
        converged <- TRUE
        break
      }
      beta <- beta_new
    }
    
    final <- evaluate_aipw(
      beta,
      pair_index,
      exposure_fit,
      pair_idx = pair_idx,
      included_sets = included_sets,
      use_zero_outcome_nuisance = use_zero_outcome_nuisance,
      return_set_components = FALSE
    )
    
    if (!converged || max(abs(final$score)) > NEWTON_TOL_SCORE * 100) {
      used_bfgs <- TRUE
      objective <- function(b) {
        score <- evaluate_aipw(
          b,
          pair_index,
          exposure_fit,
          pair_idx = pair_idx,
          included_sets = included_sets,
          use_zero_outcome_nuisance = use_zero_outcome_nuisance,
          return_set_components = FALSE
        )$score
        sum(score^2)
      }
      opt <- stats::optim(
        par = beta,
        fn = objective,
        method = "BFGS",
        control = list(maxit = 1000L, reltol = 1e-12)
      )
      beta <- opt$par
      final <- evaluate_aipw(
        beta,
        pair_index,
        exposure_fit,
        pair_idx = pair_idx,
        included_sets = included_sets,
        use_zero_outcome_nuisance = use_zero_outcome_nuisance,
        return_set_components = FALSE
      )
      converged <- opt$convergence == 0L && max(abs(final$score)) < NEWTON_TOL_SCORE * 100
    }
    
    list(
      beta = setNames(beta, c("beta_hot", "beta_cold")),
      score = final$score,
      score_norm = max(abs(final$score)),
      J = final$J,
      converged = converged,
      used_bfgs = used_bfgs,
      history = if (length(history)) rbindlist(history, fill = TRUE) else data.table()
    )
  }
  
  solve_aipw <- function(
    pair_index,
    exposure_fit,
    beta_start,
    set_map,
    collect_diagnostics = TRUE,
    check_multiple_starts = CHECK_MULTIPLE_STARTS
  ) {
    starts <- list(
      Outcome_start = as.numeric(beta_start)
    )
    if (check_multiple_starts) {
      starts$Zero_start <- c(0, 0)
      starts$Positive_perturbation <- as.numeric(beta_start) + c(START_PERTURBATION, START_PERTURBATION)
      starts$Opposing_perturbation <- as.numeric(beta_start) + c(START_PERTURBATION, -START_PERTURBATION)
    }
    
    solutions <- list()
    start_rows <- list()
    for (nm in names(starts)) {
      one <- tryCatch(
        solve_aipw_core(
          pair_index = pair_index,
          exposure_fit = exposure_fit,
          beta_start = starts[[nm]]
        ),
        error = function(e) list(
          beta = c(beta_hot = NA_real_, beta_cold = NA_real_),
          score = c(NA_real_, NA_real_),
          score_norm = Inf,
          J = matrix(NA_real_, 2L, 2L),
          converged = FALSE,
          used_bfgs = FALSE,
          history = data.table(),
          error = safe_error_message(e)
        )
      )
      solutions[[nm]] <- one
      start_rows[[nm]] <- data.table(
        Starting_value = nm,
        Start_beta_hot = starts[[nm]][1L],
        Start_beta_cold = starts[[nm]][2L],
        Estimate_beta_hot = one$beta[1L],
        Estimate_beta_cold = one$beta[2L],
        Score_norm = one$score_norm,
        Converged = one$converged,
        Used_BFGS_fallback = one$used_bfgs,
        Error = one$error %||% NA_character_
      )
    }
    
    successful <- which(vapply(solutions, function(x) isTRUE(x$converged) && all(is.finite(x$beta)), logical(1L)))
    if (length(successful) == 0L) stop("The AIPW estimating equation did not converge from any starting value.")
    best_position <- successful[which.min(vapply(solutions[successful], function(x) x$score_norm, numeric(1L)))]
    best <- solutions[[best_position]]
    
    successful_betas <- do.call(rbind, lapply(solutions[successful], function(x) as.numeric(x$beta)))
    root_spread <- if (nrow(successful_betas) > 1L) {
      max(apply(successful_betas, 2L, function(x) max(x) - min(x)))
    } else {
      0
    }
    if (is.finite(root_spread) && root_spread > MULTISTART_ROOT_TOL) {
      warning(sprintf(
        "Converged roots differed across starting values by as much as %.3e; inspect the multiple-start diagnostic.",
        root_spread
      ))
    }
    
    if (!collect_diagnostics) {
      return(list(
        beta = best$beta,
        RR = setNames(exp(best$beta), c("RR_hot", "RR_cold")),
        score = best$score,
        score_norm = best$score_norm,
        J = best$J,
        Omega = matrix(NA_real_, 2L, 2L),
        vcov_analytic = matrix(NA_real_, 2L, 2L),
        se_analytic = setNames(c(NA_real_, NA_real_), c("SE_hot", "SE_cold")),
        jacobian_min_abs_eigen = matrix_min_abs_eigen(best$J),
        jacobian_condition_number = matrix_condition_number(best$J),
        multiple_starts = rbindlist(start_rows, fill = TRUE),
        root_spread = root_spread,
        score_quantiles = data.table(),
        influential_sets = data.table(),
        leave_one_fold_out = data.table(),
        exposure_only_beta = c(beta_hot = NA_real_, beta_cold = NA_real_),
        component_correlations = data.table(),
        component_means = c(
          IPW_hot = NA_real_, IPW_cold = NA_real_,
          Augmentation_hot = NA_real_, Augmentation_cold = NA_real_
        )
      ))
    }
    
    final <- evaluate_aipw(
      beta = best$beta,
      pair_index = pair_index,
      exposure_fit = exposure_fit,
      return_set_components = TRUE
    )
    set_components <- final$set_components
    S <- nrow(set_components)
    
    Omega <- matrix(c(
      mean(set_components$score_hot^2),
      mean(set_components$score_hot * set_components$score_cold),
      mean(set_components$score_hot * set_components$score_cold),
      mean(set_components$score_cold^2)
    ), nrow = 2L, byrow = TRUE)
    Jinv <- tryCatch(solve(final$J), error = function(e) qr.solve(final$J))
    vcov_analytic <- Jinv %*% Omega %*% t(Jinv) / S
    se_analytic <- sqrt(pmax(diag(vcov_analytic), 0))
    
    influence_hot <- -(Jinv[1L, 1L] * set_components$score_hot + Jinv[1L, 2L] * set_components$score_cold)
    influence_cold <- -(Jinv[2L, 1L] * set_components$score_hot + Jinv[2L, 2L] * set_components$score_cold)
    influence_norm <- sqrt(influence_hot^2 + influence_cold^2)
    
    top_n <- min(as.integer(TOP_INFLUENTIAL_SETS), length(influence_norm))
    top_order <- order(influence_norm, decreasing = TRUE)[seq_len(top_n)]
    influential <- set_components[top_order, .(
      set_index,
      score_hot,
      score_cold,
      ipw_hot,
      ipw_cold,
      augmentation_hot,
      augmentation_cold
    )]
    influential[, `:=`(
      influence_hot = influence_hot[top_order],
      influence_cold = influence_cold[top_order],
      influence_norm = influence_norm[top_order],
      influence_rank = seq_len(.N)
    )]
    influential <- set_map[influential, on = "set_index"]
    
    score_quantiles <- rbindlist(list(
      data.table(
        Component = "Heat",
        Statistic = c("min", "p01", "p05", "median", "p95", "p99", "max"),
        Value = as.numeric(stats::quantile(
          set_components$score_hot,
          probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
          names = FALSE
        ))
      ),
      data.table(
        Component = "Cold",
        Statistic = c("min", "p01", "p05", "median", "p95", "p99", "max"),
        Value = as.numeric(stats::quantile(
          set_components$score_cold,
          probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
          names = FALSE
        ))
      )
    ))
    
    leave_one_fold_out <- data.table()
    if (collect_diagnostics) {
      loo_rows <- list()
      for (k in sort(unique(pair_index$pairs$fold))) {
        pair_idx <- which(pair_index$pairs$fold != k)
        included_sets <- set_map[fold != k, set_index]
        one <- tryCatch(
          solve_aipw_core(
            pair_index = pair_index,
            exposure_fit = exposure_fit,
            beta_start = best$beta,
            pair_idx = pair_idx,
            included_sets = included_sets
          ),
          error = function(e) NULL
        )
        loo_rows[[as.character(k)]] <- data.table(
          Omitted_fold = k,
          beta_hot = if (is.null(one)) NA_real_ else one$beta[1L],
          beta_cold = if (is.null(one)) NA_real_ else one$beta[2L],
          RR_hot = if (is.null(one)) NA_real_ else exp(one$beta[1L]),
          RR_cold = if (is.null(one)) NA_real_ else exp(one$beta[2L]),
          Score_norm = if (is.null(one)) NA_real_ else one$score_norm,
          Converged = if (is.null(one)) FALSE else one$converged
        )
      }
      leave_one_fold_out <- rbindlist(loo_rows, fill = TRUE)
    }
    
    exposure_only <- if (collect_diagnostics) {
      tryCatch(
        solve_aipw_core(
          pair_index = pair_index,
          exposure_fit = exposure_fit,
          beta_start = best$beta,
          use_zero_outcome_nuisance = TRUE
        ),
        error = function(e) NULL
      )
    } else {
      NULL
    }
    
    list(
      beta = best$beta,
      RR = setNames(exp(best$beta), c("RR_hot", "RR_cold")),
      score = final$score,
      score_norm = max(abs(final$score)),
      J = final$J,
      Omega = Omega,
      vcov_analytic = vcov_analytic,
      se_analytic = setNames(se_analytic, c("SE_hot", "SE_cold")),
      jacobian_min_abs_eigen = matrix_min_abs_eigen(final$J),
      jacobian_condition_number = matrix_condition_number(final$J),
      multiple_starts = rbindlist(start_rows, fill = TRUE),
      root_spread = root_spread,
      score_quantiles = score_quantiles,
      influential_sets = influential,
      leave_one_fold_out = leave_one_fold_out,
      exposure_only_beta = if (is.null(exposure_only)) c(beta_hot = NA_real_, beta_cold = NA_real_) else exposure_only$beta,
      component_correlations = data.table(
        Parameter = c("Heat", "Cold"),
        Correlation_IPW_augmentation = c(
          suppressWarnings(stats::cor(set_components$ipw_hot, set_components$augmentation_hot)),
          suppressWarnings(stats::cor(set_components$ipw_cold, set_components$augmentation_cold))
        )
      ),
      component_means = c(
        IPW_hot = mean(set_components$ipw_hot),
        IPW_cold = mean(set_components$ipw_cold),
        Augmentation_hot = mean(set_components$augmentation_hot),
        Augmentation_cold = mean(set_components$augmentation_cold)
      )
    )
  }
  
  # ==============================================================================================
  # 6. Attributable fractions
  # ==============================================================================================
  
  calculate_af <- function(rows, beta, set_map = NULL, keep_cases = TRUE) {
    case_idx <- which(rows$case_internal == 1)
    cases <- data.table(
      set_index = rows$set_index[case_idx],
      case_year = rows$case_year[case_idx],
      A_hot = rows$A_hot[case_idx],
      A_cold = rows$A_cold[case_idx]
    )
    
    log_rr_hot <- beta[1L] * cases$A_hot
    log_rr_cold <- beta[2L] * cases$A_cold
    log_rr_total <- log_rr_hot + log_rr_cold
    
    cases[, `:=`(
      log_RR_total = log_rr_total,
      RR_total = exp(log_rr_total),
      AF_total = 1 - exp(-log_rr_total),
      AF_heat_standalone = 1 - exp(-log_rr_hot),
      AF_cold_standalone = 1 - exp(-log_rr_cold)
    )]
    
    vH <- cases$AF_heat_standalone
    vC <- cases$AF_cold_standalone
    vHC <- cases$AF_total
    cases[, `:=`(
      AF_heat_shapley = 0.5 * (vH + vHC - vC),
      AF_cold_shapley = 0.5 * (vC + vHC - vH)
    )]
    cases[, AF_decomposition_error := AF_total - AF_heat_shapley - AF_cold_shapley]
    
    annual <- cases[, .(
      N_deaths = .N,
      AF_total = mean(AF_total),
      AF_heat_standalone = mean(AF_heat_standalone),
      AF_cold_standalone = mean(AF_cold_standalone),
      AF_heat_shapley = mean(AF_heat_shapley),
      AF_cold_shapley = mean(AF_cold_shapley)
    ), by = .(Year = case_year)]
    setorder(annual, Year)
    
    year_grid <- data.table(Year = as.integer(ANALYSIS_YEARS))
    annual <- annual[year_grid, on = "Year"]
    if (any(is.na(annual$N_deaths))) {
      missing_years <- annual[is.na(N_deaths), Year]
      stop(sprintf(
        "No deaths are available in the following requested analysis years: %s",
        paste(missing_years, collapse = ", ")
      ))
    }
    annual[, Summary := as.character(Year)]
    
    mean_annual <- data.table(
      Year = NA_integer_,
      N_deaths = sum(annual$N_deaths),
      AF_total = mean(annual$AF_total),
      AF_heat_standalone = mean(annual$AF_heat_standalone),
      AF_cold_standalone = mean(annual$AF_cold_standalone),
      AF_heat_shapley = mean(annual$AF_heat_shapley),
      AF_cold_shapley = mean(annual$AF_cold_shapley),
      Summary = sprintf("Mean annual AF, %d-%d", min(ANALYSIS_YEARS), max(ANALYSIS_YEARS))
    )
    
    if (keep_cases && !is.null(set_map)) {
      cases <- set_map[cases, on = "set_index"]
      setcolorder(cases, c(
        "set_index", "set_id_original", "case_year", "A_hot", "A_cold",
        "log_RR_total", "RR_total", "AF_total", "AF_heat_standalone",
        "AF_cold_standalone", "AF_heat_shapley", "AF_cold_shapley",
        "AF_decomposition_error"
      ))
    } else if (!keep_cases) {
      cases <- NULL
    }
    
    list(cases = cases, annual = annual, mean_annual = mean_annual)
  }
  
  # ==============================================================================================
  # 7. Overlap, weight, identification, and estimating-equation diagnostics
  # ==============================================================================================
  
  calculate_aipw_diagnostics <- function(pair_index, exposure_fit, target_fit) {
    pairs <- pair_index$pairs
    n_pairs <- nrow(pairs)
    pi1 <- numeric(n_pairs)
    sum_weighted_pi <- 0
    sum_weighted_inverse <- 0
    sum_weighted_inverse_squared <- 0
    weighted_small_001 <- 0
    weighted_small_005 <- 0
    total_pair_weight <- 0
    
    starts <- seq.int(1L, n_pairs, by = AIPW_CHUNK_SIZE)
    for (st in starts) {
      en <- min(n_pairs, st + AIPW_CHUNK_SIZE - 1L)
      idx <- st:en
      eta0 <- target_fit$beta[1L] * pairs$Z_hot_0[idx] +
        target_fit$beta[2L] * pairs$Z_cold_0[idx] + pairs$g_0[idx]
      p <- expit(-eta0)
      pi1[idx] <- p
      w <- 1 / (2 * pairs$M_s[idx])
      inverse_component_weight <- w / p
      sum_weighted_pi <- sum_weighted_pi + sum(w * p)
      sum_weighted_inverse <- sum_weighted_inverse + sum(inverse_component_weight)
      sum_weighted_inverse_squared <- sum_weighted_inverse_squared + sum(inverse_component_weight^2)
      weighted_small_001 <- weighted_small_001 + sum(w * (p < 0.01))
      weighted_small_005 <- weighted_small_005 + sum(w * (p < 0.05))
      total_pair_weight <- total_pair_weight + sum(w)
    }
    
    pair_weights <- 1 / (2 * pairs$M_s)
    q_probs <- c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1)
    q_pi <- weighted_quantile(pi1, q_probs, pair_weights)
    q_inverse <- 1 / rev(q_pi)
    effective_sample_size <- if (sum_weighted_inverse_squared > 0) {
      sum_weighted_inverse^2 / sum_weighted_inverse_squared
    } else {
      NA_real_
    }
    
    residual_covariance <- exposure_fit$residual_covariance
    component_correlation <- target_fit$component_correlations
    
    summary <- data.table(
      N_matched_sets = pair_index$n_sets,
      N_case_control_pairs = n_pairs,
      N_ordered_pseudo_pairs_implicit = 2 * n_pairs,
      beta_hot = target_fit$beta[1L],
      beta_cold = target_fit$beta[2L],
      RR_hot = exp(target_fit$beta[1L]),
      RR_cold = exp(target_fit$beta[2L]),
      Analytic_SE_hot = target_fit$se_analytic[1L],
      Analytic_SE_cold = target_fit$se_analytic[2L],
      Score_hot = target_fit$score[1L],
      Score_cold = target_fit$score[2L],
      Score_norm = target_fit$score_norm,
      IPW_component_mean_hot = target_fit$component_means["IPW_hot"],
      IPW_component_mean_cold = target_fit$component_means["IPW_cold"],
      Augmentation_mean_hot = target_fit$component_means["Augmentation_hot"],
      Augmentation_mean_cold = target_fit$component_means["Augmentation_cold"],
      Correlation_IPW_augmentation_hot = component_correlation[Parameter == "Heat", Correlation_IPW_augmentation],
      Correlation_IPW_augmentation_cold = component_correlation[Parameter == "Cold", Correlation_IPW_augmentation],
      Jacobian_min_abs_eigen = target_fit$jacobian_min_abs_eigen,
      Jacobian_condition_number = target_fit$jacobian_condition_number,
      Exposure_residual_variance_heat = residual_covariance[1L, 1L],
      Exposure_residual_variance_cold = residual_covariance[2L, 2L],
      Exposure_residual_covariance_heat_cold = residual_covariance[1L, 2L],
      Exposure_residual_covariance_min_eigen = exposure_fit$residual_covariance_min_eigen,
      Effective_sample_size_inverse_probability = effective_sample_size,
      Proportion_pi_case_first_lt_0_01 = weighted_small_001 / total_pair_weight,
      Proportion_pi_case_first_lt_0_05 = weighted_small_005 / total_pair_weight,
      pi_case_first_min = q_pi[1L],
      pi_case_first_p01 = q_pi[2L],
      pi_case_first_p05 = q_pi[3L],
      pi_case_first_mean = sum_weighted_pi / total_pair_weight,
      pi_case_first_median = q_pi[4L],
      pi_case_first_p95 = q_pi[5L],
      pi_case_first_p99 = q_pi[6L],
      pi_case_first_max = q_pi[7L],
      inverse_weight_min = q_inverse[1L],
      inverse_weight_p01 = q_inverse[2L],
      inverse_weight_p05 = q_inverse[3L],
      inverse_weight_mean = sum_weighted_inverse / total_pair_weight,
      inverse_weight_median = q_inverse[4L],
      inverse_weight_p95 = q_inverse[5L],
      inverse_weight_p99 = q_inverse[6L],
      inverse_weight_max = q_inverse[7L],
      Multiple_start_root_spread = target_fit$root_spread
    )
    
    rm(pi1, pair_weights)
    gc(FALSE)
    summary
  }
  
  # ==============================================================================================
  # 8. One complete fit for the main analysis or a bootstrap replicate
  # ==============================================================================================
  
  run_core <- function(
    rows,
    set_map,
    fixed_tunes = NULL,
    fixed_basis_specs = NULL,
    collect_diagnostics = TRUE,
    fit_full_comparator = FALSE,
    keep_case_specific = TRUE,
    run_seed,
    quiet = FALSE
  ) {
    t0 <- tic()
    setDT(rows)
    setorder(rows, set_index, -case_internal)
    rows[, row_uid := .I]
    
    outcome_fit <- fit_outcome_crossfit(
      rows = rows,
      fixed_basis_specs = fixed_basis_specs,
      collect_diagnostics = collect_diagnostics,
      fit_full_comparator = fit_full_comparator,
      quiet = quiet
    )
    
    pair_index <- build_pair_index(rows, outcome_fit$nuisance_oof)
    outcome_fit$nuisance_oof <- NULL
    gc(FALSE)
    
    exposure_fit <- fit_exposure_crossfit(
      rows = rows,
      pair_index = pair_index,
      fixed_tunes = fixed_tunes,
      collect_diagnostics = collect_diagnostics,
      run_seed = run_seed,
      quiet = quiet
    )
    
    target_fit <- solve_aipw(
      pair_index = pair_index,
      exposure_fit = exposure_fit,
      beta_start = outcome_fit$beta_start,
      set_map = set_map,
      collect_diagnostics = collect_diagnostics,
      check_multiple_starts = collect_diagnostics && CHECK_MULTIPLE_STARTS
    )
    
    af_fit <- calculate_af(
      rows = rows,
      beta = target_fit$beta,
      set_map = set_map,
      keep_cases = keep_case_specific
    )
    
    aipw_summary <- if (collect_diagnostics) {
      calculate_aipw_diagnostics(pair_index, exposure_fit, target_fit)
    } else {
      data.table()
    }
    
    estimator_comparison <- data.table(
      Estimator = c(
        outcome_fit$comparator_label,
        "Exposure-regression pathway estimator (g set to zero)",
        "Generalized AIPW estimator"
      ),
      beta_hot = c(
        outcome_fit$comparator_beta[1L],
        target_fit$exposure_only_beta[1L],
        target_fit$beta[1L]
      ),
      beta_cold = c(
        outcome_fit$comparator_beta[2L],
        target_fit$exposure_only_beta[2L],
        target_fit$beta[2L]
      )
    )
    estimator_comparison[, `:=`(
      RR_hot = exp(beta_hot),
      RR_cold = exp(beta_cold)
    )]
    
    result <- list(
      beta = target_fit$beta,
      RR = target_fit$RR,
      analytic_se = target_fit$se_analytic,
      af = af_fit,
      tunes = exposure_fit$tunes,
      basis_specs = outcome_fit$basis_specs,
      comparator_beta = outcome_fit$comparator_beta,
      comparator_label = outcome_fit$comparator_label,
      outcome_diagnostics = outcome_fit$diagnostics,
      outcome_coefficients = outcome_fit$coefficients,
      outcome_calibration = outcome_fit$calibration,
      outcome_residual_scores = outcome_fit$residual_scores,
      outcome_fold_betas = outcome_fit$fold_betas,
      outcome_fold_stability_summary = outcome_fit$fold_stability_summary,
      exposure_diagnostics = exposure_fit$diagnostics,
      exposure_calibration = exposure_fit$calibration,
      exposure_residual_associations = exposure_fit$residual_associations,
      exposure_quantile_diagnostics = exposure_fit$quantile_diagnostics,
      exposure_season_diagnostics = exposure_fit$season_diagnostics,
      aipw_summary = aipw_summary,
      aipw_score_quantiles = target_fit$score_quantiles,
      aipw_influential_sets = target_fit$influential_sets,
      aipw_leave_one_fold_out = target_fit$leave_one_fold_out,
      aipw_multiple_starts = target_fit$multiple_starts,
      estimator_comparison = estimator_comparison,
      score_norm = target_fit$score_norm,
      jacobian_min_abs_eigen = target_fit$jacobian_min_abs_eigen,
      jacobian_condition_number = target_fit$jacobian_condition_number
    )
    
    rm(pair_index, exposure_fit, target_fit, outcome_fit)
    gc(FALSE)
    
    toc(t0, sprintf(
      "Complete fit finished: RR_H=%.6f, RR_C=%.6f",
      result$RR[1L], result$RR[2L]
    ), quiet = quiet)
    result
  }
  
  # ==============================================================================================
  # 9. Main-sample estimation
  # ==============================================================================================
  
  t_all <- tic()
  fit_main <- run_core(
    rows = data_main,
    set_map = set_map_main,
    fixed_tunes = NULL,
    fixed_basis_specs = NULL,
    collect_diagnostics = TRUE,
    fit_full_comparator = FIT_FULL_OUTCOME_COMPARATOR,
    keep_case_specific = SAVE_CASE_SPECIFIC,
    run_seed = ANALYSIS_SEED,
    quiet = FALSE
  )
  
  fixed_tunes_boot <- fit_main$tunes
  fixed_basis_specs_boot <- fit_main$basis_specs
  
  # Save the potentially large case-specific table before the bootstrap and release it from memory.
  early_prefix <- file.path(OUT_DIR, paste0(DATA_TAG, "_case_crossover_AIPW"))
  early_case_file <- paste0(early_prefix, "_case_specific_AF.csv")
  if (SAVE_CASE_SPECIFIC && !is.null(fit_main$af$cases)) {
    fwrite(fit_main$af$cases, early_case_file)
    fit_main$af$cases <- NULL
    gc(FALSE)
  }
  
  # ==============================================================================================
  # 10. Matched-set bootstrap stratified by case year
  # ==============================================================================================
  
  resample_sets_within_year <- function(rows, set_map, seed) {
    set.seed(seed)
    plan <- set_map[, {
      sampled_source <- sample(set_index, .N, replace = TRUE)
      data.table(source_set_index = sampled_source)
    }, by = case_year]
    plan[, new_set_index := .I]
    
    boot_rows <- rows[
      plan[, .(source_set_index, new_set_index)],
      on = .(set_index = source_set_index),
      allow.cartesian = TRUE,
      nomatch = 0L
    ]
    boot_rows[, set_index := new_set_index]
    boot_rows[, new_set_index := NULL]
    setorder(boot_rows, set_index, -case_internal)
    boot_rows[, row_uid := .I]
    
    source_meta <- set_map[, .(
      source_set_index = set_index,
      source_original_id = set_id_original,
      source_fold = fold
    )]
    boot_map <- source_meta[plan, on = "source_set_index"]
    boot_map[, set_index := new_set_index]
    boot_map[, set_id_original := paste0(source_original_id, "__bootstrap_copy_", set_index)]
    boot_map[, fold := source_fold]
    boot_map <- boot_map[, .(set_index, set_id_original, case_year, fold)]
    setorder(boot_map, set_index)
    
    rm(source_meta, plan)
    gc(FALSE)
    list(rows = boot_rows, set_map = boot_map)
  }
  
  boot_rr <- data.table(
    Bootstrap = seq_len(BOOT_B),
    beta_hot = NA_real_,
    beta_cold = NA_real_,
    RR_hot = NA_real_,
    RR_cold = NA_real_,
    Score_norm = NA_real_,
    Success = FALSE,
    Error = NA_character_
  )
  boot_af_rows <- list()
  
  if (BOOT_B > 0L) {
    message("------------------------------------------------------------------------------------------------")
    message("Starting the simplified year-stratified matched-set bootstrap")
    message("MMT, lag windows, outer folds, outcome-basis specifications, and XGBoost hyperparameters are fixed.")
    message("------------------------------------------------------------------------------------------------")
    
    for (b in seq_len(BOOT_B)) {
      tb <- tic()
      one <- tryCatch({
        boot_data <- resample_sets_within_year(
          rows = data_main,
          set_map = set_map_main,
          seed = BOOT_SEED + b
        )
        fit_b <- run_core(
          rows = boot_data$rows,
          set_map = boot_data$set_map,
          fixed_tunes = fixed_tunes_boot,
          fixed_basis_specs = fixed_basis_specs_boot,
          collect_diagnostics = FALSE,
          fit_full_comparator = FALSE,
          keep_case_specific = FALSE,
          run_seed = BOOT_SEED + b * 100000L,
          quiet = TRUE
        )
        rm(boot_data)
        gc(FALSE)
        list(fit = fit_b)
      }, error = function(e) list(error = safe_error_message(e)))
      
      if (!is.null(one$error)) {
        boot_rr[b, `:=`(
          Success = FALSE,
          Error = one$error
        )]
      } else {
        fit_b <- one$fit
        boot_rr[b, `:=`(
          beta_hot = fit_b$beta[1L],
          beta_cold = fit_b$beta[2L],
          RR_hot = exp(fit_b$beta[1L]),
          RR_cold = exp(fit_b$beta[2L]),
          Score_norm = fit_b$score_norm,
          Success = TRUE,
          Error = NA_character_
        )]
        
        annual_b <- copy(fit_b$af$annual)
        annual_b[, `:=`(
          Bootstrap = b,
          Summary_type = "Annual"
        )]
        mean_b <- copy(fit_b$af$mean_annual)
        mean_b[, `:=`(
          Bootstrap = b,
          Summary_type = "Mean_annual"
        )]
        boot_af_rows[[length(boot_af_rows) + 1L]] <- rbindlist(
          list(annual_b, mean_b),
          fill = TRUE
        )
      }
      
      if (b == 1L || b %% 10L == 0L || b == BOOT_B) {
        toc(tb, sprintf(
          "Bootstrap replicate %d/%d completed; cumulative successful replicates: %d",
          b, BOOT_B, sum(boot_rr$Success)
        ))
      }
      rm(one)
      gc(FALSE)
    }
  }
  
  boot_af <- if (length(boot_af_rows) > 0L) {
    rbindlist(boot_af_rows, fill = TRUE)
  } else {
    data.table()
  }
  B_ok <- sum(boot_rr$Success)
  if (BOOT_B > 0L && B_ok < BOOT_MIN_OK) {
    stop(sprintf(
      "Too few successful bootstrap replicates: %d/%d; the minimum required number is %d.",
      B_ok, BOOT_B, BOOT_MIN_OK
    ))
  }
  
  # ==============================================================================================
  # 11. RR estimates and bootstrap confidence intervals
  # ==============================================================================================
  
  beta_main <- fit_main$beta
  comparator_beta <- fit_main$comparator_beta
  
  make_rr_row <- function(side, index, bootstrap_column) {
    bootstrap_beta <- if (BOOT_B > 0L) {
      boot_rr[Success == TRUE][[bootstrap_column]]
    } else {
      numeric(0L)
    }
    se_bootstrap <- if (length(bootstrap_beta) >= 2L) stats::sd(bootstrap_beta) else NA_real_
    ci_lower_log <- if (is.finite(se_bootstrap)) {
      beta_main[index] - stats::qnorm(0.975) * se_bootstrap
    } else {
      NA_real_
    }
    ci_upper_log <- if (is.finite(se_bootstrap)) {
      beta_main[index] + stats::qnorm(0.975) * se_bootstrap
    } else {
      NA_real_
    }
    p_value <- if (is.finite(se_bootstrap) && se_bootstrap > 0) {
      2 * (1 - stats::pnorm(abs(beta_main[index] / se_bootstrap)))
    } else {
      NA_real_
    }
    
    data.table(
      Outcome = OUTCOME_NAME,
      Effect = side,
      MMT = MMT,
      Lag_window = if (index == 1L) {
        sprintf("lag%d-%d", min(HOT_LAGS), max(HOT_LAGS))
      } else {
        sprintf("lag%d-%d", min(COLD_LAGS), max(COLD_LAGS))
      },
      Temperature_change = if (index == 1L) {
        "+1 degree C heat-side deviation"
      } else {
        "+1 degree C cold-side deviation (= 1 degree C lower temperature)"
      },
      beta_AIPW = beta_main[index],
      SE_analytic_diagnostic = fit_main$analytic_se[index],
      SE_bootstrap = se_bootstrap,
      P_Wald_bootstrap = p_value,
      RR = exp(beta_main[index]),
      RR_CI_lower = exp(ci_lower_log),
      RR_CI_upper = exp(ci_upper_log),
      Outcome_only_comparator = fit_main$comparator_label,
      beta_outcome_only = comparator_beta[index],
      RR_outcome_only = exp(comparator_beta[index]),
      Bootstrap_target = BOOT_B,
      Bootstrap_success = B_ok
    )
  }
  
  rr_results <- rbindlist(list(
    make_rr_row("Heat", 1L, "beta_hot"),
    make_rr_row("Cold", 2L, "beta_cold")
  ))
  
  # ==============================================================================================
  # 12. Annual and mean annual AF estimates with percentile bootstrap confidence intervals
  # ==============================================================================================
  
  point_af <- rbindlist(list(
    copy(fit_main$af$annual)[, Summary_type := "Annual"],
    copy(fit_main$af$mean_annual)[, Summary_type := "Mean_annual"]
  ), fill = TRUE)
  
  burden_columns <- c(
    "AF_total",
    "AF_heat_standalone",
    "AF_cold_standalone",
    "AF_heat_shapley",
    "AF_cold_shapley"
  )
  
  af_results <- melt(
    point_af,
    id.vars = c("Year", "N_deaths", "Summary", "Summary_type"),
    measure.vars = burden_columns,
    variable.name = "Burden_component",
    value.name = "AF"
  )
  af_results[, `:=`(
    Outcome = OUTCOME_NAME,
    MMT = MMT,
    AF_CI_lower = NA_real_,
    AF_CI_upper = NA_real_,
    Bootstrap_success = B_ok
  )]
  
  if (nrow(boot_af) > 0L) {
    boot_af_long <- melt(
      boot_af,
      id.vars = c("Bootstrap", "Year", "N_deaths", "Summary", "Summary_type"),
      measure.vars = burden_columns,
      variable.name = "Burden_component",
      value.name = "AF_boot"
    )
    af_ci <- boot_af_long[, .(
      AF_CI_lower = as.numeric(stats::quantile(AF_boot, 0.025, names = FALSE, na.rm = TRUE)),
      AF_CI_upper = as.numeric(stats::quantile(AF_boot, 0.975, names = FALSE, na.rm = TRUE)),
      Bootstrap_success_AF = sum(is.finite(AF_boot))
    ), by = .(Year, Summary_type, Burden_component)]
    
    af_results[
      af_ci,
      on = .(Year, Summary_type, Burden_component),
      `:=`(
        AF_CI_lower = i.AF_CI_lower,
        AF_CI_upper = i.AF_CI_upper,
        Bootstrap_success_AF = i.Bootstrap_success_AF
      )
    ]
  } else {
    af_results[, Bootstrap_success_AF := 0L]
  }
  
  setcolorder(af_results, c(
    "Outcome", "MMT", "Summary_type", "Summary", "Year", "N_deaths",
    "Burden_component", "AF", "AF_CI_lower", "AF_CI_upper",
    "Bootstrap_success_AF", "Bootstrap_success"
  ))
  setorder(af_results, Summary_type, Year, Burden_component)
  
  # ==============================================================================================
  # 13. Diagnostic result tables
  # ==============================================================================================
  
  add_outcome_label <- function(x) {
    x <- copy(x)
    if (nrow(x) > 0L) x[, Outcome := OUTCOME_NAME]
    x
  }
  
  outcome_diag <- add_outcome_label(fit_main$outcome_diagnostics)
  outcome_coefficients <- add_outcome_label(fit_main$outcome_coefficients)
  outcome_calibration <- add_outcome_label(fit_main$outcome_calibration)
  outcome_residual_scores <- add_outcome_label(fit_main$outcome_residual_scores)
  outcome_fold_betas <- add_outcome_label(fit_main$outcome_fold_betas)
  outcome_fold_stability_summary <- add_outcome_label(fit_main$outcome_fold_stability_summary)
  exposure_diag <- add_outcome_label(fit_main$exposure_diagnostics)
  exposure_calibration <- add_outcome_label(fit_main$exposure_calibration)
  exposure_residual_associations <- add_outcome_label(fit_main$exposure_residual_associations)
  exposure_quantile_diag <- add_outcome_label(fit_main$exposure_quantile_diagnostics)
  exposure_season_diag <- add_outcome_label(fit_main$exposure_season_diagnostics)
  aipw_diag <- add_outcome_label(fit_main$aipw_summary)
  aipw_score_quantiles <- add_outcome_label(fit_main$aipw_score_quantiles)
  aipw_influential_sets <- add_outcome_label(fit_main$aipw_influential_sets)
  aipw_leave_one_fold_out <- add_outcome_label(fit_main$aipw_leave_one_fold_out)
  aipw_multiple_starts <- add_outcome_label(fit_main$aipw_multiple_starts)
  estimator_comparison <- add_outcome_label(fit_main$estimator_comparison)
  tune_diag <- flatten_tunes(fit_main$tunes)
  if (nrow(tune_diag) > 0L) tune_diag[, Outcome := OUTCOME_NAME]
  
  # ==============================================================================================
  # 14. Save output files
  # ==============================================================================================
  
  prefix <- file.path(OUT_DIR, paste0(DATA_TAG, "_case_crossover_AIPW"))
  
  files <- list(
    RR = paste0(prefix, "_RR_results.csv"),
    AF = paste0(prefix, "_AF_annual_mean_results.csv"),
    case_specific_AF = paste0(prefix, "_case_specific_AF.csv"),
    outcome_folds = paste0(prefix, "_diagnostic_outcome_folds.csv"),
    outcome_coefficients = paste0(prefix, "_diagnostic_outcome_coefficients.csv"),
    outcome_calibration = paste0(prefix, "_diagnostic_outcome_calibration.csv"),
    outcome_residual_scores = paste0(prefix, "_diagnostic_outcome_residual_scores.csv"),
    outcome_fold_betas = paste0(prefix, "_diagnostic_outcome_fold_estimates.csv"),
    outcome_fold_stability_summary = paste0(prefix, "_diagnostic_outcome_fold_stability_summary.csv"),
    exposure_folds = paste0(prefix, "_diagnostic_exposure_folds.csv"),
    exposure_calibration = paste0(prefix, "_diagnostic_exposure_calibration.csv"),
    exposure_residual_associations = paste0(prefix, "_diagnostic_exposure_residual_associations.csv"),
    exposure_quantiles = paste0(prefix, "_diagnostic_exposure_quantiles.csv"),
    exposure_seasons = paste0(prefix, "_diagnostic_exposure_seasons.csv"),
    AIPW_summary = paste0(prefix, "_diagnostic_AIPW_summary.csv"),
    AIPW_score_quantiles = paste0(prefix, "_diagnostic_AIPW_score_quantiles.csv"),
    AIPW_influential_sets = paste0(prefix, "_diagnostic_AIPW_influential_sets.csv"),
    AIPW_leave_one_fold_out = paste0(prefix, "_diagnostic_AIPW_leave_one_fold_out.csv"),
    AIPW_multiple_starts = paste0(prefix, "_diagnostic_AIPW_multiple_starts.csv"),
    estimator_comparison = paste0(prefix, "_diagnostic_estimator_comparison.csv"),
    XGBoost_hyperparameters = paste0(prefix, "_XGBoost_hyperparameters.csv"),
    bootstrap_RR = paste0(prefix, "_bootstrap_RR_trace.csv"),
    bootstrap_AF = paste0(prefix, "_bootstrap_AF_trace.csv"),
    compact_RDS = paste0(prefix, "_compact_results.rds"),
    session_info = paste0(prefix, "_session_info.txt")
  )
  
  fwrite(rr_results, files$RR)
  fwrite(af_results, files$AF)
  write_if_nonempty <- function(x, path) {
    if (!is.null(x) && nrow(x) > 0L) fwrite(x, path)
  }
  
  write_if_nonempty(outcome_diag, files$outcome_folds)
  write_if_nonempty(outcome_coefficients, files$outcome_coefficients)
  write_if_nonempty(outcome_calibration, files$outcome_calibration)
  write_if_nonempty(outcome_residual_scores, files$outcome_residual_scores)
  write_if_nonempty(outcome_fold_betas, files$outcome_fold_betas)
  write_if_nonempty(outcome_fold_stability_summary, files$outcome_fold_stability_summary)
  write_if_nonempty(exposure_diag, files$exposure_folds)
  write_if_nonempty(exposure_calibration, files$exposure_calibration)
  write_if_nonempty(exposure_residual_associations, files$exposure_residual_associations)
  write_if_nonempty(exposure_quantile_diag, files$exposure_quantiles)
  write_if_nonempty(exposure_season_diag, files$exposure_seasons)
  write_if_nonempty(aipw_diag, files$AIPW_summary)
  write_if_nonempty(aipw_score_quantiles, files$AIPW_score_quantiles)
  write_if_nonempty(aipw_influential_sets, files$AIPW_influential_sets)
  write_if_nonempty(aipw_leave_one_fold_out, files$AIPW_leave_one_fold_out)
  write_if_nonempty(aipw_multiple_starts, files$AIPW_multiple_starts)
  write_if_nonempty(estimator_comparison, files$estimator_comparison)
  write_if_nonempty(tune_diag, files$XGBoost_hyperparameters)
  fwrite(boot_rr, files$bootstrap_RR)
  write_if_nonempty(boot_af, files$bootstrap_AF)
  
  session_lines <- capture.output(sessionInfo())
  writeLines(session_lines, files$session_info)
  
  if (SAVE_COMPACT_RDS) {
    saveRDS(list(
      parameters = list(
        DATA_RDS = DATA_RDS,
        OUTCOME_NAME = OUTCOME_NAME,
        MMT = MMT,
        ANALYSIS_YEARS = ANALYSIS_YEARS,
        HOT_LAGS = HOT_LAGS,
        COLD_LAGS = COLD_LAGS,
        RH_LAGS = RH_LAGS,
        RH_SPLINE_DF = RH_SPLINE_DF,
        K_fold = K_fold,
        BOOT_B = BOOT_B,
        tune_frac = tune_frac,
        tune_try_random = tune_try_random,
        tune_inner_folds = tune_inner_folds,
        FIT_FULL_OUTCOME_COMPARATOR = FIT_FULL_OUTCOME_COMPARATOR,
        CHECK_MULTIPLE_STARTS = CHECK_MULTIPLE_STARTS,
        SEED_MASTER = SEED_MASTER,
        ANALYSIS_SEED = ANALYSIS_SEED,
        bootstrap_conditions = c(
          "MMT fixed",
          "lag windows fixed",
          "outer fold allocation fixed",
          "outcome basis specifications fixed",
          "fold-specific XGBoost hyperparameters fixed"
        )
      ),
      RR_results = rr_results,
      AF_results = af_results,
      case_specific_AF_file = if (SAVE_CASE_SPECIFIC) files$case_specific_AF else NULL,
      diagnostics = list(
        outcome_folds = outcome_diag,
        outcome_coefficients = outcome_coefficients,
        outcome_calibration = outcome_calibration,
        outcome_residual_scores = outcome_residual_scores,
        outcome_fold_estimates = outcome_fold_betas,
        outcome_fold_stability_summary = outcome_fold_stability_summary,
        exposure_folds = exposure_diag,
        exposure_calibration = exposure_calibration,
        exposure_residual_associations = exposure_residual_associations,
        exposure_quantiles = exposure_quantile_diag,
        exposure_seasons = exposure_season_diag,
        AIPW_summary = aipw_diag,
        AIPW_score_quantiles = aipw_score_quantiles,
        AIPW_influential_sets = aipw_influential_sets,
        AIPW_leave_one_fold_out = aipw_leave_one_fold_out,
        AIPW_multiple_starts = aipw_multiple_starts,
        estimator_comparison = estimator_comparison
      ),
      XGBoost_hyperparameters = tune_diag,
      bootstrap_RR = boot_rr,
      bootstrap_AF = boot_af,
      files = files
    ), files$compact_RDS)
  }
  
  message("------------------------------------------------------------------------------------------------")
  message(sprintf("[%s] Main relative-rate results", OUTCOME_NAME))
  message(sprintf(
    "Heat: RR=%.6f (95%% CI %.6f, %.6f)",
    rr_results[Effect == "Heat", RR],
    rr_results[Effect == "Heat", RR_CI_lower],
    rr_results[Effect == "Heat", RR_CI_upper]
  ))
  message(sprintf(
    "Cold: RR=%.6f (95%% CI %.6f, %.6f)",
    rr_results[Effect == "Cold", RR],
    rr_results[Effect == "Cold", RR_CI_lower],
    rr_results[Effect == "Cold", RR_CI_upper]
  ))
  message(sprintf("Results were saved under: %s", OUT_DIR))
  toc(t_all, "All analyses completed")
  
  invisible(list(
    RR_results = rr_results,
    AF_results = af_results,
    case_specific_AF_file = if (SAVE_CASE_SPECIFIC) files$case_specific_AF else NULL,
    diagnostics = list(
      outcome_folds = outcome_diag,
      outcome_coefficients = outcome_coefficients,
      outcome_calibration = outcome_calibration,
      outcome_residual_scores = outcome_residual_scores,
      outcome_fold_estimates = outcome_fold_betas,
      outcome_fold_stability_summary = outcome_fold_stability_summary,
      exposure_folds = exposure_diag,
      exposure_calibration = exposure_calibration,
      exposure_residual_associations = exposure_residual_associations,
      exposure_quantiles = exposure_quantile_diag,
      exposure_seasons = exposure_season_diag,
      AIPW_summary = aipw_diag,
      AIPW_score_quantiles = aipw_score_quantiles,
      AIPW_influential_sets = aipw_influential_sets,
      AIPW_leave_one_fold_out = aipw_leave_one_fold_out,
      AIPW_multiple_starts = aipw_multiple_starts,
      estimator_comparison = estimator_comparison
    ),
    bootstrap = list(RR = boot_rr, AF = boot_af),
    XGBoost_hyperparameters = tune_diag,
    files = files
  ))
}

# Backward-compatible alias for earlier analysis scripts.
aipw <- case_crossover_aipw

####################################################################################################
# EXAMPLE CALLS
# -------------
# The examples below are not executed automatically. Copy and edit one call for each outcome.
####################################################################################################

# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/LRI.rds",
#   OUTCOME_NAME = "LRI",
#   MMT = 22.3,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/CMM.rds",
#   OUTCOME_NAME = "CMM",
#   MMT = 23.3,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/CKD.rds",
#   OUTCOME_NAME = "CKD",
#   MMT = 23.7,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/HHD.rds",
#   OUTCOME_NAME = "HHD",
#   MMT = 23.5,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/SHAIV.rds",
#   OUTCOME_NAME = "SHAIV",
#   MMT = -22.4,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/UI.rds",
#   OUTCOME_NAME = "UI",
#   MMT = -19.2,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/TI.rds",
#   OUTCOME_NAME = "TI",
#   MMT = -21.5,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/IHD.rds",
#   OUTCOME_NAME = "IHD",
#   MMT = 23.0,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/DM.rds",
#   OUTCOME_NAME = "DM",
#   MMT = 22.9,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/COPD.rds",
#   OUTCOME_NAME = "COPD",
#   MMT = 23.1,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )
# 
# result <- case_crossover_aipw(
#   DATA_RDS = "/xxx/STROKE.rds",
#   OUTCOME_NAME = "STROKE",
#   MMT = 23.0,
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L,
#   K_fold = 3L
# )


