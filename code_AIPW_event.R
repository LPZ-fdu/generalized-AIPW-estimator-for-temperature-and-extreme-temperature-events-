####################################################################################################
# REVIEWER-READY ANALYSIS CODE
# Cross-fitted generalized augmented inverse probability weighting estimator for the additional
# conditional mortality rate ratio associated with recent extreme-temperature event history in a
# time-stratified case-crossover design
####################################################################################################
#
# OVERVIEW
# --------
# This script implements the outcome- and event-specific analysis described in Section B of the
# methodological appendix. Heatwaves and cold spells are analysed separately. For one mortality
# outcome and one fixed event definition, the script estimates the conditional mortality rate ratio
# associated with a binary recent event-history exposure after adjustment for the same heat-side and
# cold-side non-optimal-temperature components used in Section A, together with other measured
# time-varying covariates.
#
# The implementation has four principal components:
#
#   1. Event-history and non-optimal-temperature adjustment construction
#      E_event = 1 if at least one qualifying event day occurred within EVENT_HISTORY_LAGS.
#      A_hot   = max(mean temperature over HOT_LAGS  - MMT, 0).
#      A_cold  = max(MMT - mean temperature over COLD_LAGS, 0).
#
#   2. Prospective outcome nuisance model
#      A cross-fitted conditional logistic regression is fitted to the original one-case-multiple-
#      control matched risk sets. XGBoost is not used for outcome-model estimation.
#
#   3. Retrospective exposure nuisance model
#      Each case-control comparison is represented by two ordered orientations. XGBoost squared-
#      error regression estimates E(Z | D = 0, X), where Z is the pairwise event-history contrast
#      and D = 0 denotes the orientation in which the first member is the control day.
#
#   4. Generalized AIPW target estimator
#      The event-history log mortality rate-ratio coefficient is obtained by solving the scalar,
#      cross-fitted generalized AIPW estimating equation. This is the one-dimensional counterpart of
#      the joint heat/cold estimator used for non-optimal daily temperature.
#
# The script also calculates full-calendar-year case-specific attributable fractions, annual
# attributable fractions, equal-year mean annual attributable fractions, bootstrap confidence
# intervals, and the model diagnostics specified in Section B. Deaths outside the event-specific
# season are assigned zero event-history exposure and remain in the annual denominator.
#
# THIS SCRIPT DOES NOT IMPLEMENT A SAMPLED-DAY PROPENSITY-SCORE AIPW ESTIMATOR
# --------------------------------------------------------------------------------
# No propensity score P(E = 1 | C), propensity-score clipping, inverse-treatment weighting, or SMD
# balance analysis is used. The inverse probability in the generalized AIPW score is the conditional
# probability that the first member of an ordered discordant pair is the case day.
#
# REQUIRED INPUT DATA
# -------------------
# DATA_RDS must contain one row per sampled day and, at minimum:
#
#   - ID_COL: matched-set identifier (default: "id")
#   - CASE_COL: case indicator coded 1 for the case day and 0 for control days
#   - YEAR_COL or DATE_COL: calendar information used to identify the case year
#   - MONTH_COL or DATE_COL: calendar month used for event-season restriction
#   - temp_lag0, temp_lag1, ... through max(HOT_LAGS, COLD_LAGS), plus any additional lags
#     required to reconstruct the fixed event definition when EVENT_LAG_COLS are not supplied
#   - rh_lag0, rh_lag1, ... through max(RH_LAGS)
#   - HOLIDAY_COL, if available; otherwise the holiday indicator is set to zero
#   - any variables named in OUTCOME_EXTRA_VARS or EXPOSURE_EXTRA_VARS
#
# The recent event-history exposure can be supplied in either of two ways:
#
#   A. Existing lag-specific event indicators (recommended)
#      Supply EVENT_LAG_COLS in the same order as EVENT_HISTORY_LAGS. Each column must be coded 0/1
#      and indicate whether the corresponding lag date belonged to a qualifying event episode.
#
#   B. Row-wise reconstruction from lagged sampled-day temperatures
#      When EVENT_LAG_COLS are not supplied, the script reproduces the event-definition screening
#      procedure. It calculates city-specific percentile thresholds from unique city-date-temperature
#      records and uses each sampled-day record's own temp_lag0, temp_lag1, ... sequence to construct
#      lag-specific event indicators. Different residential grids within the same city-date may have
#      different temperatures and event status. The maximum required temperature lag is:
#
#          max(EVENT_HISTORY_LAGS) + MIN_DURATION_DAYS - 1
#
#      THRESHOLD_BASE = "year_round" reproduces the primary screening analysis. Every day in the
#      minimum-duration block must fall within the corresponding warm or cool season.
#
# EVENT_DEFINITION, THRESHOLD_PERCENTILE, and MIN_DURATION_DAYS are treated as fixed analytical
# choices. If existing event-lag columns are supplied, THRESHOLD_PERCENTILE and MIN_DURATION_DAYS are
# used for metadata and support reporting but are not used to reconstruct the supplied indicators.
#
# TYPICAL SINGLE-EVENT USE
# ------------------------
# result <- extreme_event_case_crossover_aipw(
#   DATA_RDS              = "/path/to/COPD.rds",
#   MMT                   = 23.1,
#   EVENT_TYPE            = "heatwave",
#   EVENT_DEFINITION      = "HW_P97p5_D3",
#   THRESHOLD_PERCENTILE  = 0.975,
#   MIN_DURATION_DAYS     = 3L,
#   EVENT_LAG_COLS        = paste0("HW_P97p5_D3_lag", 0:10),
#   OUTCOME_NAME          = "COPD",
#   OUT_DIR               = "/path/to/results",
#   BOOT_B                = 500L
# )
#
# TYPICAL DEFINITION-TABLE USE
# ----------------------------
# result <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS       = "/path/to/COPD.rds",
#   MMT            = 23.1,
#   OUTCOME_NAME   = "COPD",
#   DEFINITION_XLSX = "/path/to/best_definitions.xlsx",
#   OUT_DIR        = "/path/to/results",
#   BOOT_B         = 500L
# )
#
# MAIN OUTPUTS FOR EACH EVENT TYPE
# --------------------------------
# *_RR_results.csv
#   Generalized AIPW event-history RR, bootstrap confidence interval, P value, analytic diagnostic
#   standard error, and conditional-logistic comparator.
#
# *_AF_annual_mean_results.csv
#   Full-calendar-year annual and equal-year mean annual event-history attributable fractions, with
#   outside-season deaths assigned zero event history and percentile bootstrap confidence intervals.
#
# *_case_specific_AF.csv
#   Full-calendar-year case-specific event-history status, fitted event-related RR, and standalone
#   conditional event-history attributable fraction. Outside-season deaths have E_event = 0.
#
# *_event_support.csv
#   Matched-set, sampled-day, event-day, exposed-case, and informative-pair support summaries.
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
# *_diagnostic_exposure_prediction_distribution.csv
#   XGBoost retrospective exposure conditional-mean diagnostics among held-out D = 0 pairs.
#
# *_diagnostic_AIPW_summary.csv
# *_diagnostic_AIPW_score_quantiles.csv
# *_diagnostic_AIPW_influential_sets.csv
# *_diagnostic_AIPW_leave_one_fold_out.csv
# *_diagnostic_AIPW_multiple_starts.csv
# *_diagnostic_estimator_comparison.csv
#   Conditional case-probability, inverse-probability, Jacobian, score, influence, fold-deletion,
#   multiple-start, and component-estimator diagnostics.
#
# *_bootstrap_RR_trace.csv and *_bootstrap_AF_trace.csv
#   Replicate-specific estimates for reproducibility and later aggregation across mutually exclusive
#   mortality outcomes.
#
# *_XGBoost_hyperparameters.csv, *_compact_results.rds, and *_session_info.txt
#   Selected fold-specific exposure-model settings, compact machine-readable results, and the software
#   environment.
#
# IMPORTANT IMPLEMENTATION NOTES
# ------------------------------
# - Heatwave and cold-spell analyses are separate scalar models.
# - Seasonal restriction is applied at the matched-set level according to the case-day month; all
#   sampled days in each retained matched set are kept.
# - The conditional logistic outcome model is fitted to the original matched risk sets, not to
#   pseudo-pairs.
# - Both ordered orientations are represented in the generalized AIPW score. Only the D = 0
#   orientation is used to train the retrospective exposure conditional-mean model.
# - Every ordered pseudo-pair has weight 1/(2 M_s); every D = 0 exposure-training record has weight
#   1/M_s. Thus each original matched set contributes total weight one.
# - No probability clipping is used in the primary generalized AIPW estimator. Floating-point
#   overflow safeguards are numerical checks rather than statistical truncation.
# - Analytic sandwich standard errors are diagnostic. Primary confidence intervals use the matched-
#   set bootstrap.
# - The event model adjusts jointly for the heat-side and cold-side non-optimal-temperature summaries
#   defined exactly as in Section A. Their coefficients are nuisance adjustment coefficients and do not
#   replace the primary non-optimal-temperature coefficients estimated in Section A.
# - The bootstrap resamples event-season matched sets within case year, retains their original outer-
#   fold assignments, fixes the event definition, MMT, heat/cold lag windows, outcome-basis
#   specifications, and fold-specific XGBoost hyperparameters, and refits all nuisance and target models.
# - Annual AF confidence intervals condition on the observed full-year outcome-specific death counts;
#   outside-season deaths contribute zero event-history AF.
# - Heatwave and cold-spell AFs are combined additively because their analysis seasons do not overlap.
#   The approximate total temperature AF is the sum of the independently estimated non-optimal and
#   additional extreme-event AF point estimates; joint uncertainty is not propagated.
# - The implementation is memory-conscious: raw lag columns and event-history source columns are
#   discarded after compact variables are constructed, full conditional-logistic objects are not
#   retained across folds, pair features are generated on demand, and bootstrap fits use reduced
#   diagnostics.
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

extreme_event_case_crossover_aipw <- function(
    DATA_RDS,
    MMT,
    EVENT_TYPE = c("heatwave", "cold_spell"),
    EVENT_DEFINITION = NULL,
    THRESHOLD_PERCENTILE = NULL,
    MIN_DURATION_DAYS = NULL,
    OUT_DIR = file.path(getwd(), "extreme_event_case_crossover_AIPW_results"),
    OUTCOME_NAME = NULL,
    BOOT_B = 500L,
    K_fold = 3L,
    ANALYSIS_YEARS = 2013:2019,
    EVENT_HISTORY_LAGS = NULL,
    HOT_LAGS = 0:3,
    COLD_LAGS = 0:14,
    RH_LAGS = 0:3,
    WARM_MONTHS = 5:9,
    COOL_MONTHS = c(11L, 12L, 1L, 2L, 3L),
    HOT_ADJUSTMENT_DF = 1L,
    COLD_ADJUSTMENT_DF = 1L,
    RH_SPLINE_DF = 3L,
    ID_COL = "id",
    CASE_COL = "case",
    YEAR_COL = "year",
    DATE_COL = "date",
    MONTH_COL = NULL,
    CITY_COL = "city",
    HOLIDAY_COL = "holiday",
    EVENT_LAG_COLS = NULL,
    EVENT_THRESHOLD_REFERENCE_YEARS = ANALYSIS_YEARS,
    THRESHOLD_BASE = c("year_round", "season_specific"),
    MIN_QUALIFYING_EVENT_DAYS = 5L,
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
    DIAGNOSTIC_PREDICTION_SAMPLE = 200000L,
    NEWTON_MAX_ITER = 100L,
    NEWTON_TOL_SCORE = 1e-8,
    NEWTON_TOL_BETA = 1e-8,
    NEWTON_MAX_STEP = 1,
    CHECK_MULTIPLE_STARTS = TRUE,
    START_PERTURBATION = 0.10,
    MULTISTART_ROOT_TOL = 1e-5,
    TOP_INFLUENTIAL_SETS = 100L,
    BOOT_MIN_SUCCESS_FRAC = 0.70,
    SAVE_CASE_SPECIFIC = TRUE,
    SAVE_COMPACT_RDS = TRUE,
    SEED_MASTER = 20260712L
) {
  
  # ==============================================================================================
  # 0. Argument validation, event-specific defaults, and general utilities
  # ==============================================================================================
  
  stopifnot(is.character(DATA_RDS), length(DATA_RDS) == 1L, nzchar(DATA_RDS))
  if (!file.exists(DATA_RDS)) stop(sprintf("Input file does not exist: %s", DATA_RDS))
  stopifnot(is.numeric(MMT), length(MMT) == 1L, is.finite(MMT))
  stopifnot(K_fold >= 2L)
  stopifnot(length(ANALYSIS_YEARS) >= 1L)
  stopifnot(length(HOT_LAGS) >= 1L, length(COLD_LAGS) >= 1L, length(RH_LAGS) >= 1L)
  stopifnot(HOT_ADJUSTMENT_DF >= 1L, COLD_ADJUSTMENT_DF >= 1L, RH_SPLINE_DF >= 1L)
  stopifnot(BOOT_B >= 0L)
  if (BOOT_B == 1L) stop("BOOT_B must be 0 or at least 2.")
  if (!CLOGIT_METHOD %in% c("exact", "efron", "breslow")) {
    stop("CLOGIT_METHOD must be one of: exact, efron, breslow.")
  }
  if (tune_frac <= 0 || tune_frac > 1) stop("tune_frac must be in (0, 1].")
  if (DIAGNOSTIC_GROUPS < 2L) stop("DIAGNOSTIC_GROUPS must be at least 2.")
  if (MIN_QUALIFYING_EVENT_DAYS < 1L) stop("MIN_QUALIFYING_EVENT_DAYS must be positive.")
  if (DIAGNOSTIC_PREDICTION_SAMPLE < 1000L) stop("DIAGNOSTIC_PREDICTION_SAMPLE must be at least 1,000.")
  if (XGB_PREDICT_CHUNK_SIZE < 1000L || AIPW_CHUNK_SIZE < 1000L) {
    stop("Chunk sizes must be at least 1,000 rows.")
  }
  extra_names <- unique(c(OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS))
  if (length(extra_names) > 0L && any(make.names(extra_names) != extra_names)) {
    stop("OUTCOME_EXTRA_VARS and EXPOSURE_EXTRA_VARS must use syntactically valid R column names.")
  }
  reserved_input_names <- unique(na.omit(c(
    ID_COL, CASE_COL, YEAR_COL, DATE_COL, MONTH_COL, CITY_COL, HOLIDAY_COL
  )))
  if (length(intersect(extra_names, reserved_input_names)) > 0L) {
    stop("Additional covariates must not duplicate identifier, case, calendar, city, or holiday columns.")
  }
  
  normalize_event_type <- function(x) {
    x <- tolower(gsub("[- ]", "_", as.character(x)[1L]))
    if (x %in% c("heat", "heatwave", "heat_wave", "hw")) return("heatwave")
    if (x %in% c("cold", "coldspell", "cold_spell", "cs")) return("cold_spell")
    stop("EVENT_TYPE must identify a heatwave or cold spell.")
  }
  
  EVENT_TYPE <- normalize_event_type(EVENT_TYPE[1L])
  THRESHOLD_BASE <- match.arg(THRESHOLD_BASE)
  event_short <- if (EVENT_TYPE == "heatwave") "heat" else "cold"
  event_label <- if (EVENT_TYPE == "heatwave") "Heatwave" else "Cold spell"
  event_months <- if (EVENT_TYPE == "heatwave") as.integer(WARM_MONTHS) else as.integer(COOL_MONTHS)
  event_direction <- if (EVENT_TYPE == "heatwave") "upper" else "lower"
  
  if (is.null(EVENT_HISTORY_LAGS)) {
    EVENT_HISTORY_LAGS <- if (EVENT_TYPE == "heatwave") 0:10 else 0:21
  }
  EVENT_HISTORY_LAGS <- sort(unique(as.integer(EVENT_HISTORY_LAGS)))
  HOT_LAGS <- sort(unique(as.integer(HOT_LAGS)))
  COLD_LAGS <- sort(unique(as.integer(COLD_LAGS)))
  RH_LAGS <- sort(unique(as.integer(RH_LAGS)))
  
  if (!identical(EVENT_HISTORY_LAGS, 0:max(EVENT_HISTORY_LAGS))) {
    stop("EVENT_HISTORY_LAGS must be the contiguous sequence 0:L_E.")
  }
  if (any(HOT_LAGS < 0L) || any(COLD_LAGS < 0L) || any(RH_LAGS < 0L)) {
    stop("Lag indices must be non-negative.")
  }
  
  EVENT_DEFINITION <- as.character(EVENT_DEFINITION %||% paste0(event_label, "_fixed_definition"))[1L]
  if (!nzchar(EVENT_DEFINITION)) stop("EVENT_DEFINITION must be non-empty.")
  
  if (!is.null(THRESHOLD_PERCENTILE)) {
    THRESHOLD_PERCENTILE <- as.numeric(THRESHOLD_PERCENTILE)[1L]
    if (THRESHOLD_PERCENTILE > 1) THRESHOLD_PERCENTILE <- THRESHOLD_PERCENTILE / 100
    if (!is.finite(THRESHOLD_PERCENTILE) || THRESHOLD_PERCENTILE <= 0 || THRESHOLD_PERCENTILE >= 1) {
      stop("THRESHOLD_PERCENTILE must lie strictly between 0 and 1.")
    }
  }
  if (!is.null(THRESHOLD_PERCENTILE)) {
    if (EVENT_TYPE == "heatwave" && THRESHOLD_PERCENTILE < 0.5) {
      warning("The heatwave threshold percentile is below 0.5; verify the selected definition.")
    }
    if (EVENT_TYPE == "cold_spell" && THRESHOLD_PERCENTILE > 0.5) {
      warning("The cold-spell threshold percentile is above 0.5; verify the selected definition.")
    }
  }
  if (!is.null(MIN_DURATION_DAYS)) {
    MIN_DURATION_DAYS <- as.integer(MIN_DURATION_DAYS)[1L]
    if (!is.finite(MIN_DURATION_DAYS) || MIN_DURATION_DAYS < 1L) {
      stop("MIN_DURATION_DAYS must be a positive integer.")
    }
  }
  
  if (!is.null(EVENT_LAG_COLS)) {
    EVENT_LAG_COLS <- as.character(EVENT_LAG_COLS)
    if (length(EVENT_LAG_COLS) != length(EVENT_HISTORY_LAGS)) {
      stop("EVENT_LAG_COLS must have the same length and order as EVENT_HISTORY_LAGS.")
    }
  } else if (is.null(THRESHOLD_PERCENTILE) || is.null(MIN_DURATION_DAYS)) {
    stop(
      paste0(
        "Supply EVENT_LAG_COLS, or supply THRESHOLD_PERCENTILE and MIN_DURATION_DAYS so that ",
        "event history can be reconstructed row by row from the sampled-day lagged temperatures."
      )
    )
  }
  
  set.seed(SEED_MASTER)
  Sys.setenv(OMP_NUM_THREADS = as.character(xgb_nthread))
  
  stable_string_seed <- function(x) {
    ints <- utf8ToInt(as.character(x)[1L])
    if (length(ints) == 0L) return(0L)
    as.integer(sum((seq_along(ints) + 17L) * ints) %% 100000000L)
  }
  
  DATA_TAG <- tools::file_path_sans_ext(basename(DATA_RDS))
  OUTCOME_NAME <- as.character(OUTCOME_NAME %||% DATA_TAG)[1L]
  EVENT_TAG <- gsub("[^A-Za-z0-9]+", "_", EVENT_DEFINITION)
  EVENT_TAG <- gsub("^_+|_+$", "", EVENT_TAG)
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  
  ANALYSIS_SEED <- as.integer(
    SEED_MASTER + stable_string_seed(paste(OUTCOME_NAME, EVENT_TYPE, EVENT_DEFINITION, sep = "|"))
  )
  BOOT_SEED <- ANALYSIS_SEED + 50000L
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
    if (any(x > 700, na.rm = TRUE)) {
      stop(
        "The AIPW estimating equation encountered exp(eta) overflow risk (eta > 700); inspect overlap.",
        call. = FALSE
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
    stop(sprintf("Variable %s must be numerically encoded before analysis.", name))
  }
  
  coerce_to_date <- function(x, name) {
    if (inherits(x, "Date")) return(x)
    if (inherits(x, "IDate")) return(as.Date(x))
    if (inherits(x, "POSIXt")) return(as.Date(x))
    out <- suppressWarnings(as.Date(x))
    if (all(is.na(out))) stop(sprintf("Variable %s cannot be converted to Date.", name))
    out
  }
  
  rmse_from_sums <- function(sw, sse) {
    if (!is.finite(sw) || sw <= 0) return(NA_real_)
    sqrt(sse / sw)
  }
  
  r2_from_sums <- function(sw, sy, sy2, sse) {
    if (!is.finite(sw) || sw <= 0) return(NA_real_)
    sst <- sy2 - sy^2 / sw
    if (!is.finite(sst) || sst <= 0) return(NA_real_)
    1 - sse / sst
  }
  
  weighted_regression_from_sums <- function(sw, sy, sp, sp2, syp) {
    if (!is.finite(sw) || sw <= 0) return(c(intercept = NA_real_, slope = NA_real_))
    denom <- sp2 - sp^2 / sw
    if (!is.finite(denom) || denom <= 0) return(c(intercept = NA_real_, slope = NA_real_))
    slope <- (syp - sy * sp / sw) / denom
    intercept <- sy / sw - slope * sp / sw
    c(intercept = intercept, slope = slope)
  }
  
  weighted_quantile <- function(x, probs, w = NULL) {
    ok <- is.finite(x)
    x <- x[ok]
    if (length(x) == 0L) return(rep(NA_real_, length(probs)))
    if (is.null(w)) {
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
    cumulative <- cumsum(w) / sum(w)
    vapply(probs, function(p) x[which(cumulative >= p)[1L]], numeric(1L))
  }
  
  softmax_by_set <- function(eta, set_index) {
    work <- data.table(index = seq_along(eta), set_index = set_index, eta = as.numeric(eta))
    work[, probability := {
      shifted <- eta - max(eta)
      exponentiated <- exp(shifted)
      exponentiated / sum(exponentiated)
    }, by = set_index]
    out <- numeric(length(eta))
    out[work$index] <- work$probability
    out
  }
  
  add_grouped_values <- function(target, group, values) {
    aggregated <- rowsum(values, group, reorder = FALSE)
    target[as.integer(rownames(aggregated))] <-
      target[as.integer(rownames(aggregated))] + aggregated[, 1L]
    target
  }
  
  matrix_condition_number <- function(M) {
    if (length(M) == 1L) return(if (is.finite(M) && abs(M) > 0) 1 else Inf)
    values <- tryCatch(svd(M, nu = 0L, nv = 0L)$d, error = function(e) NA_real_)
    if (any(!is.finite(values)) || min(values) <= 0) return(Inf)
    max(values) / min(values)
  }
  
  predict_xgb_chunks <- function(model, X, chunk_size = XGB_PREDICT_CHUNK_SIZE) {
    n <- nrow(X)
    if (n == 0L) return(numeric(0L))
    answer <- numeric(n)
    starts <- seq.int(1L, n, by = chunk_size)
    for (st in starts) {
      en <- min(n, st + chunk_size - 1L)
      dm <- xgboost::xgb.DMatrix(X[st:en, , drop = FALSE])
      answer[st:en] <- stats::predict(model, dm)
      rm(dm)
      gc(FALSE)
    }
    answer
  }
  
  make_group_folds <- function(group_id, K, seed) {
    unique_groups <- unique(as.integer(group_id))
    if (length(unique_groups) < 2L) stop("Too few matched sets are available for cross-fitting.")
    K <- min(max(2L, as.integer(K)), length(unique_groups))
    set.seed(seed)
    data.table(
      group_id = unique_groups,
      fold = sample(rep(seq_len(K), length.out = length(unique_groups)))
    )
  }
  
  make_grouped_cv_indices <- function(group_id, K, seed) {
    map <- make_group_folds(group_id, K, seed)
    row_fold <- map[data.table(group_id = as.integer(group_id)), on = "group_id", fold]
    lapply(sort(unique(row_fold)), function(k) which(row_fold == k))
  }
  
  flatten_tunes <- function(tunes) {
    if (length(tunes) == 0L) return(data.table())
    rows <- lapply(names(tunes), function(fold_name) {
      tune <- tunes[[fold_name]]
      if (is.null(tune)) return(NULL)
      out <- data.table(
        Fold = as.integer(fold_name),
        CV_RMSE = as.numeric(tune$rmse %||% NA_real_),
        Nrounds = as.integer(tune$nrounds %||% NA_integer_)
      )
      params <- tune$params %||% list()
      for (nm in names(params)) out[, (nm) := as.character(params[[nm]])]
      out
    })
    rbindlist(rows, fill = TRUE)
  }
  
  message("================================================================================================")
  message("Case-crossover generalized AIPW for recent extreme-temperature event history")
  message("================================================================================================")
  message(sprintf("Outcome:          %s", OUTCOME_NAME))
  message(sprintf("Event type:       %s", event_label))
  message(sprintf("Event definition: %s", EVENT_DEFINITION))
  message(sprintf("Input:            %s", DATA_RDS))
  message(sprintf("MMT:              %.4f", MMT))
  message(sprintf("Years:            %s", paste(ANALYSIS_YEARS, collapse = ",")))
  message(sprintf("Event lags:       lag%d-%d", min(EVENT_HISTORY_LAGS), max(EVENT_HISTORY_LAGS)))
  message(sprintf("Heat adjustment:  lag%d-%d", min(HOT_LAGS), max(HOT_LAGS)))
  message(sprintf("Cold adjustment:  lag%d-%d", min(COLD_LAGS), max(COLD_LAGS)))
  message(sprintf("Outer folds:      %d | Bootstrap replicates: %d", K_fold, BOOT_B))
  
  # ==============================================================================================
  # 1. Event-calendar construction and compact sampled-day data preparation
  # ==============================================================================================
  
  auto_detect_event_lag_columns <- function(column_names) {
    lags <- EVENT_HISTORY_LAGS
    patterns <- list(
      paste0(EVENT_DEFINITION, "_lag", lags),
      paste0(EVENT_DEFINITION, "_L", lags),
      paste0(EVENT_DEFINITION, ".lag", lags),
      paste0(EVENT_DEFINITION, "_lag_", lags)
    )
    for (candidate in patterns) {
      if (all(candidate %in% column_names)) return(candidate)
    }
    NULL
  }
  
  validate_binary_event_columns <- function(raw, columns) {
    for (nm in columns) {
      raw[, (nm) := numericize(get(nm), nm)]
      values <- unique(raw[[nm]][is.finite(raw[[nm]])])
      if (!all(values %in% c(0, 1))) {
        stop(sprintf("Event-history column %s must contain only 0 and 1.", nm))
      }
    }
    invisible(NULL)
  }
  
  build_rowwise_event_history <- function(raw_analysis) {
    if (is.null(THRESHOLD_PERCENTILE) || is.null(MIN_DURATION_DAYS)) {
      stop(
        "THRESHOLD_PERCENTILE and MIN_DURATION_DAYS are required when event history is reconstructed."
      )
    }
    if (is.null(raw_analysis)) stop("No analysis data are available for row-wise event reconstruction.")
    if (!CITY_COL %in% names(raw_analysis)) {
      stop(sprintf("CITY_COL (%s) is required for row-wise event reconstruction.", CITY_COL))
    }
    if (all(is.na(raw_analysis$date_internal))) {
      stop("DATE_COL is required for row-wise event reconstruction and seasonal duration checks.")
    }
    
    max_event_temperature_lag <-
      max(EVENT_HISTORY_LAGS) + as.integer(MIN_DURATION_DAYS) - 1L
    event_temperature_cols <- sprintf("temp_lag%d", 0:max_event_temperature_lag)
    missing_event_temperature_cols <- setdiff(event_temperature_cols, names(raw_analysis))
    if (length(missing_event_temperature_cols) > 0L) {
      stop(sprintf(
        "Row-wise event reconstruction requires the following temperature columns: %s",
        paste(missing_event_temperature_cols, collapse = ", ")
      ))
    }
    
    n_rows <- nrow(raw_analysis)
    work <- data.table(
      row_source_index = seq_len(n_rows),
      city_internal = trimws(as.character(raw_analysis[[CITY_COL]])),
      date_internal = raw_analysis$date_internal,
      year_internal = raw_analysis$row_year_internal
    )
    
    if (any(is.na(work$city_internal) | !nzchar(work$city_internal))) {
      stop("CITY_COL contains missing or empty values required for event-threshold assignment.")
    }
    
    # This reproduces the threshold construction used in the definition-screening analysis:
    # unique residentially assigned city-date-temperature records are retained, so a city-date is
    # not forced to have a single temperature when different residential grids have different values.
    threshold_reference <- unique(data.table(
      city_internal = work$city_internal,
      date_internal = work$date_internal,
      year_internal = work$year_internal,
      temperature_internal = raw_analysis[["temp_lag0"]]
    ))
    threshold_reference <- threshold_reference[
      !is.na(city_internal) & nzchar(city_internal) &
        !is.na(date_internal) & is.finite(temperature_internal) &
        year_internal %in% EVENT_THRESHOLD_REFERENCE_YEARS
    ]
    if (nrow(threshold_reference) == 0L) {
      stop("No valid city-specific temperature records remain for threshold estimation.")
    }
    
    # The primary screening analysis used year-round city-specific percentiles. A season-specific
    # reference distribution can be requested explicitly for sensitivity analysis.
    if (THRESHOLD_BASE == "season_specific") {
      threshold_reference[, reference_month := as.integer(format(date_internal, "%m"))]
      threshold_reference <- threshold_reference[reference_month %in% event_months]
      threshold_reference[, reference_month := NULL]
      if (nrow(threshold_reference) == 0L) {
        stop("No valid temperature records remain after season-specific threshold restriction.")
      }
    }
    
    threshold_table <- threshold_reference[, .(
      threshold_value = as.numeric(stats::quantile(
        temperature_internal,
        probs = THRESHOLD_PERCENTILE,
        names = FALSE,
        na.rm = TRUE,
        type = 7L
      )),
      n_reference_records = .N,
      n_reference_city_dates = uniqueN(date_internal)
    ), by = city_internal]
    
    work <- threshold_table[work, on = "city_internal"]
    setorder(work, row_source_index)
    if (any(!is.finite(work$threshold_value))) {
      stop("At least one sampled city has no valid event threshold.")
    }
    
    recent_event <- integer(n_rows)
    event_lag0 <- rep(NA_integer_, n_rows)
    complete_event_history <- rep(TRUE, n_rows)
    
    for (event_lag in EVENT_HISTORY_LAGS) {
      qualifying_window <- rep(TRUE, n_rows)
      valid_window <- is.finite(work$threshold_value) & !is.na(work$date_internal)
      
      for (duration_offset in 0:(as.integer(MIN_DURATION_DAYS) - 1L)) {
        lag_index <- event_lag + duration_offset
        temperature_values <- raw_analysis[[sprintf("temp_lag%d", lag_index)]]
        valid_window <- valid_window & is.finite(temperature_values)
        
        if (event_direction == "upper") {
          qualifying_window <- qualifying_window &
            temperature_values >= work$threshold_value
        } else {
          qualifying_window <- qualifying_window &
            temperature_values <= work$threshold_value
        }
        
        lag_date <- work$date_internal - lag_index
        lag_month <- as.integer(format(lag_date, "%m"))
        valid_window <- valid_window & !is.na(lag_month)
        qualifying_window <- qualifying_window & lag_month %in% event_months
      }
      
      event_indicator <- as.integer(qualifying_window)
      event_indicator[!valid_window] <- NA_integer_
      
      if (event_lag == 0L) event_lag0 <- event_indicator
      recent_event[!is.na(event_indicator) & event_indicator == 1L] <- 1L
      complete_event_history <- complete_event_history & !is.na(event_indicator)
    }
    
    # This matches rowSums(event_history, na.rm = FALSE) in the screening code: any missing
    # lag-specific event indicator makes the recent-history summary missing for that sampled day.
    recent_event[!complete_event_history] <- NA_integer_
    
    qualifying_event_days <- unique(work[
      !is.na(event_lag0) & event_lag0 == 1L & year_internal %in% ANALYSIS_YEARS,
      .(city_internal, date_internal)
    ])
    
    sampled_city_dates <- unique(work[
      year_internal %in% ANALYSIS_YEARS & !is.na(date_internal),
      .(city_internal, date_internal)
    ])
    
    support <- data.table(
      Event_calendar_source =
        "Row-wise reconstruction from residentially assigned lagged temperatures",
      Number_of_cities = uniqueN(work$city_internal),
      Number_of_calendar_days = nrow(sampled_city_dates),
      Number_of_reference_days = nrow(threshold_reference),
      Number_of_distinct_event_episodes = NA_integer_,
      Number_of_qualifying_event_days = nrow(qualifying_event_days),
      Threshold_percentile = THRESHOLD_PERCENTILE,
      Minimum_duration_days = MIN_DURATION_DAYS,
      Maximum_temperature_lag_required = max_event_temperature_lag,
      Threshold_base = if (THRESHOLD_BASE == "year_round") {
        "Year-round city-specific distribution of residentially assigned daily mean temperature"
      } else {
        "Season-specific city-specific distribution of residentially assigned daily mean temperature"
      }
    )
    
    rm(
      work,
      threshold_reference,
      threshold_table,
      qualifying_event_days,
      sampled_city_dates
    )
    gc(FALSE)
    
    list(
      E_event = recent_event,
      event_lag0 = event_lag0,
      support = support,
      episodes = data.table()
    )
  }
  
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
    
    hot_temperature_cols <- sprintf("temp_lag%d", HOT_LAGS)
    cold_temperature_cols <- sprintf("temp_lag%d", COLD_LAGS)
    rh_cols <- sprintf("rh_lag%d", RH_LAGS)
    
    detected_event_cols <- EVENT_LAG_COLS
    if (is.null(detected_event_cols)) {
      detected_event_cols <- auto_detect_event_lag_columns(names(raw))
    }
    
    event_temperature_cols <- character(0L)
    if (is.null(detected_event_cols)) {
      if (is.null(THRESHOLD_PERCENTILE) || is.null(MIN_DURATION_DAYS)) {
        stop(
          paste0(
            "Event-history columns were not found. THRESHOLD_PERCENTILE and ",
            "MIN_DURATION_DAYS are required for row-wise event reconstruction."
          )
        )
      }
      max_event_temperature_lag <-
        max(EVENT_HISTORY_LAGS) + as.integer(MIN_DURATION_DAYS) - 1L
      event_temperature_cols <- sprintf("temp_lag%d", 0:max_event_temperature_lag)
    }
    
    required_lag <- unique(c(
      hot_temperature_cols,
      cold_temperature_cols,
      rh_cols,
      event_temperature_cols
    ))
    missing_lag <- setdiff(required_lag, names(raw))
    if (length(missing_lag) > 0L) {
      stop(sprintf("Missing required lag columns: %s", paste(missing_lag, collapse = ", ")))
    }
    
    raw[, set_id_original := as.character(get(ID_COL))]
    raw[, case_internal := numericize(get(CASE_COL), CASE_COL)]
    if (!all(raw$case_internal %in% c(0, 1))) stop("The case indicator must contain only 0 and 1.")
    
    for (nm in required_lag) raw[, (nm) := numericize(get(nm), nm)]
    
    date_name <- DATE_COL
    if (!is.null(date_name) && date_name %in% names(raw)) {
      raw[, date_internal := coerce_to_date(get(date_name), date_name)]
    } else {
      raw[, date_internal := as.Date(NA)]
    }
    
    if (!is.null(YEAR_COL) && YEAR_COL %in% names(raw)) {
      raw[, row_year_internal := as.integer(numericize(get(YEAR_COL), YEAR_COL))]
    } else if (any(!is.na(raw$date_internal))) {
      raw[, row_year_internal := as.integer(format(date_internal, "%Y"))]
    } else {
      stop("No usable year column was found. Supply YEAR_COL or DATE_COL.")
    }
    
    if (!is.null(MONTH_COL) && MONTH_COL %in% names(raw)) {
      raw[, row_month_internal := as.integer(numericize(get(MONTH_COL), MONTH_COL))]
    } else if (any(!is.na(raw$date_internal))) {
      raw[, row_month_internal := as.integer(format(date_internal, "%m"))]
    } else {
      stop("No usable month column was found. Supply MONTH_COL or DATE_COL.")
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
    
    raw[, T_hot_internal := rowMeans(.SD), .SDcols = hot_temperature_cols]
    raw[, T_cold_internal := rowMeans(.SD), .SDcols = cold_temperature_cols]
    raw[, A_hot_internal := pmax(T_hot_internal - MMT, 0)]
    raw[, A_cold_internal := pmax(MMT - T_cold_internal, 0)]
    raw[, rh_summary_internal := rowMeans(.SD), .SDcols = rh_cols]
    
    # detected_event_cols was determined before lag-column validation so that additional temperature
    # lags are required only when event history must be reconstructed row by row.
    
    event_calendar_support <- data.table(
      Event_calendar_source = NA_character_,
      Number_of_cities = NA_integer_,
      Number_of_calendar_days = NA_integer_,
      Number_of_reference_days = NA_integer_,
      Number_of_distinct_event_episodes = NA_integer_,
      Number_of_qualifying_event_days = NA_integer_,
      Threshold_percentile = THRESHOLD_PERCENTILE %||% NA_real_,
      Minimum_duration_days = MIN_DURATION_DAYS %||% NA_integer_
    )
    event_episode_table <- data.table()
    
    if (!is.null(detected_event_cols)) {
      missing_event_cols <- setdiff(detected_event_cols, names(raw))
      if (length(missing_event_cols) > 0L) {
        stop(sprintf("Missing event-history columns: %s", paste(missing_event_cols, collapse = ", ")))
      }
      validate_binary_event_columns(raw, detected_event_cols)
      raw[, E_event := as.integer(rowSums(.SD, na.rm = FALSE) > 0), .SDcols = detected_event_cols]
      event_calendar_support[, Event_calendar_source :=
                               "Existing lag-specific event indicators; calendar support is limited to observed city-date records"]
      
      # Support summaries are calculated from observed city-date records only. Different residential
      # grids within the same city-date may legitimately have different event indicators; these are
      # retained in the analysis and are not treated as data conflicts.
      if (CITY_COL %in% names(raw) && any(!is.na(raw$date_internal))) {
        event_day_records <- data.table(
          city_internal = trimws(as.character(raw[[CITY_COL]])),
          date_internal = raw$date_internal,
          event_day = as.integer(raw[[detected_event_cols[1L]]])
        )
        event_day_records <- event_day_records[
          !is.na(city_internal) & nzchar(city_internal) & !is.na(date_internal)
        ]
        
        event_day_unique <- event_day_records[, {
          finite_event <- event_day[is.finite(event_day)]
          list(
            event_day = if (length(finite_event) > 0L) {
              as.integer(any(finite_event == 1L))
            } else {
              NA_integer_
            },
            n_spatial_event_values = uniqueN(finite_event)
          )
        }, by = .(city_internal, date_internal)]
        
        event_calendar_support[, `:=`(
          Number_of_cities = uniqueN(event_day_unique$city_internal),
          Number_of_calendar_days = nrow(event_day_unique),
          Number_of_distinct_event_episodes = NA_integer_,
          Number_of_qualifying_event_days = sum(event_day_unique$event_day == 1L, na.rm = TRUE),
          Number_of_city_dates_with_spatially_varying_event_status =
            sum(event_day_unique$n_spatial_event_values > 1L, na.rm = TRUE)
        )]
        
        rm(event_day_records, event_day_unique)
        gc(FALSE)
      }
    } else {
      rowwise_event <- build_rowwise_event_history(raw_analysis = raw)
      raw[, E_event := rowwise_event$E_event]
      event_calendar_support <- rowwise_event$support
      event_episode_table <- rowwise_event$episodes
      rm(rowwise_event)
      gc(FALSE)
    }
    
    if (any(!is.na(raw$E_event) & !(raw$E_event %in% c(0, 1)))) {
      stop("Constructed recent event history contains values other than 0 and 1.")
    }
    known_event_days <- event_calendar_support$Number_of_qualifying_event_days[1L]
    support_is_complete <- !grepl(
      "limited to observed city-date records",
      event_calendar_support$Event_calendar_source[1L],
      fixed = TRUE
    )
    if (support_is_complete && is.finite(known_event_days) && known_event_days < MIN_QUALIFYING_EVENT_DAYS) {
      stop(sprintf(
        "The fixed event definition has only %d qualifying event days; at least %d are required.",
        as.integer(known_event_days), as.integer(MIN_QUALIFYING_EVENT_DAYS)
      ))
    }
    
    complete_vars <- unique(c(
      "case_internal", "row_year_internal", "row_month_internal", "E_event",
      "A_hot_internal", "A_cold_internal", "rh_summary_internal", "holiday_internal",
      OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
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
      },
      case_month = if (sum(case_internal == 1) == 1L) {
        row_month_internal[case_internal == 1][1L]
      } else {
        NA_integer_
      },
      case_event = if (sum(case_internal == 1) == 1L) {
        E_event[case_internal == 1][1L]
      } else {
        NA_integer_
      },
      event_variation = uniqueN(E_event) > 1L
    ), by = set_id_original]
    
    structurally_valid_sets <- set_check[
      n_cases == 1L & n_rows >= 2L & case_year %in% ANALYSIS_YEARS & case_month %in% 1:12,
      .(set_id_original, case_year, case_month, case_event, event_variation, all_complete)
    ]
    
    n_dropped_invalid <- nrow(set_check) - nrow(structurally_valid_sets)
    if (nrow(structurally_valid_sets) < max(10L, K_fold * 2L)) {
      stop(sprintf("Too few structurally valid full-year matched sets remain: %d.", nrow(structurally_valid_sets)))
    }
    
    # Full-year denominators include all structurally valid deaths. Outside-season deaths contribute
    # zero event-history AF and therefore do not require complete seasonal nuisance-model covariates.
    full_year_counts <- structurally_valid_sets[, .(N_full_year_deaths = .N), by = .(Year = case_year)]
    full_year_grid <- data.table(Year = as.integer(ANALYSIS_YEARS))
    full_year_counts <- full_year_counts[full_year_grid, on = "Year"]
    full_year_counts[is.na(N_full_year_deaths), N_full_year_deaths := 0L]
    
    full_year_case_map <- structurally_valid_sets[, .(
      set_id_original,
      case_year,
      case_month
    )]
    setorder(full_year_case_map, case_year, set_id_original)
    
    valid_season_sets <- structurally_valid_sets[
      case_month %in% event_months & all_complete,
      .(set_id_original, case_year, case_month, case_event, event_variation)
    ]
    n_dropped_incomplete_season <- structurally_valid_sets[
      case_month %in% event_months & !all_complete,
      .N
    ]
    if (n_dropped_incomplete_season > 0L) {
      stop(sprintf(
        paste0(
          "%d event-season matched sets have incomplete analysis variables. ",
          "They cannot be assigned zero event burden in the full-year denominator; correct the input data first."
        ),
        n_dropped_incomplete_season
      ))
    }
    if (nrow(valid_season_sets) < max(10L, K_fold * 2L)) {
      stop(sprintf("Too few complete event-season matched sets remain: %d.", nrow(valid_season_sets)))
    }
    if (!any(valid_season_sets$event_variation)) {
      stop("Recent event history has no within-set variation in the seasonal matched sets.")
    }
    
    # Repeatedly randomize year-stratified folds until every outer training sample contains at least
    # one matched set with within-set event-history variation.
    fold_assignment_success <- FALSE
    for (attempt in seq_len(100L)) {
      set.seed(ANALYSIS_SEED + attempt)
      candidate <- copy(valid_season_sets)
      candidate[, fold := sample(
        rep(seq_len(min(K_fold, .N)), length.out = .N)
      ), by = case_year]
      fold_values <- sort(unique(candidate$fold))
      training_supported <- vapply(
        fold_values,
        function(k) any(candidate[fold != k, event_variation]),
        logical(1L)
      )
      heldout_nonempty <- vapply(
        fold_values,
        function(k) candidate[fold == k, .N] > 0L,
        logical(1L)
      )
      if (all(training_supported) && all(heldout_nonempty) && length(fold_values) >= 2L) {
        valid_season_sets <- candidate
        fold_assignment_success <- TRUE
        break
      }
    }
    if (!fold_assignment_success) {
      stop("A valid outer-fold allocation with event-history support could not be generated.")
    }
    valid_season_sets[, set_index := .I]
    
    compact_names <- unique(c(
      "set_id_original", "case_internal", "E_event", "A_hot_internal", "A_cold_internal",
      "rh_summary_internal", "holiday_internal", OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
    ))
    raw_compact <- raw[, ..compact_names]
    rm(raw)
    gc(FALSE)
    
    joined <- valid_season_sets[
      raw_compact,
      on = "set_id_original",
      nomatch = 0L,
      allow.cartesian = TRUE
    ]
    
    row_names <- unique(c(
      "set_index", "case_internal", "case_year", "case_month", "case_event",
      "event_variation", "fold", "E_event", "A_hot_internal", "A_cold_internal",
      "rh_summary_internal", "holiday_internal", OUTCOME_EXTRA_VARS, EXPOSURE_EXTRA_VARS
    ))
    rows <- joined[, ..row_names]
    setorder(rows, set_index, -case_internal)
    rows[, row_uid := .I]
    
    set_map <- valid_season_sets[, .(
      set_index,
      set_id_original,
      case_year,
      case_month,
      case_event,
      event_variation,
      fold
    )]
    setorder(set_map, set_index)
    
    rm(raw_compact, joined, set_check, structurally_valid_sets, valid_season_sets)
    gc(FALSE)
    
    message(sprintf(
      paste0(
        "Prepared compact seasonal data: %d matched sets and %d sampled-day rows; ",
        "dropped %d structurally invalid or out-of-period sets and %d incomplete seasonal sets."
      ),
      nrow(set_map), nrow(rows), n_dropped_invalid, n_dropped_incomplete_season
    ))
    toc(t0, "Data preparation completed")
    
    list(
      rows = rows,
      set_map = set_map,
      full_year_counts = full_year_counts,
      full_year_case_map = full_year_case_map,
      event_calendar_support = event_calendar_support,
      event_episode_table = event_episode_table,
      detected_event_columns = detected_event_cols
    )
  }
  
  prepared <- prepare_data(DATA_RDS)
  data_main <- prepared$rows
  set_map_main <- prepared$set_map
  full_year_counts_main <- prepared$full_year_counts
  full_year_case_map_main <- prepared$full_year_case_map
  event_calendar_support_main <- prepared$event_calendar_support
  event_episode_table_main <- prepared$event_episode_table
  detected_event_columns_main <- prepared$detected_event_columns
  rm(prepared)
  gc(FALSE)
  
  # ==============================================================================================
  # 2. Cross-fitted conditional logistic outcome nuisance model
  # ==============================================================================================
  
  derive_natural_spline_spec <- function(x, requested_df, label) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) stop(sprintf("No finite %s values are available in an outcome training fold.", label))
    use_spline <- length(unique(x)) >= max(5L, requested_df + 1L) && requested_df >= 2L
    if (use_spline) {
      n_internal <- max(0L, requested_df - 1L)
      boundary <- range(x)
      if (n_internal > 0L) {
        probabilities <- seq(0, 1, length.out = n_internal + 2L)[-c(1L, n_internal + 2L)]
        knots <- unique(as.numeric(stats::quantile(x, probs = probabilities, names = FALSE, type = 7L)))
        knots <- knots[knots > boundary[1L] & knots < boundary[2L]]
      } else {
        knots <- numeric(0L)
      }
      list(
        type = "natural_spline",
        knots = knots,
        boundary_knots = boundary,
        requested_df = as.integer(requested_df)
      )
    } else {
      list(
        type = "linear",
        knots = numeric(0L),
        boundary_knots = range(x),
        requested_df = 1L
      )
    }
  }
  
  evaluate_spline_spec <- function(x, spec, prefix) {
    if (identical(spec$type, "natural_spline")) {
      out <- splines::ns(
        x,
        knots = spec$knots,
        Boundary.knots = spec$boundary_knots,
        intercept = FALSE
      )
      colnames(out) <- paste0(prefix, "_ns", seq_len(ncol(out)))
    } else {
      out <- matrix(x, ncol = 1L)
      colnames(out) <- paste0(prefix, "_linear")
    }
    out
  }
  
  derive_outcome_basis_spec <- function(rows, train_idx) {
    list(
      heat_adjustment = derive_natural_spline_spec(
        rows$A_hot_internal[train_idx], HOT_ADJUSTMENT_DF, "heat-side non-optimal-temperature deviation"
      ),
      cold_adjustment = derive_natural_spline_spec(
        rows$A_cold_internal[train_idx], COLD_ADJUSTMENT_DF, "cold-side non-optimal-temperature deviation"
      ),
      humidity = derive_natural_spline_spec(
        rows$rh_summary_internal[train_idx], RH_SPLINE_DF, "relative humidity"
      ),
      active_columns = NULL
    )
  }
  
  build_outcome_basis <- function(rows, idx, spec, determine_active = FALSE) {
    heat_adjustment_basis <- evaluate_spline_spec(
      rows$A_hot_internal[idx], spec$heat_adjustment, "A_hot_adjustment"
    )
    cold_adjustment_basis <- evaluate_spline_spec(
      rows$A_cold_internal[idx], spec$cold_adjustment, "A_cold_adjustment"
    )
    humidity_basis <- evaluate_spline_spec(
      rows$rh_summary_internal[idx], spec$humidity, "humidity"
    )
    
    pieces <- list(
      heat_adjustment_basis,
      cold_adjustment_basis,
      humidity_basis,
      matrix(rows$holiday_internal[idx], ncol = 1L)
    )
    column_names <- c(
      colnames(heat_adjustment_basis),
      colnames(cold_adjustment_basis),
      colnames(humidity_basis),
      "holiday_internal"
    )
    
    if (length(OUTCOME_EXTRA_VARS) > 0L) {
      extra_matrix <- do.call(cbind, lapply(OUTCOME_EXTRA_VARS, function(nm) rows[[nm]][idx]))
      if (is.null(dim(extra_matrix))) extra_matrix <- matrix(extra_matrix, ncol = 1L)
      colnames(extra_matrix) <- OUTCOME_EXTRA_VARS
      pieces[[length(pieces) + 1L]] <- extra_matrix
      column_names <- c(column_names, OUTCOME_EXTRA_VARS)
    }
    
    basis <- do.call(cbind, pieces)
    colnames(basis) <- column_names
    storage.mode(basis) <- "double"
    basis[!is.finite(basis)] <- 0
    
    if (determine_active || is.null(spec$active_columns)) {
      active <- vapply(seq_len(ncol(basis)), function(j) {
        value <- basis[, j]
        length(value) > 1L && is.finite(stats::sd(value)) && stats::sd(value) > 0
      }, logical(1L))
      spec$active_columns <- colnames(basis)[active]
    }
    
    active_columns <- intersect(spec$active_columns %||% character(0L), colnames(basis))
    if (length(active_columns) == 0L) {
      basis <- matrix(numeric(0L), nrow = length(idx), ncol = 0L)
    } else {
      basis <- basis[, active_columns, drop = FALSE]
    }
    
    list(matrix = basis, spec = spec)
  }
  
  compact_clogit_fit <- function(fit, nuisance_names, fold_id) {
    coefficients <- stats::coef(fit)
    if (!"E_event" %in% names(coefficients)) {
      stop(sprintf("Outcome fold %s did not return the event-history coefficient.", fold_id))
    }
    if (!is.finite(coefficients["E_event"])) {
      stop(sprintf("Outcome fold %s returned a non-finite event-history coefficient.", fold_id))
    }
    
    gamma <- setNames(rep(0, length(nuisance_names)), nuisance_names)
    common <- intersect(nuisance_names, names(coefficients))
    if (length(common) > 0L) {
      if (any(!is.finite(coefficients[common]))) {
        failed_terms <- common[!is.finite(coefficients[common])]
        stop(sprintf(
          "Outcome fold %s returned non-estimable nuisance coefficients: %s",
          fold_id, paste(failed_terms, collapse = ", ")
        ))
      }
      gamma[common] <- coefficients[common]
    }
    
    covariance <- tryCatch(stats::vcov(fit), error = function(e) NULL)
    coefficient_se <- setNames(rep(NA_real_, length(coefficients)), names(coefficients))
    if (!is.null(covariance) && nrow(covariance) == length(coefficients)) {
      coefficient_se <- sqrt(pmax(diag(covariance), 0))
      names(coefficient_se) <- names(coefficients)
    }
    
    information_condition_number <- if (!is.null(covariance) && all(is.finite(covariance))) {
      matrix_condition_number(covariance)
    } else {
      NA_real_
    }
    
    fail_text <- as.character(fit$fail %||% "")
    converged <- length(fail_text) == 0L || !nzchar(fail_text[1L])
    
    list(
      beta = as.numeric(coefficients["E_event"]),
      gamma = gamma,
      coefficient_se = coefficient_se,
      loglik = if (!is.null(fit$loglik)) as.numeric(tail(fit$loglik, 1L)) else NA_real_,
      iterations = as.integer((fit$iter %||% NA_integer_)[1L]),
      information_condition_number = information_condition_number,
      maximum_absolute_coefficient = max(abs(coefficients), na.rm = TRUE),
      maximum_coefficient_se = if (all(is.na(coefficient_se))) NA_real_ else max(coefficient_se, na.rm = TRUE),
      non_estimable_nuisance_coefficients = 0L,
      converged = converged,
      coefficient_table = data.table(
        Term = names(coefficients),
        Estimate = as.numeric(coefficients),
        Standard_error = as.numeric(coefficient_se[names(coefficients)])
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
      E_event = rows$E_event[train_idx],
      check.names = FALSE
    )
    if (ncol(Btr) > 0L) {
      for (j in seq_len(ncol(Btr))) fit_df[[colnames(Btr)[j]]] <- Btr[, j]
    }
    rm(Btr, basis_train)
    gc(FALSE)
    
    rhs <- c("E_event", nuisance_names, "strata(set_index)")
    formula <- stats::as.formula(paste("case_internal ~", paste(rhs, collapse = " + ")))
    
    fit <- survival::clogit(
      formula = formula,
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
      beta_event_outcome = compact$beta,
      SE_beta_event_outcome = as.numeric(compact$coefficient_se["E_event"]),
      RR_event_outcome = exp(compact$beta),
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
      eta_test <- compact$beta * rows$E_event[test_idx] + nuisance_lp_test
      set_test <- rows$set_index[test_idx]
      y_test <- rows$case_internal[test_idx]
      p_test <- softmax_by_set(eta_test, set_test)
      
      if (any(!is.finite(p_test)) || any(p_test < 0) || any(p_test > 1)) {
        stop(sprintf("Outcome fold %s produced invalid conditional probabilities.", fold_id))
      }
      
      set_lengths <- rle(set_test)$lengths
      probability_sums <- rowsum(p_test, set_test, reorder = FALSE)[, 1L]
      maximum_sum_error <- max(abs(probability_sums - 1), na.rm = TRUE)
      case_probability <- p_test[y_test == 1]
      if (length(case_probability) != length(set_lengths)) {
        stop(sprintf("Outcome fold %s has an invalid number of case-day probabilities.", fold_id))
      }
      
      probability_quantiles <- as.numeric(stats::quantile(
        p_test,
        probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
        names = FALSE,
        na.rm = TRUE
      ))
      case_probability_quantiles <- as.numeric(stats::quantile(
        case_probability,
        probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
        names = FALSE,
        na.rm = TRUE
      ))
      
      heldout_nll <- -mean(log(pmax(case_probability, .Machine$double.xmin)))
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
      probability_group <- if (length(cuts) >= 3L) {
        findInterval(p_test, cuts, all.inside = TRUE)
      } else {
        rep(1L, length(p_test))
      }
      calibration_work <- data.table(
        group = probability_group,
        observed = y_test,
        predicted = p_test,
        weight = row_set_weight
      )
      calibration <- calibration_work[, .(
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
        Proportion_case_probability_lt_0_01 = mean(case_probability < 0.01),
        Proportion_case_probability_lt_0_05 = mean(case_probability < 0.05),
        Max_set_probability_sum_error = maximum_sum_error,
        Heldout_residual_score_mean_abs = residual_mean_abs,
        Heldout_residual_score_max_abs = residual_max_abs
      )]
      
      rm(
        eta_test, set_test, y_test, p_test, set_lengths, probability_sums,
        case_probability, squared_error, set_brier, row_set_weight, cuts,
        probability_group, calibration_work, residual
      )
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
    matrix_basis <- basis$matrix
    nuisance_names <- colnames(matrix_basis) %||% character(0L)
    
    fit_df <- data.frame(
      case_internal = rows$case_internal,
      set_index = rows$set_index,
      E_event = rows$E_event,
      check.names = FALSE
    )
    if (ncol(matrix_basis) > 0L) {
      for (j in seq_len(ncol(matrix_basis))) fit_df[[colnames(matrix_basis)[j]]] <- matrix_basis[, j]
    }
    rm(matrix_basis, basis)
    gc(FALSE)
    
    rhs <- c("E_event", nuisance_names, "strata(set_index)")
    formula <- stats::as.formula(paste("case_internal ~", paste(rhs, collapse = " + ")))
    fit <- survival::clogit(
      formula = formula,
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
      
      if (!any(rows$event_variation[train_idx])) {
        stop(sprintf("Outcome outer training fold %d has no within-set event-history variation.", k))
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
        beta_event = one$beta
      )
      
      if (!quiet) {
        if (collect_diagnostics) {
          message(sprintf(
            "Outcome fold %d/%d: beta_event=%.6f, held-out NLL=%.4f, Brier=%.6f",
            ii, length(folds), one$beta,
            one$diagnostics$Heldout_negative_loglik_per_set,
            one$diagnostics$Heldout_set_weighted_Brier
          ))
        } else {
          message(sprintf("Outcome fold %d/%d: beta_event=%.6f", ii, length(folds), one$beta))
        }
      }
      
      rm(one, train_idx, test_idx)
      gc(FALSE)
    }
    
    if (any(!is.finite(nuisance_oof))) {
      stop("At least one outcome nuisance out-of-fold prediction is missing or non-finite.")
    }
    
    beta_table <- rbindlist(fold_betas)
    beta_start <- stats::weighted.mean(beta_table$beta_event, beta_table$N_test_sets)
    
    beta_stability_summary <- data.table(
      Parameter = "Event history",
      Mean = mean(beta_table$beta_event),
      Standard_deviation = stats::sd(beta_table$beta_event),
      Minimum = min(beta_table$beta_event),
      Maximum = max(beta_table$beta_event)
    )
    
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
      case_year = as.integer(rows$case_year[control_rows]),
      control_row = as.integer(control_rows),
      case_row = as.integer(paired_case_rows),
      M_s = as.integer(controls_per_set[control_set]),
      Z_event_0 = rows$E_event[control_rows] - rows$E_event[paired_case_rows],
      g_0 = nuisance_oof[control_rows] - nuisance_oof[paired_case_rows]
    )
    
    if (any(!is.finite(pairs$Z_event_0)) || any(!is.finite(pairs$g_0))) {
      stop("The ordered pseudo-pair contrasts contain non-finite values.")
    }
    if (!all(pairs$Z_event_0 %in% c(-1, 0, 1))) {
      stop("The pairwise event-history contrast must lie in {-1, 0, 1}.")
    }
    
    context_names <- unique(c(
      "A_hot_internal", "A_cold_internal", "rh_summary_internal", "holiday_internal",
      EXPOSURE_EXTRA_VARS
    ))
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
    
    column_position <- 0L
    for (nm in context_names) {
      first <- rows[[nm]][first_rows]
      second <- rows[[nm]][second_rows]
      X[, column_position + 1L] <- first
      X[, column_position + 2L] <- second
      X[, column_position + 3L] <- first - second
      X[, column_position + 4L] <- (first + second) / 2
      column_position <- column_position + 4L
    }
    X[, ncol(X)] <- pairs$M_s[idx]
    X[!is.finite(X)] <- 0
    storage.mode(X) <- "double"
    X
  }
  
  # ==============================================================================================
  # 4. Cross-fitted XGBoost retrospective exposure conditional-mean model
  # ==============================================================================================
  
  tune_xgb_regression <- function(rows, pair_index, outer_train_idx, seed, tag) {
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
    y_tune <- pairs$Z_event_0[tune_idx]
    w_tune <- 1 / pairs$M_s[tune_idx]
    groups_tune <- pairs$set_index[tune_idx]
    
    dtrain <- xgboost::xgb.DMatrix(data = X_tune, label = y_tune, weight = w_tune)
    rm(X_tune)
    gc(FALSE)
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
    
    rm(dtrain, y_tune, w_tune, groups_tune)
    gc(FALSE)
    
    if (is.null(best)) stop(sprintf("%s did not produce a valid XGBoost tuning result.", tag))
    best$params$seed <- NULL
    best
  }
  
  initialize_metric_sums <- function() {
    c(sw = 0, sy = 0, sy2 = 0, sp = 0, sp2 = 0, syp = 0, sse = 0, sae = 0, sr = 0)
  }
  
  update_metric_sums <- function(sums, y, prediction, weight) {
    residual <- y - prediction
    sums["sw"] <- sums["sw"] + sum(weight)
    sums["sy"] <- sums["sy"] + sum(weight * y)
    sums["sy2"] <- sums["sy2"] + sum(weight * y^2)
    sums["sp"] <- sums["sp"] + sum(weight * prediction)
    sums["sp2"] <- sums["sp2"] + sum(weight * prediction^2)
    sums["syp"] <- sums["syp"] + sum(weight * y * prediction)
    sums["sse"] <- sums["sse"] + sum(weight * residual^2)
    sums["sae"] <- sums["sae"] + sum(weight * abs(residual))
    sums["sr"] <- sums["sr"] + sum(weight * residual)
    sums
  }
  
  metric_table_from_sums <- function(train_sums, test_sums, tune, fold, n_train, n_test) {
    calibration <- weighted_regression_from_sums(
      test_sums["sw"], test_sums["sy"], test_sums["sp"], test_sums["sp2"], test_sums["syp"]
    )
    rmse_train <- rmse_from_sums(train_sums["sw"], train_sums["sse"])
    rmse_test <- rmse_from_sums(test_sums["sw"], test_sums["sse"])
    r2_train <- r2_from_sums(train_sums["sw"], train_sums["sy"], train_sums["sy2"], train_sums["sse"])
    r2_test <- r2_from_sums(test_sums["sw"], test_sums["sy"], test_sums["sy2"], test_sums["sse"])
    
    data.table(
      Fold = as.integer(fold),
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
      Calibration_intercept_test = calibration["intercept"],
      Calibration_slope_test = calibration["slope"],
      CV_RMSE = as.numeric(tune$rmse %||% NA_real_),
      Nrounds = as.integer(tune$nrounds)
    )
  }
  
  weighted_correlation_from_sums <- function(sw, sr, sr2, sx, sx2, srx) {
    covariance <- srx - sr * sx / sw
    variance_r <- sr2 - sr^2 / sw
    variance_x <- sx2 - sx^2 / sw
    ifelse(variance_r > 0 & variance_x > 0, covariance / sqrt(variance_r * variance_x), NA_real_)
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
    
    residualized_reverse <- rep(NA_real_, n_pairs)
    base_D0_set <- numeric(n_sets)
    tunes <- list()
    fold_diagnostics <- list()
    calibration_rows <- list()
    residual_association_rows <- list()
    prediction_distribution_rows <- list()
    z_group_rows <- list()
    
    residual_moments <- c(sw = 0, sr = 0, sr2 = 0)
    
    for (fold_position in seq_along(folds)) {
      k <- folds[fold_position]
      train_idx <- which(pairs$fold != k)
      test_idx <- which(pairs$fold == k)
      if (length(train_idx) < 20L || length(test_idx) < 1L) {
        stop(sprintf("Exposure outer fold %d has insufficient training or held-out pairs.", k))
      }
      
      tune <- if (!is.null(fixed_tunes)) {
        fixed_tunes[[as.character(k)]]
      } else {
        tune_xgb_regression(
          rows = rows,
          pair_index = pair_index,
          outer_train_idx = train_idx,
          seed = run_seed + 10000L + k,
          tag = sprintf("Event exposure fold %d", k)
        )
      }
      if (is.null(tune)) stop(sprintf("Exposure outer fold %d has no fixed XGBoost settings.", k))
      tune$params$seed <- NULL
      tunes[[as.character(k)]] <- tune
      
      X_train <- build_pair_feature_matrix(rows, pair_index, train_idx, orientation = 0L)
      y_train <- pairs$Z_event_0[train_idx]
      w_train <- 1 / pairs$M_s[train_idx]
      dtrain <- xgboost::xgb.DMatrix(data = X_train, label = y_train, weight = w_train)
      rm(X_train)
      gc(FALSE)
      
      params <- tune$params
      params$seed <- NULL
      model_seed <- if (!is.null(fixed_tunes)) {
        ANALYSIS_SEED + 20000L + k
      } else {
        run_seed + 20000L + k
      }
      set.seed(model_seed)
      model <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = tune$nrounds,
        verbose = 0
      )
      prediction_train <- stats::predict(model, dtrain)
      train_sums <- update_metric_sums(initialize_metric_sums(), y_train, prediction_train, w_train)
      rm(dtrain, prediction_train, y_train, w_train)
      gc(FALSE)
      
      test_sums <- initialize_metric_sums()
      feature_names <- pair_index$feature_names
      association_accumulator <- data.table(
        Feature = feature_names,
        sw = 0, sr = 0, sr2 = 0, sx = 0, sx2 = 0, srx = 0
      )
      z_accumulator <- data.table(
        Observed_Z = c(-1L, 0L, 1L),
        sw = 0, sr = 0, sr2 = 0
      )
      
      prediction_sample_0 <- numeric(0L)
      prediction_sample_1 <- numeric(0L)
      prediction_sample_weight <- numeric(0L)
      diagnostic_positions <- if (collect_diagnostics) {
        set.seed(run_seed + 30000L + k)
        sort(sample(
          seq_along(test_idx),
          size = min(length(test_idx), DIAGNOSTIC_PREDICTION_SAMPLE),
          replace = FALSE
        ))
      } else {
        integer(0L)
      }
      prediction_min_0 <- Inf
      prediction_max_0 <- -Inf
      prediction_min_1 <- Inf
      prediction_max_1 <- -Inf
      prediction_outside_0 <- 0L
      prediction_outside_1 <- 0L
      prediction_outside_weight_0 <- 0
      prediction_outside_weight_1 <- 0
      prediction_weight_sum <- 0
      prediction_count <- 0L
      
      starts <- seq.int(1L, length(test_idx), by = XGB_PREDICT_CHUNK_SIZE)
      for (st in starts) {
        en <- min(length(test_idx), st + XGB_PREDICT_CHUNK_SIZE - 1L)
        idx <- test_idx[st:en]
        weight <- 1 / pairs$M_s[idx]
        pair_weight <- 1 / (2 * pairs$M_s[idx])
        observed_z <- pairs$Z_event_0[idx]
        
        X0 <- build_pair_feature_matrix(rows, pair_index, idx, orientation = 0L)
        prediction_0 <- predict_xgb_chunks(model, X0)
        residual_0 <- observed_z - prediction_0
        test_sums <- update_metric_sums(test_sums, observed_z, prediction_0, weight)
        
        base_D0_set <- add_grouped_values(
          base_D0_set,
          pairs$set_index[idx],
          pair_weight * (-(observed_z - prediction_0))
        )
        
        residual_moments["sw"] <- residual_moments["sw"] + sum(weight)
        residual_moments["sr"] <- residual_moments["sr"] + sum(weight * residual_0)
        residual_moments["sr2"] <- residual_moments["sr2"] + sum(weight * residual_0^2)
        
        if (collect_diagnostics) {
          for (j in seq_along(feature_names)) {
            x <- X0[, j]
            association_accumulator[j, `:=`(
              sw = sw + sum(weight),
              sr = sr + sum(weight * residual_0),
              sr2 = sr2 + sum(weight * residual_0^2),
              sx = sx + sum(weight * x),
              sx2 = sx2 + sum(weight * x^2),
              srx = srx + sum(weight * residual_0 * x)
            )]
          }
          
          z_chunk <- data.table(Observed_Z = observed_z, weight = weight, residual = residual_0)[, .(
            sw = sum(weight),
            sr = sum(weight * residual),
            sr2 = sum(weight * residual^2)
          ), by = Observed_Z]
          z_accumulator[z_chunk, on = "Observed_Z", `:=`(
            sw = sw + i.sw,
            sr = sr + i.sr,
            sr2 = sr2 + i.sr2
          )]
        }
        
        prediction_min_0 <- min(prediction_min_0, min(prediction_0))
        prediction_max_0 <- max(prediction_max_0, max(prediction_0))
        prediction_outside_0 <- prediction_outside_0 + sum(prediction_0 < -1 | prediction_0 > 1)
        prediction_outside_weight_0 <- prediction_outside_weight_0 + sum(
          weight * (prediction_0 < -1 | prediction_0 > 1)
        )
        
        if (collect_diagnostics) {
          selected_global <- diagnostic_positions[
            diagnostic_positions >= st & diagnostic_positions <= en
          ]
          if (length(selected_global) > 0L) {
            local_position <- selected_global - st + 1L
            prediction_sample_0 <- c(prediction_sample_0, prediction_0[local_position])
            prediction_sample_weight <- c(prediction_sample_weight, weight[local_position])
          }
        }
        
        rm(X0, prediction_0, residual_0)
        gc(FALSE)
        
        X1 <- build_pair_feature_matrix(rows, pair_index, idx, orientation = 1L)
        prediction_1 <- predict_xgb_chunks(model, X1)
        residualized_reverse[idx] <- -observed_z - prediction_1
        
        prediction_min_1 <- min(prediction_min_1, min(prediction_1))
        prediction_max_1 <- max(prediction_max_1, max(prediction_1))
        prediction_outside_1 <- prediction_outside_1 + sum(prediction_1 < -1 | prediction_1 > 1)
        prediction_outside_weight_1 <- prediction_outside_weight_1 + sum(
          weight * (prediction_1 < -1 | prediction_1 > 1)
        )
        prediction_weight_sum <- prediction_weight_sum + sum(weight)
        prediction_count <- prediction_count + length(idx)
        
        if (collect_diagnostics && length(selected_global) > 0L) {
          prediction_sample_1 <- c(prediction_sample_1, prediction_1[local_position])
        }
        
        rm(X1, prediction_1, observed_z, weight, pair_weight, idx)
        gc(FALSE)
      }
      
      fold_diagnostics[[as.character(k)]] <- metric_table_from_sums(
        train_sums, test_sums, tune, k, length(train_idx), length(test_idx)
      )
      
      if (collect_diagnostics) {
        calibration <- weighted_regression_from_sums(
          test_sums["sw"], test_sums["sy"], test_sums["sp"], test_sums["sp2"], test_sums["syp"]
        )
        calibration_rows[[as.character(k)]] <- data.table(
          Fold = k,
          Calibration_intercept = calibration["intercept"],
          Calibration_slope = calibration["slope"],
          Mean_observed = test_sums["sy"] / test_sums["sw"],
          Mean_predicted = test_sums["sp"] / test_sums["sw"]
        )
        
        association_accumulator[, `:=`(
          Fold = k,
          Residual_feature_correlation = weighted_correlation_from_sums(
            sw, sr, sr2, sx, sx2, srx
          )
        )]
        residual_association_rows[[as.character(k)]] <- association_accumulator[, .(
          Fold, Feature, Residual_feature_correlation
        )]
        
        z_accumulator[, `:=`(
          Fold = k,
          Residual_mean = ifelse(sw > 0, sr / sw, NA_real_),
          Residual_RMSE = ifelse(sw > 0, sqrt(sr2 / sw), NA_real_)
        )]
        z_group_rows[[as.character(k)]] <- z_accumulator[, .(
          Fold, Observed_Z, Residual_mean, Residual_RMSE, Total_weight = sw
        )]
        
        quantile_values <- function(x, w) {
          if (length(x) == 0L) return(rep(NA_real_, 5L))
          weighted_quantile(x, c(0.01, 0.05, 0.50, 0.95, 0.99), w = w)
        }
        q0 <- quantile_values(prediction_sample_0, prediction_sample_weight)
        q1 <- quantile_values(prediction_sample_1, prediction_sample_weight)
        prediction_distribution_rows[[as.character(k)]] <- rbindlist(list(
          data.table(
            Fold = k,
            Orientation = "D=0: control first, case second",
            N_predictions = prediction_count,
            Quantile_sample_size = length(prediction_sample_0),
            Minimum = prediction_min_0,
            P01 = q0[1L], P05 = q0[2L], Median = q0[3L], P95 = q0[4L], P99 = q0[5L],
            Maximum = prediction_max_0,
            Proportion_outside_minus1_plus1 = prediction_outside_0 / prediction_count,
            Set_weighted_proportion_outside_minus1_plus1 =
              prediction_outside_weight_0 / prediction_weight_sum
          ),
          data.table(
            Fold = k,
            Orientation = "D=1: case first, control second",
            N_predictions = prediction_count,
            Quantile_sample_size = length(prediction_sample_1),
            Minimum = prediction_min_1,
            P01 = q1[1L], P05 = q1[2L], Median = q1[3L], P95 = q1[4L], P99 = q1[5L],
            Maximum = prediction_max_1,
            Proportion_outside_minus1_plus1 = prediction_outside_1 / prediction_count,
            Set_weighted_proportion_outside_minus1_plus1 =
              prediction_outside_weight_1 / prediction_weight_sum
          )
        ))
      }
      
      if (!quiet) {
        message(sprintf(
          "Exposure fold %d/%d: held-out RMSE=%.4f, held-out R2=%.4f",
          fold_position, length(folds),
          fold_diagnostics[[as.character(k)]]$RMSE_test,
          fold_diagnostics[[as.character(k)]]$R2_test
        ))
      }
      
      rm(
        model, train_idx, test_idx, train_sums, test_sums, association_accumulator,
        z_accumulator, prediction_sample_0, prediction_sample_1,
        prediction_sample_weight, diagnostic_positions
      )
      gc(FALSE)
    }
    
    if (any(!is.finite(residualized_reverse))) {
      stop("At least one retrospective exposure nuisance prediction is missing or non-finite.")
    }
    
    residual_mean <- residual_moments["sr"] / residual_moments["sw"]
    residual_variance <- residual_moments["sr2"] / residual_moments["sw"] - residual_mean^2
    
    list(
      residualized_reverse = residualized_reverse,
      base_D0_set = base_D0_set,
      tunes = tunes,
      diagnostics = rbindlist(fold_diagnostics, fill = TRUE),
      calibration = rbindlist(calibration_rows, fill = TRUE),
      residual_associations = rbindlist(residual_association_rows, fill = TRUE),
      prediction_distribution = rbindlist(prediction_distribution_rows, fill = TRUE),
      residual_by_observed_z = rbindlist(z_group_rows, fill = TRUE),
      residual_mean = residual_mean,
      residual_variance = residual_variance
    )
  }
  
  # ==============================================================================================
  # 5. Scalar generalized AIPW estimating equation and numerical solution
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
      included_sets <- if (use_all_pairs) {
        seq_len(n_sets_total)
      } else {
        sort(unique(pairs$set_index[pair_idx]))
      }
    }
    S <- length(included_sets)
    if (S < 1L) stop("No matched sets are available for the AIPW estimating equation.")
    
    score_sum <- sum(exposure_fit$base_D0_set[included_sets])
    jacobian_sum <- 0
    
    if (return_set_components) {
      set_score <- numeric(n_sets_total)
      set_score[included_sets] <- exposure_fit$base_D0_set[included_sets]
      set_ipw <- numeric(n_sets_total)
    }
    
    starts <- seq.int(1L, n_evaluated_pairs, by = AIPW_CHUNK_SIZE)
    for (st in starts) {
      en <- min(n_evaluated_pairs, st + AIPW_CHUNK_SIZE - 1L)
      idx <- if (use_all_pairs) st:en else pair_idx[st:en]
      z0 <- pairs$Z_event_0[idx]
      g0 <- if (use_zero_outcome_nuisance) 0 else pairs$g_0[idx]
      eta0 <- beta * z0 + g0
      exponential_eta0 <- clamp_machine_exp(eta0)
      weight <- 1 / (2 * pairs$M_s[idx])
      
      d1_score <- weight * exposure_fit$residualized_reverse[idx] * exponential_eta0
      score_sum <- score_sum + sum(d1_score)
      jacobian_sum <- jacobian_sum + sum(
        weight * exposure_fit$residualized_reverse[idx] * exponential_eta0 * z0
      )
      
      if (return_set_components) {
        group <- pairs$set_index[idx]
        set_score <- add_grouped_values(set_score, group, d1_score)
        ipw_component <- weight * (-z0 - z0 * exponential_eta0)
        set_ipw <- add_grouped_values(set_ipw, group, ipw_component)
      }
    }
    
    out <- list(
      score = score_sum / S,
      J = jacobian_sum / S,
      n_sets = S,
      included_sets = included_sets
    )
    
    if (return_set_components) {
      out$set_components <- data.table(
        set_index = included_sets,
        score = set_score[included_sets],
        ipw_component = set_ipw[included_sets]
      )
      out$set_components[, augmentation_component := score - ipw_component]
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
    beta <- as.numeric(beta_start)[1L]
    if (!is.finite(beta)) beta <- 0
    converged <- FALSE
    used_bfgs <- FALSE
    newton_iterations <- 0L
    bfgs_function_evaluations <- 0L
    history <- list()
    
    for (iteration in seq_len(max_iter)) {
      newton_iterations <- iteration
      evaluation <- evaluate_aipw(
        beta,
        pair_index,
        exposure_fit,
        pair_idx = pair_idx,
        included_sets = included_sets,
        use_zero_outcome_nuisance = use_zero_outcome_nuisance,
        return_set_components = FALSE
      )
      score_norm <- abs(evaluation$score)
      if (score_norm < NEWTON_TOL_SCORE) {
        converged <- TRUE
        break
      }
      if (!is.finite(evaluation$J) || abs(evaluation$J) < 1e-12) break
      
      step <- evaluation$score / evaluation$J
      if (!is.finite(step)) break
      if (abs(step) > NEWTON_MAX_STEP) step <- sign(step) * NEWTON_MAX_STEP
      
      old_objective <- evaluation$score^2
      alpha <- 1
      accepted <- FALSE
      for (line_search in seq_len(25L)) {
        beta_try <- beta - alpha * step
        evaluation_try <- tryCatch(
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
        if (
          !is.null(evaluation_try) && is.finite(evaluation_try$score) &&
          evaluation_try$score^2 < old_objective
        ) {
          accepted <- TRUE
          break
        }
        alpha <- alpha / 2
      }
      if (!accepted) break
      
      beta_new <- beta_try
      history[[iteration]] <- data.table(
        Iteration = iteration,
        beta_event = beta_new,
        Score_norm = abs(evaluation_try$score),
        Step_scale = alpha
      )
      if (
        abs(beta_new - beta) < NEWTON_TOL_BETA &&
        abs(evaluation_try$score) < NEWTON_TOL_SCORE * 10
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
    
    if (!converged || abs(final$score) > NEWTON_TOL_SCORE * 100) {
      used_bfgs <- TRUE
      objective <- function(candidate) {
        score <- evaluate_aipw(
          candidate,
          pair_index,
          exposure_fit,
          pair_idx = pair_idx,
          included_sets = included_sets,
          use_zero_outcome_nuisance = use_zero_outcome_nuisance,
          return_set_components = FALSE
        )$score
        score^2
      }
      optimisation <- stats::optim(
        par = beta,
        fn = objective,
        method = "BFGS",
        control = list(maxit = 1000L, reltol = 1e-12)
      )
      beta <- as.numeric(optimisation$par)[1L]
      final <- evaluate_aipw(
        beta,
        pair_index,
        exposure_fit,
        pair_idx = pair_idx,
        included_sets = included_sets,
        use_zero_outcome_nuisance = use_zero_outcome_nuisance,
        return_set_components = FALSE
      )
      bfgs_function_evaluations <- as.integer(optimisation$counts[["function"]] %||% NA_integer_)
      converged <- optimisation$convergence == 0L && abs(final$score) < NEWTON_TOL_SCORE * 100
    }
    
    if (!is.finite(final$J) || abs(final$J) < 1e-10) {
      converged <- FALSE
    }
    
    list(
      beta = beta,
      score = final$score,
      score_norm = abs(final$score),
      J = final$J,
      converged = converged,
      used_bfgs = used_bfgs,
      newton_iterations = newton_iterations,
      bfgs_function_evaluations = bfgs_function_evaluations,
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
    starts <- list(Outcome_start = as.numeric(beta_start)[1L])
    if (check_multiple_starts) {
      starts$Zero_start <- 0
      starts$Positive_perturbation <- as.numeric(beta_start)[1L] + START_PERTURBATION
      starts$Negative_perturbation <- as.numeric(beta_start)[1L] - START_PERTURBATION
    }
    
    solutions <- list()
    start_rows <- list()
    for (name in names(starts)) {
      one <- tryCatch(
        solve_aipw_core(
          pair_index = pair_index,
          exposure_fit = exposure_fit,
          beta_start = starts[[name]]
        ),
        error = function(e) list(
          beta = NA_real_,
          score = NA_real_,
          score_norm = Inf,
          J = NA_real_,
          converged = FALSE,
          used_bfgs = FALSE,
          newton_iterations = NA_integer_,
          bfgs_function_evaluations = NA_integer_,
          history = data.table(),
          error = safe_error_message(e)
        )
      )
      solutions[[name]] <- one
      start_rows[[name]] <- data.table(
        Starting_value = name,
        Start_beta_event = starts[[name]],
        Estimate_beta_event = one$beta,
        Score_norm = one$score_norm,
        Converged = one$converged,
        Used_BFGS_fallback = one$used_bfgs,
        Newton_iterations = one$newton_iterations %||% NA_integer_,
        BFGS_function_evaluations = one$bfgs_function_evaluations %||% NA_integer_,
        Error = one$error %||% NA_character_
      )
    }
    
    successful <- which(vapply(
      solutions,
      function(x) isTRUE(x$converged) && is.finite(x$beta),
      logical(1L)
    ))
    if (length(successful) == 0L) stop("The AIPW estimating equation did not converge from any starting value.")
    best_position <- successful[which.min(vapply(
      solutions[successful], function(x) x$score_norm, numeric(1L)
    ))]
    best <- solutions[[best_position]]
    
    successful_betas <- vapply(solutions[successful], function(x) x$beta, numeric(1L))
    root_spread <- if (length(successful_betas) > 1L) {
      max(successful_betas) - min(successful_betas)
    } else {
      0
    }
    if (is.finite(root_spread) && root_spread > MULTISTART_ROOT_TOL) {
      warning(sprintf(
        "Converged roots differed across starting values by %.3e; inspect the multiple-start diagnostic.",
        root_spread
      ))
    }
    
    if (!collect_diagnostics) {
      return(list(
        beta = best$beta,
        RR = exp(best$beta),
        score = best$score,
        score_norm = best$score_norm,
        J = best$J,
        Omega = NA_real_,
        variance_analytic = NA_real_,
        se_analytic = NA_real_,
        multiple_starts = rbindlist(start_rows, fill = TRUE),
        root_spread = root_spread,
        score_quantiles = data.table(),
        influential_sets = data.table(),
        fold_specific = data.table(),
        leave_one_fold_out = data.table(),
        converged = best$converged,
        used_bfgs = best$used_bfgs,
        newton_iterations = best$newton_iterations,
        bfgs_function_evaluations = best$bfgs_function_evaluations,
        exposure_only_beta = NA_real_,
        component_correlation = NA_real_,
        component_means = c(IPW = NA_real_, Augmentation = NA_real_)
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
    
    Omega <- mean(set_components$score^2)
    variance_analytic <- Omega / (S * final$J^2)
    se_analytic <- sqrt(max(variance_analytic, 0))
    influence <- -set_components$score / final$J
    
    top_n <- min(as.integer(TOP_INFLUENTIAL_SETS), length(influence))
    top_order <- order(abs(influence), decreasing = TRUE)[seq_len(top_n)]
    influential <- set_components[top_order]
    influential[, `:=`(
      influence = influence[top_order],
      absolute_influence = abs(influence[top_order]),
      influence_rank = seq_len(.N)
    )]
    influential <- set_map[influential, on = "set_index"]
    
    score_quantiles <- data.table(
      Statistic = c("min", "p01", "p05", "median", "p95", "p99", "max"),
      Value = as.numeric(stats::quantile(
        set_components$score,
        probs = c(0, 0.01, 0.05, 0.50, 0.95, 0.99, 1),
        names = FALSE
      ))
    )
    
    fold_specific <- data.table()
    leave_one_fold_out <- data.table()
    if (collect_diagnostics) {
      fold_rows <- list()
      loo_rows <- list()
      for (k in sort(unique(pair_index$pairs$fold))) {
        pair_idx <- which(pair_index$pairs$fold == k)
        included_sets <- set_map[fold == k, set_index]
        one_fold <- tryCatch(
          solve_aipw_core(
            pair_index = pair_index,
            exposure_fit = exposure_fit,
            beta_start = best$beta,
            pair_idx = pair_idx,
            included_sets = included_sets
          ),
          error = function(e) NULL
        )
        fold_rows[[as.character(k)]] <- data.table(
          Fold = k,
          N_matched_sets = length(included_sets),
          beta_event = if (is.null(one_fold)) NA_real_ else one_fold$beta,
          RR_event = if (is.null(one_fold)) NA_real_ else exp(one_fold$beta),
          Score_norm = if (is.null(one_fold)) NA_real_ else one_fold$score_norm,
          Converged = if (is.null(one_fold)) FALSE else one_fold$converged
        )
      }
      fold_specific <- rbindlist(fold_rows, fill = TRUE)
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
          beta_event = if (is.null(one)) NA_real_ else one$beta,
          RR_event = if (is.null(one)) NA_real_ else exp(one$beta),
          Score_norm = if (is.null(one)) NA_real_ else one$score_norm,
          Converged = if (is.null(one)) FALSE else one$converged
        )
      }
      leave_one_fold_out <- rbindlist(loo_rows, fill = TRUE)
    }
    
    exposure_only <- tryCatch(
      solve_aipw_core(
        pair_index = pair_index,
        exposure_fit = exposure_fit,
        beta_start = best$beta,
        use_zero_outcome_nuisance = TRUE
      ),
      error = function(e) NULL
    )
    
    list(
      beta = best$beta,
      RR = exp(best$beta),
      score = final$score,
      score_norm = abs(final$score),
      J = final$J,
      Omega = Omega,
      variance_analytic = variance_analytic,
      se_analytic = se_analytic,
      multiple_starts = rbindlist(start_rows, fill = TRUE),
      root_spread = root_spread,
      score_quantiles = score_quantiles,
      influential_sets = influential,
      fold_specific = fold_specific,
      leave_one_fold_out = leave_one_fold_out,
      converged = best$converged,
      used_bfgs = best$used_bfgs,
      newton_iterations = best$newton_iterations,
      bfgs_function_evaluations = best$bfgs_function_evaluations,
      exposure_only_beta = if (is.null(exposure_only)) NA_real_ else exposure_only$beta,
      component_correlation = suppressWarnings(stats::cor(
        set_components$ipw_component, set_components$augmentation_component
      )),
      component_means = c(
        IPW = mean(set_components$ipw_component),
        Augmentation = mean(set_components$augmentation_component)
      )
    )
  }
  
  # ==============================================================================================
  # 6. Event-history attributable fractions
  # ==============================================================================================
  
  calculate_af <- function(
    rows,
    beta,
    full_year_counts,
    full_year_case_map = NULL,
    set_map = NULL,
    keep_cases = TRUE
  ) {
    if (is.null(set_map) || nrow(set_map) == 0L) {
      stop("set_map is required for attributable-fraction calculation.")
    }
    
    seasonal_cases <- set_map[, .(
      set_index,
      set_id_original,
      case_year,
      case_month,
      E_event = as.integer(case_event)
    )]
    if (any(!seasonal_cases$E_event %in% c(0L, 1L))) {
      stop("Seasonal case-day event history must be binary before burden calculation.")
    }
    
    seasonal_cases[, `:=`(
      log_RR_event = beta * E_event,
      RR_event = exp(beta * E_event),
      AF_event_conditional = 1 - exp(-beta * E_event)
    )]
    
    annual_numerator <- seasonal_cases[, .(
      N_event_season_deaths = .N,
      N_exposed_deaths = sum(E_event == 1L),
      AF_numerator = sum(AF_event_conditional)
    ), by = .(Year = case_year)]
    
    annual <- merge(
      full_year_counts,
      annual_numerator,
      by = "Year",
      all.x = TRUE,
      sort = TRUE
    )
    annual[is.na(N_event_season_deaths), `:=`(
      N_event_season_deaths = 0L,
      N_exposed_deaths = 0L,
      AF_numerator = 0
    )]
    annual[, `:=`(
      N_deaths = N_full_year_deaths,
      AF = fifelse(N_full_year_deaths > 0L, AF_numerator / N_full_year_deaths, NA_real_),
      Summary = as.character(Year),
      AF_definition = paste0(
        "Additional event-history AF on a full-calendar-year denominator; ",
        "combine heatwave and cold-spell components additively, and add the resulting extreme-event ",
        "point estimate to the independently estimated non-optimal-temperature AF as a two-model approximation"
      )
    )]
    annual <- annual[, .(
      Year,
      N_deaths,
      N_event_season_deaths,
      N_exposed_deaths,
      AF,
      AF_definition,
      Summary
    )]
    setorder(annual, Year)
    
    if (any(!is.finite(annual$AF))) {
      missing_years <- annual[!is.finite(AF), Year]
      stop(sprintf(
        "Annual event-history AF could not be calculated for years: %s",
        paste(missing_years, collapse = ", ")
      ))
    }
    
    mean_annual <- data.table(
      Year = NA_integer_,
      N_deaths = sum(annual$N_deaths),
      N_event_season_deaths = sum(annual$N_event_season_deaths),
      N_exposed_deaths = sum(annual$N_exposed_deaths),
      AF = mean(annual$AF),
      AF_definition = annual$AF_definition[1L],
      Summary = sprintf(
        "Mean annual AF, %d-%d",
        min(ANALYSIS_YEARS),
        max(ANALYSIS_YEARS)
      )
    )
    
    cases <- NULL
    if (keep_cases) {
      if (is.null(full_year_case_map) || nrow(full_year_case_map) == 0L) {
        stop("full_year_case_map is required when SAVE_CASE_SPECIFIC = TRUE.")
      }
      cases <- copy(full_year_case_map)
      if (anyDuplicated(cases$set_id_original)) {
        stop("full_year_case_map contains duplicated matched-set identifiers.")
      }
      cases[, `:=`(
        set_index = NA_integer_,
        E_event = 0L,
        In_event_analysis_season = case_month %in% event_months
      )]
      cases[
        seasonal_cases,
        on = "set_id_original",
        `:=`(
          set_index = i.set_index,
          E_event = i.E_event
        )
      ]
      if (any(cases$In_event_analysis_season & is.na(cases$set_index))) {
        stop("At least one event-season death is missing from the fitted seasonal analysis.")
      }
      cases[, `:=`(
        log_RR_event = beta * E_event,
        RR_event = exp(beta * E_event),
        AF_event_conditional = 1 - exp(-beta * E_event)
      )]
      setcolorder(cases, c(
        "set_index",
        "set_id_original",
        "case_year",
        "case_month",
        "In_event_analysis_season",
        "E_event",
        "log_RR_event",
        "RR_event",
        "AF_event_conditional"
      ))
    }
    
    list(cases = cases, annual = annual, mean_annual = mean_annual)
  }
  
  # ==============================================================================================
  # 7. Event support, overlap, and estimating-equation diagnostics
  # ==============================================================================================
  
  build_event_support <- function(rows, set_map, pair_index, calendar_support) {
    pairs <- pair_index$pairs
    
    summary_row <- data.table(
      Support_scope = "Overall seasonal analysis",
      Year = NA_integer_,
      Threshold_percentile = THRESHOLD_PERCENTILE %||% NA_real_,
      Minimum_duration_days = MIN_DURATION_DAYS %||% NA_integer_,
      Event_lag_min = min(EVENT_HISTORY_LAGS),
      Event_lag_max = max(EVENT_HISTORY_LAGS),
      Heat_adjustment_lag_min = min(HOT_LAGS),
      Heat_adjustment_lag_max = max(HOT_LAGS),
      Cold_adjustment_lag_min = min(COLD_LAGS),
      Cold_adjustment_lag_max = max(COLD_LAGS),
      N_seasonal_matched_sets = nrow(set_map),
      N_sampled_day_rows = nrow(rows),
      N_event_history_rows = sum(rows$E_event == 1L),
      Proportion_event_history_rows = mean(rows$E_event == 1L),
      N_exposed_case_days = sum(set_map$case_event == 1L),
      Proportion_exposed_case_days = mean(set_map$case_event == 1L),
      N_event_history_control_days = sum(rows$case_internal == 0L & rows$E_event == 1L),
      Proportion_event_history_control_days = mean(rows$E_event[rows$case_internal == 0L] == 1L),
      N_sets_with_event_variation = sum(set_map$event_variation),
      Proportion_sets_with_event_variation = mean(set_map$event_variation),
      N_case_control_comparisons = nrow(pairs),
      N_ordered_pseudo_pairs_implicit = 2L * nrow(pairs),
      N_informative_comparisons = sum(abs(pairs$Z_event_0) == 1L),
      Proportion_informative_comparisons = mean(abs(pairs$Z_event_0) == 1L),
      N_equal_event_status_comparisons = sum(pairs$Z_event_0 == 0L),
      Proportion_equal_event_status_comparisons = mean(pairs$Z_event_0 == 0L),
      N_Z_minus1 = sum(pairs$Z_event_0 == -1L),
      N_Z_zero = sum(pairs$Z_event_0 == 0L),
      N_Z_plus1 = sum(pairs$Z_event_0 == 1L)
    )
    
    year_set <- set_map[, .(
      N_seasonal_matched_sets = .N,
      N_exposed_case_days = sum(case_event == 1L),
      Proportion_exposed_case_days = mean(case_event == 1L),
      N_sets_with_event_variation = sum(event_variation),
      Proportion_sets_with_event_variation = mean(event_variation)
    ), by = .(Year = case_year)]
    
    year_rows <- rows[, .(
      N_sampled_day_rows = .N,
      N_event_history_rows = sum(E_event == 1L),
      Proportion_event_history_rows = mean(E_event == 1L),
      N_event_history_control_days = sum(case_internal == 0L & E_event == 1L),
      Proportion_event_history_control_days = mean(E_event[case_internal == 0L] == 1L)
    ), by = .(Year = case_year)]
    
    year_pairs <- pairs[, .(
      N_case_control_comparisons = .N,
      N_ordered_pseudo_pairs_implicit = 2L * .N,
      N_informative_comparisons = sum(abs(Z_event_0) == 1L),
      Proportion_informative_comparisons = mean(abs(Z_event_0) == 1L),
      N_equal_event_status_comparisons = sum(Z_event_0 == 0L),
      Proportion_equal_event_status_comparisons = mean(Z_event_0 == 0L),
      N_Z_minus1 = sum(Z_event_0 == -1L),
      N_Z_zero = sum(Z_event_0 == 0L),
      N_Z_plus1 = sum(Z_event_0 == 1L)
    ), by = .(Year = case_year)]
    
    by_year <- Reduce(
      function(x, y) merge(x, y, by = "Year", all = TRUE, sort = TRUE),
      list(year_set, year_rows, year_pairs)
    )
    by_year[, `:=`(
      Support_scope = "Calendar-year seasonal support",
      Threshold_percentile = THRESHOLD_PERCENTILE %||% NA_real_,
      Minimum_duration_days = MIN_DURATION_DAYS %||% NA_integer_,
      Event_lag_min = min(EVENT_HISTORY_LAGS),
      Event_lag_max = max(EVENT_HISTORY_LAGS),
      Heat_adjustment_lag_min = min(HOT_LAGS),
      Heat_adjustment_lag_max = max(HOT_LAGS),
      Cold_adjustment_lag_min = min(COLD_LAGS),
      Cold_adjustment_lag_max = max(COLD_LAGS)
    )]
    
    output <- rbindlist(list(summary_row, by_year), use.names = TRUE, fill = TRUE)
    if (nrow(calendar_support) > 0L) {
      for (nm in names(calendar_support)) {
        value <- calendar_support[[nm]][1L]
        if (nm %in% names(output)) {
          output[Support_scope == "Overall seasonal analysis", (nm) := value]
        } else {
          missing_value <- if (is.character(value)) {
            NA_character_
          } else if (is.integer(value)) {
            NA_integer_
          } else {
            NA_real_
          }
          output[, (nm) := missing_value]
          output[Support_scope == "Overall seasonal analysis", (nm) := value]
        }
      }
    }
    output[]
  }
  
  calculate_aipw_diagnostics <- function(pair_index, exposure_fit, target_fit) {
    pairs <- pair_index$pairs
    n_pairs <- nrow(pairs)
    probability_sample <- numeric(0L)
    inverse_sample <- numeric(0L)
    sample_pair_weight <- numeric(0L)
    set.seed(ANALYSIS_SEED + 90000L)
    diagnostic_positions <- sort(sample(
      seq_len(n_pairs),
      size = min(n_pairs, DIAGNOSTIC_PREDICTION_SAMPLE),
      replace = FALSE
    ))
    probability_min <- Inf
    probability_max <- -Inf
    inverse_min <- Inf
    inverse_max <- -Inf
    weighted_lt_001 <- 0
    weighted_lt_005 <- 0
    weighted_probability_sum <- 0
    weighted_sum <- 0
    weighted_square_sum <- 0
    total_pair_weight <- 0
    
    starts <- seq.int(1L, n_pairs, by = AIPW_CHUNK_SIZE)
    for (st in starts) {
      en <- min(n_pairs, st + AIPW_CHUNK_SIZE - 1L)
      idx <- st:en
      eta0 <- target_fit$beta * pairs$Z_event_0[idx] + pairs$g_0[idx]
      probability_D1_reverse <- expit(-eta0)
      inverse_probability <- 1 / probability_D1_reverse
      pair_weight <- 1 / (2 * pairs$M_s[idx])
      
      probability_min <- min(probability_min, min(probability_D1_reverse))
      probability_max <- max(probability_max, max(probability_D1_reverse))
      inverse_min <- min(inverse_min, min(inverse_probability))
      inverse_max <- max(inverse_max, max(inverse_probability))
      weighted_lt_001 <- weighted_lt_001 + sum(pair_weight * (probability_D1_reverse < 0.01))
      weighted_lt_005 <- weighted_lt_005 + sum(pair_weight * (probability_D1_reverse < 0.05))
      
      weighted_probability_sum <- weighted_probability_sum + sum(pair_weight * probability_D1_reverse)
      weighted_contribution <- pair_weight * inverse_probability
      weighted_sum <- weighted_sum + sum(weighted_contribution)
      weighted_square_sum <- weighted_square_sum + sum(weighted_contribution^2)
      total_pair_weight <- total_pair_weight + sum(pair_weight)
      
      selected_global <- diagnostic_positions[
        diagnostic_positions >= st & diagnostic_positions <= en
      ]
      if (length(selected_global) > 0L) {
        local_position <- selected_global - st + 1L
        probability_sample <- c(probability_sample, probability_D1_reverse[local_position])
        inverse_sample <- c(inverse_sample, inverse_probability[local_position])
        sample_pair_weight <- c(sample_pair_weight, pair_weight[local_position])
      }
    }
    
    probability_quantiles <- weighted_quantile(
      probability_sample,
      probs = c(0.01, 0.05, 0.50, 0.95, 0.99),
      w = sample_pair_weight
    )
    inverse_quantiles <- weighted_quantile(
      inverse_sample,
      probs = c(0.01, 0.05, 0.50, 0.95, 0.99),
      w = sample_pair_weight
    )
    effective_sample_size <- if (weighted_square_sum > 0) {
      weighted_sum^2 / weighted_square_sum
    } else {
      NA_real_
    }
    
    data.table(
      N_matched_sets = pair_index$n_sets,
      N_case_control_comparisons = n_pairs,
      N_ordered_pseudo_pairs_implicit = 2L * n_pairs,
      N_informative_comparisons = sum(abs(pairs$Z_event_0) == 1L),
      Proportion_informative_comparisons = mean(abs(pairs$Z_event_0) == 1L),
      beta_event = target_fit$beta,
      RR_event = exp(target_fit$beta),
      Analytic_SE_diagnostic = target_fit$se_analytic,
      Final_score = target_fit$score,
      Score_norm = target_fit$score_norm,
      Target_converged = target_fit$converged,
      Newton_iterations = target_fit$newton_iterations,
      Used_BFGS_fallback = target_fit$used_bfgs,
      BFGS_function_evaluations = target_fit$bfgs_function_evaluations,
      Empirical_Jacobian = target_fit$J,
      Absolute_Jacobian = abs(target_fit$J),
      Jacobian_condition_number = if (is.finite(target_fit$J) && abs(target_fit$J) > 0) 1 else Inf,
      Exposure_residual_mean_D0 = exposure_fit$residual_mean,
      Exposure_residual_variance_D0 = exposure_fit$residual_variance,
      IPW_component_mean = target_fit$component_means["IPW"],
      Augmentation_component_mean = target_fit$component_means["Augmentation"],
      Correlation_IPW_augmentation = target_fit$component_correlation,
      Effective_sample_size_inverse_probability = effective_sample_size,
      Quantile_sample_size = length(probability_sample),
      Proportion_case_probability_lt_0_01 = weighted_lt_001 / total_pair_weight,
      Proportion_case_probability_lt_0_05 = weighted_lt_005 / total_pair_weight,
      Case_probability_min = probability_min,
      Case_probability_mean = weighted_probability_sum / total_pair_weight,
      Case_probability_p01 = probability_quantiles[1L],
      Case_probability_p05 = probability_quantiles[2L],
      Case_probability_median = probability_quantiles[3L],
      Case_probability_p95 = probability_quantiles[4L],
      Case_probability_p99 = probability_quantiles[5L],
      Case_probability_max = probability_max,
      Inverse_probability_min = inverse_min,
      Inverse_probability_mean = weighted_sum / total_pair_weight,
      Inverse_probability_p01 = inverse_quantiles[1L],
      Inverse_probability_p05 = inverse_quantiles[2L],
      Inverse_probability_median = inverse_quantiles[3L],
      Inverse_probability_p95 = inverse_quantiles[4L],
      Inverse_probability_p99 = inverse_quantiles[5L],
      Inverse_probability_max = inverse_max,
      Total_reverse_orientation_weight = total_pair_weight,
      Total_implicit_ordered_pair_weight = 2 * total_pair_weight
    )
  }
  
  # ==============================================================================================
  # 8. One complete fit for the main analysis or a bootstrap replicate
  # ==============================================================================================
  
  run_core <- function(
    rows,
    set_map,
    full_year_counts,
    full_year_case_map = NULL,
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
      full_year_counts = full_year_counts,
      full_year_case_map = full_year_case_map,
      set_map = set_map,
      keep_cases = keep_case_specific
    )
    
    aipw_summary <- if (collect_diagnostics) {
      calculate_aipw_diagnostics(pair_index, exposure_fit, target_fit)
    } else {
      data.table()
    }
    
    event_support <- if (collect_diagnostics) {
      build_event_support(rows, set_map, pair_index, event_calendar_support_main)
    } else {
      data.table()
    }
    
    estimator_comparison <- data.table(
      Estimator = c(
        outcome_fit$comparator_label,
        "Exposure-regression pathway estimator (g set to zero)",
        "Generalized AIPW estimator"
      ),
      beta_event = c(
        outcome_fit$comparator_beta,
        target_fit$exposure_only_beta,
        target_fit$beta
      )
    )
    estimator_comparison[, RR_event := exp(beta_event)]
    
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
      exposure_prediction_distribution = exposure_fit$prediction_distribution,
      exposure_residual_by_observed_z = exposure_fit$residual_by_observed_z,
      aipw_summary = aipw_summary,
      aipw_score_quantiles = target_fit$score_quantiles,
      aipw_influential_sets = target_fit$influential_sets,
      aipw_fold_specific = target_fit$fold_specific,
      aipw_leave_one_fold_out = target_fit$leave_one_fold_out,
      aipw_multiple_starts = target_fit$multiple_starts,
      estimator_comparison = estimator_comparison,
      event_support = event_support,
      score_norm = target_fit$score_norm,
      empirical_jacobian = target_fit$J
    )
    
    rm(pair_index, exposure_fit, target_fit, outcome_fit)
    gc(FALSE)
    
    toc(t0, sprintf("Complete fit finished: event-history RR=%.6f", result$RR), quiet = quiet)
    result
  }
  
  # ==============================================================================================
  # 9. Main-sample estimation
  # ==============================================================================================
  
  t_all <- tic()
  fit_main <- run_core(
    rows = data_main,
    set_map = set_map_main,
    full_year_counts = full_year_counts_main,
    full_year_case_map = full_year_case_map_main,
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
  
  output_prefix <- file.path(
    OUT_DIR,
    paste0(DATA_TAG, "_", event_short, "_", EVENT_TAG, "_case_crossover_AIPW")
  )
  early_case_file <- paste0(output_prefix, "_case_specific_AF.csv")
  if (SAVE_CASE_SPECIFIC && !is.null(fit_main$af$cases)) {
    fit_main$af$cases[, `:=`(
      Outcome = OUTCOME_NAME,
      Event_type = EVENT_TYPE,
      Event_definition = EVENT_DEFINITION,
      MMT = MMT
    )]
    setcolorder(fit_main$af$cases, c(
      "Outcome", "Event_type", "Event_definition", "MMT",
      setdiff(names(fit_main$af$cases), c("Outcome", "Event_type", "Event_definition", "MMT"))
    ))
    fwrite(fit_main$af$cases, early_case_file)
    fit_main$af$cases <- NULL
    gc(FALSE)
  }
  
  # ==============================================================================================
  # 10. Year-stratified matched-set bootstrap
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
      source_fold = fold,
      source_case_month = case_month,
      source_case_event = case_event,
      source_event_variation = event_variation
    )]
    boot_map <- source_meta[plan, on = "source_set_index"]
    boot_map[, `:=`(
      set_index = new_set_index,
      set_id_original = paste0(source_original_id, "__bootstrap_copy_", new_set_index),
      fold = source_fold,
      case_month = source_case_month,
      case_event = source_case_event,
      event_variation = source_event_variation
    )]
    boot_map <- boot_map[, .(
      set_index, set_id_original, case_year, case_month, case_event, event_variation, fold
    )]
    setorder(boot_map, set_index)
    
    rm(source_meta, plan)
    gc(FALSE)
    list(rows = boot_rows, set_map = boot_map)
  }
  
  boot_rr <- data.table(
    Bootstrap = seq_len(BOOT_B),
    beta_event = NA_real_,
    RR_event = NA_real_,
    Score_norm = NA_real_,
    Success = FALSE,
    Error = NA_character_
  )
  boot_af_rows <- list()
  
  if (BOOT_B > 0L) {
    message("------------------------------------------------------------------------------------------------")
    message("Starting the simplified year-stratified matched-set bootstrap")
    message(paste0(
      "The event definition, MMT, lag windows, outer folds, outcome-basis specifications, ",
      "and XGBoost hyperparameters are fixed."
    ))
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
          full_year_counts = full_year_counts_main,
          full_year_case_map = NULL,
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
        boot_rr[b, `:=`(Success = FALSE, Error = one$error)]
      } else {
        fit_b <- one$fit
        boot_rr[b, `:=`(
          beta_event = fit_b$beta,
          RR_event = exp(fit_b$beta),
          Score_norm = fit_b$score_norm,
          Success = TRUE,
          Error = NA_character_
        )]
        
        annual_b <- copy(fit_b$af$annual)
        annual_b[, `:=`(Bootstrap = b, Summary_type = "Annual")]
        mean_b <- copy(fit_b$af$mean_annual)
        mean_b[, `:=`(Bootstrap = b, Summary_type = "Mean_annual")]
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
  beta_boot <- if (BOOT_B > 0L) boot_rr[Success == TRUE, beta_event] else numeric(0L)
  se_boot <- if (length(beta_boot) >= 2L) stats::sd(beta_boot) else NA_real_
  ci_lower_log <- if (is.finite(se_boot)) beta_main - stats::qnorm(0.975) * se_boot else NA_real_
  ci_upper_log <- if (is.finite(se_boot)) beta_main + stats::qnorm(0.975) * se_boot else NA_real_
  p_value <- if (is.finite(se_boot) && se_boot > 0) {
    2 * (1 - stats::pnorm(abs(beta_main / se_boot)))
  } else {
    NA_real_
  }
  
  rr_results <- data.table(
    Outcome = OUTCOME_NAME,
    Event_type = EVENT_TYPE,
    Event_definition = EVENT_DEFINITION,
    Threshold_percentile = THRESHOLD_PERCENTILE %||% NA_real_,
    Minimum_duration_days = MIN_DURATION_DAYS %||% NA_integer_,
    MMT = MMT,
    Event_history_lag_window = sprintf(
      "lag%d-%d", min(EVENT_HISTORY_LAGS), max(EVENT_HISTORY_LAGS)
    ),
    Heat_adjustment_lag_window = sprintf(
      "lag%d-%d", min(HOT_LAGS), max(HOT_LAGS)
    ),
    Cold_adjustment_lag_window = sprintf(
      "lag%d-%d", min(COLD_LAGS), max(COLD_LAGS)
    ),
    Estimand = paste0(
      "Conditional mortality rate ratio for recent ", event_label,
      " history, adjusted for heat- and cold-side non-optimal-temperature components"
    ),
    beta_AIPW = beta_main,
    SE_analytic_diagnostic = fit_main$analytic_se,
    SE_bootstrap = se_boot,
    P_Wald_bootstrap = p_value,
    RR = exp(beta_main),
    RR_CI_lower = exp(ci_lower_log),
    RR_CI_upper = exp(ci_upper_log),
    beta_outcome_only = fit_main$comparator_beta,
    RR_outcome_only = exp(fit_main$comparator_beta),
    Outcome_comparator = fit_main$comparator_label,
    Bootstrap_target = BOOT_B,
    Bootstrap_success = B_ok
  )
  
  # ==============================================================================================
  # 12. Annual and mean annual AF estimates with percentile bootstrap confidence intervals
  # ==============================================================================================
  
  point_af <- rbindlist(list(
    copy(fit_main$af$annual)[, Summary_type := "Annual"],
    copy(fit_main$af$mean_annual)[, Summary_type := "Mean_annual"]
  ), fill = TRUE)
  
  af_results <- copy(point_af)
  af_results[, `:=`(
    Outcome = OUTCOME_NAME,
    Event_type = EVENT_TYPE,
    Event_definition = EVENT_DEFINITION,
    MMT = MMT,
    AF_CI_lower = NA_real_,
    AF_CI_upper = NA_real_,
    Bootstrap_success = B_ok
  )]
  
  if (nrow(boot_af) > 0L) {
    ci_af <- boot_af[, .(
      AF_CI_lower = as.numeric(stats::quantile(AF, 0.025, na.rm = TRUE, names = FALSE)),
      AF_CI_upper = as.numeric(stats::quantile(AF, 0.975, na.rm = TRUE, names = FALSE)),
      Bootstrap_success_AF = sum(is.finite(AF))
    ), by = .(Year, Summary_type)]
    
    af_results[
      ci_af,
      on = .(Year, Summary_type),
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
    "Outcome", "Event_type", "Event_definition", "MMT", "Summary_type",
    "Summary", "Year", "N_deaths", "N_event_season_deaths", "N_exposed_deaths",
    "AF", "AF_CI_lower", "AF_CI_upper", "AF_definition",
    "Bootstrap_success_AF", "Bootstrap_success"
  ))
  setorder(af_results, Summary_type, Year)
  
  # ==============================================================================================
  # 13. Label and assemble model diagnostics
  # ==============================================================================================
  
  add_common_labels <- function(table) {
    if (is.null(table) || nrow(table) == 0L) return(data.table())
    table <- copy(table)
    table[, `:=`(
      Outcome = OUTCOME_NAME,
      Event_type = EVENT_TYPE,
      Event_definition = EVENT_DEFINITION,
      MMT = MMT
    )]
    table
  }
  
  outcome_diag <- add_common_labels(fit_main$outcome_diagnostics)
  outcome_coefficients <- add_common_labels(fit_main$outcome_coefficients)
  outcome_calibration <- add_common_labels(fit_main$outcome_calibration)
  outcome_residual_scores <- add_common_labels(fit_main$outcome_residual_scores)
  outcome_fold_betas <- add_common_labels(fit_main$outcome_fold_betas)
  outcome_fold_stability <- add_common_labels(fit_main$outcome_fold_stability_summary)
  
  exposure_diag <- add_common_labels(fit_main$exposure_diagnostics)
  exposure_calibration <- add_common_labels(fit_main$exposure_calibration)
  exposure_residual_associations <- add_common_labels(fit_main$exposure_residual_associations)
  exposure_prediction_distribution <- add_common_labels(fit_main$exposure_prediction_distribution)
  exposure_residual_by_z <- add_common_labels(fit_main$exposure_residual_by_observed_z)
  
  aipw_diag <- add_common_labels(fit_main$aipw_summary)
  aipw_score_quantiles <- add_common_labels(fit_main$aipw_score_quantiles)
  aipw_influential_sets <- add_common_labels(fit_main$aipw_influential_sets)
  aipw_fold_specific <- add_common_labels(fit_main$aipw_fold_specific)
  aipw_leave_one_fold_out <- add_common_labels(fit_main$aipw_leave_one_fold_out)
  aipw_multiple_starts <- add_common_labels(fit_main$aipw_multiple_starts)
  estimator_comparison <- add_common_labels(fit_main$estimator_comparison)
  event_support <- add_common_labels(fit_main$event_support)
  
  tune_diagnostics <- flatten_tunes(fit_main$tunes)
  tune_diagnostics <- add_common_labels(tune_diagnostics)
  event_episode_table <- add_common_labels(event_episode_table_main)
  
  # ==============================================================================================
  # 14. File output
  # ==============================================================================================
  
  files <- list(
    RR = paste0(output_prefix, "_RR_results.csv"),
    AF = paste0(output_prefix, "_AF_annual_mean_results.csv"),
    case_specific_AF = if (SAVE_CASE_SPECIFIC) early_case_file else NULL,
    event_support = paste0(output_prefix, "_event_support.csv"),
    event_episodes = paste0(output_prefix, "_event_episodes.csv"),
    outcome_folds = paste0(output_prefix, "_diagnostic_outcome_folds.csv"),
    outcome_coefficients = paste0(output_prefix, "_diagnostic_outcome_coefficients.csv"),
    outcome_calibration = paste0(output_prefix, "_diagnostic_outcome_calibration.csv"),
    outcome_residual_scores = paste0(output_prefix, "_diagnostic_outcome_residual_scores.csv"),
    outcome_fold_betas = paste0(output_prefix, "_diagnostic_outcome_fold_estimates.csv"),
    outcome_fold_stability = paste0(output_prefix, "_diagnostic_outcome_fold_stability.csv"),
    exposure_folds = paste0(output_prefix, "_diagnostic_exposure_folds.csv"),
    exposure_calibration = paste0(output_prefix, "_diagnostic_exposure_calibration.csv"),
    exposure_residual_associations = paste0(output_prefix, "_diagnostic_exposure_residual_associations.csv"),
    exposure_prediction_distribution = paste0(output_prefix, "_diagnostic_exposure_prediction_distribution.csv"),
    exposure_residual_by_z = paste0(output_prefix, "_diagnostic_exposure_residual_by_observed_Z.csv"),
    AIPW_summary = paste0(output_prefix, "_diagnostic_AIPW_summary.csv"),
    AIPW_score_quantiles = paste0(output_prefix, "_diagnostic_AIPW_score_quantiles.csv"),
    AIPW_influential_sets = paste0(output_prefix, "_diagnostic_AIPW_influential_sets.csv"),
    AIPW_fold_specific = paste0(output_prefix, "_diagnostic_AIPW_fold_specific.csv"),
    AIPW_leave_one_fold_out = paste0(output_prefix, "_diagnostic_AIPW_leave_one_fold_out.csv"),
    AIPW_multiple_starts = paste0(output_prefix, "_diagnostic_AIPW_multiple_starts.csv"),
    estimator_comparison = paste0(output_prefix, "_diagnostic_estimator_comparison.csv"),
    XGBoost_hyperparameters = paste0(output_prefix, "_XGBoost_hyperparameters.csv"),
    bootstrap_RR = paste0(output_prefix, "_bootstrap_RR_trace.csv"),
    bootstrap_AF = paste0(output_prefix, "_bootstrap_AF_trace.csv"),
    compact_RDS = paste0(output_prefix, "_compact_results.rds"),
    session_info = paste0(output_prefix, "_session_info.txt")
  )
  
  write_if_nonempty <- function(table, path) {
    if (!is.null(table) && nrow(table) > 0L) fwrite(table, path)
    invisible(path)
  }
  
  fwrite(rr_results, files$RR)
  fwrite(af_results, files$AF)
  write_if_nonempty(event_support, files$event_support)
  write_if_nonempty(event_episode_table, files$event_episodes)
  write_if_nonempty(outcome_diag, files$outcome_folds)
  write_if_nonempty(outcome_coefficients, files$outcome_coefficients)
  write_if_nonempty(outcome_calibration, files$outcome_calibration)
  write_if_nonempty(outcome_residual_scores, files$outcome_residual_scores)
  write_if_nonempty(outcome_fold_betas, files$outcome_fold_betas)
  write_if_nonempty(outcome_fold_stability, files$outcome_fold_stability)
  write_if_nonempty(exposure_diag, files$exposure_folds)
  write_if_nonempty(exposure_calibration, files$exposure_calibration)
  write_if_nonempty(exposure_residual_associations, files$exposure_residual_associations)
  write_if_nonempty(exposure_prediction_distribution, files$exposure_prediction_distribution)
  write_if_nonempty(exposure_residual_by_z, files$exposure_residual_by_z)
  write_if_nonempty(aipw_diag, files$AIPW_summary)
  write_if_nonempty(aipw_score_quantiles, files$AIPW_score_quantiles)
  write_if_nonempty(aipw_influential_sets, files$AIPW_influential_sets)
  write_if_nonempty(aipw_fold_specific, files$AIPW_fold_specific)
  write_if_nonempty(aipw_leave_one_fold_out, files$AIPW_leave_one_fold_out)
  write_if_nonempty(aipw_multiple_starts, files$AIPW_multiple_starts)
  write_if_nonempty(estimator_comparison, files$estimator_comparison)
  write_if_nonempty(tune_diagnostics, files$XGBoost_hyperparameters)
  fwrite(boot_rr, files$bootstrap_RR)
  write_if_nonempty(boot_af, files$bootstrap_AF)
  
  session_lines <- capture.output(utils::sessionInfo())
  writeLines(session_lines, files$session_info)
  
  if (SAVE_COMPACT_RDS) {
    saveRDS(list(
      parameters = list(
        DATA_RDS = normalizePath(DATA_RDS, winslash = "/", mustWork = FALSE),
        OUTCOME_NAME = OUTCOME_NAME,
        EVENT_TYPE = EVENT_TYPE,
        EVENT_DEFINITION = EVENT_DEFINITION,
        THRESHOLD_PERCENTILE = THRESHOLD_PERCENTILE,
        MIN_DURATION_DAYS = MIN_DURATION_DAYS,
        MMT = MMT,
        ANALYSIS_YEARS = ANALYSIS_YEARS,
        EVENT_HISTORY_LAGS = EVENT_HISTORY_LAGS,
        HOT_LAGS = HOT_LAGS,
        COLD_LAGS = COLD_LAGS,
        RH_LAGS = RH_LAGS,
        WARM_MONTHS = WARM_MONTHS,
        COOL_MONTHS = COOL_MONTHS,
        HOT_ADJUSTMENT_DF = HOT_ADJUSTMENT_DF,
        COLD_ADJUSTMENT_DF = COLD_ADJUSTMENT_DF,
        RH_SPLINE_DF = RH_SPLINE_DF,
        K_fold = K_fold,
        BOOT_B = BOOT_B,
        tune_frac = tune_frac,
        tune_try_random = tune_try_random,
        SEED_MASTER = SEED_MASTER,
        detected_event_columns = detected_event_columns_main,
        MIN_QUALIFYING_EVENT_DAYS = MIN_QUALIFYING_EVENT_DAYS,
        THRESHOLD_BASE = THRESHOLD_BASE,
        bootstrap_conditions = paste(
          "Event definition fixed; MMT fixed; event-history and heat/cold adjustment lag windows fixed;",
          "outer-fold allocation retained; outcome-basis specifications fixed; XGBoost hyperparameters fixed."
        )
      ),
      RR_results = rr_results,
      AF_results = af_results,
      event_support = event_support,
      outcome_diagnostics = outcome_diag,
      outcome_coefficients = outcome_coefficients,
      outcome_calibration = outcome_calibration,
      outcome_residual_scores = outcome_residual_scores,
      exposure_diagnostics = exposure_diag,
      exposure_calibration = exposure_calibration,
      exposure_residual_associations = exposure_residual_associations,
      exposure_prediction_distribution = exposure_prediction_distribution,
      exposure_residual_by_observed_z = exposure_residual_by_z,
      AIPW_diagnostics = aipw_diag,
      AIPW_score_quantiles = aipw_score_quantiles,
      AIPW_influential_sets = aipw_influential_sets,
      AIPW_fold_specific = aipw_fold_specific,
      AIPW_leave_one_fold_out = aipw_leave_one_fold_out,
      AIPW_multiple_starts = aipw_multiple_starts,
      estimator_comparison = estimator_comparison,
      XGBoost_hyperparameters = tune_diagnostics,
      bootstrap_RR = boot_rr,
      bootstrap_AF = boot_af,
      files = files
    ), files$compact_RDS)
  }
  
  message("------------------------------------------------------------------------------------------------")
  message(sprintf("[%s | %s] Main event-history RR", OUTCOME_NAME, event_label))
  message(sprintf(
    "RR=%.6f (95%% CI %.6f, %.6f)",
    rr_results$RR, rr_results$RR_CI_lower, rr_results$RR_CI_upper
  ))
  message(sprintf("Results saved under: %s", OUT_DIR))
  toc(t_all, "All analyses completed")
  
  invisible(list(
    RR_results = rr_results,
    AF_results = af_results,
    event_support = event_support,
    diagnostics = list(
      outcome_folds = outcome_diag,
      outcome_coefficients = outcome_coefficients,
      outcome_calibration = outcome_calibration,
      outcome_residual_scores = outcome_residual_scores,
      exposure_folds = exposure_diag,
      exposure_calibration = exposure_calibration,
      exposure_residual_associations = exposure_residual_associations,
      exposure_prediction_distribution = exposure_prediction_distribution,
      exposure_residual_by_observed_z = exposure_residual_by_z,
      AIPW_summary = aipw_diag,
      AIPW_score_quantiles = aipw_score_quantiles,
      AIPW_influential_sets = aipw_influential_sets,
      AIPW_fold_specific = aipw_fold_specific,
      AIPW_leave_one_fold_out = aipw_leave_one_fold_out,
      AIPW_multiple_starts = aipw_multiple_starts,
      estimator_comparison = estimator_comparison
    ),
    bootstrap = list(RR = boot_rr, AF = boot_af),
    tunes = fit_main$tunes,
    basis_specs = fit_main$basis_specs,
    files = files
  ))
}

####################################################################################################
# OPTIONAL WRAPPERS FOR A BEST-DEFINITION TABLE
####################################################################################################

read_extreme_event_definition_table <- function(
    DEFINITION_XLSX,
    SHEET = 1,
    OUTCOME_COL = "outcome",
    EVENT_COL = "spell",
    SEASON_COL = "season",
    DEFINITION_COL = "definition",
    THRESHOLD_COL = "threshold_pct",
    DURATION_COL = "duration_days"
) {
  if (!requireNamespace("readxl", quietly = TRUE)) {
    stop("Install the readxl package to use a definition-table wrapper.")
  }
  if (!file.exists(DEFINITION_XLSX)) {
    stop(sprintf("Definition table does not exist: %s", DEFINITION_XLSX))
  }
  table <- as.data.table(readxl::read_excel(DEFINITION_XLSX, sheet = SHEET))
  required <- c(OUTCOME_COL, EVENT_COL, SEASON_COL, DEFINITION_COL, THRESHOLD_COL, DURATION_COL)
  missing <- setdiff(required, names(table))
  if (length(missing) > 0L) {
    stop(sprintf("Definition table is missing columns: %s", paste(missing, collapse = ", ")))
  }
  setnames(
    table,
    c(OUTCOME_COL, EVENT_COL, SEASON_COL, DEFINITION_COL, THRESHOLD_COL, DURATION_COL),
    c("outcome", "event", "season", "definition", "threshold_percentile", "minimum_duration_days")
  )
  table[, `:=`(
    outcome = as.character(outcome),
    event = tolower(as.character(event)),
    season = as.character(season),
    definition = as.character(definition),
    threshold_percentile = as.numeric(threshold_percentile),
    minimum_duration_days = as.integer(minimum_duration_days)
  )]
  table[threshold_percentile > 1, threshold_percentile := threshold_percentile / 100]
  table[]
}

run_outcome_extreme_events_from_definition_table <- function(
    DATA_RDS,
    MMT,
    OUTCOME_NAME,
    DEFINITION_XLSX,
    OUT_DIR = file.path(getwd(), "extreme_event_case_crossover_AIPW_results"),
    SHEET = 1,
    BOOT_B = 500L,
    ...
) {
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  definitions <- read_extreme_event_definition_table(DEFINITION_XLSX, SHEET = SHEET)
  selected <- definitions[outcome == OUTCOME_NAME]
  if (nrow(selected) == 0L) {
    stop(sprintf("No event definition was found for outcome %s.", OUTCOME_NAME))
  }
  
  identify_event <- function(x) {
    x <- tolower(x)
    if (x %in% c("heat", "heatwave", "heat_wave", "hw")) return("heatwave")
    if (x %in% c("cold", "coldspell", "cold_spell", "cs")) return("cold_spell")
    NA_character_
  }
  selected[, event_type_internal := vapply(event, identify_event, character(1L))]
  if (any(is.na(selected$event_type_internal))) {
    stop("At least one event label in the definition table is not recognisable as heat or cold.")
  }
  
  results <- list()
  for (event_type in c("heatwave", "cold_spell")) {
    row <- selected[event_type_internal == event_type]
    if (nrow(row) == 0L) next
    if (nrow(row) > 1L) {
      stop(sprintf(
        "More than one selected definition was found for outcome %s and event %s.",
        OUTCOME_NAME, event_type
      ))
    }
    
    results[[event_type]] <- extreme_event_case_crossover_aipw(
      DATA_RDS = DATA_RDS,
      MMT = MMT,
      EVENT_TYPE = event_type,
      EVENT_DEFINITION = row$definition[1L],
      THRESHOLD_PERCENTILE = row$threshold_percentile[1L],
      MIN_DURATION_DAYS = row$minimum_duration_days[1L],
      OUT_DIR = OUT_DIR,
      OUTCOME_NAME = OUTCOME_NAME,
      BOOT_B = BOOT_B,
      ...
    )
  }
  
  rr_components <- rbindlist(lapply(results, `[[`, "RR_results"), fill = TRUE)
  af_components_long <- rbindlist(lapply(results, `[[`, "AF_results"), fill = TRUE)
  
  combine_two_event_components <- function(results, outcome_name) {
    if (!all(c("heatwave", "cold_spell") %in% names(results))) {
      return(list(point = data.table(), bootstrap = data.table()))
    }
    
    hw <- copy(results$heatwave$AF_results)
    cs <- copy(results$cold_spell$AF_results)
    keys <- c("Year", "Summary_type")
    keep_hw <- c(keys, "Summary", "N_deaths", "N_event_season_deaths", "N_exposed_deaths",
                 "AF", "AF_CI_lower", "AF_CI_upper", "Bootstrap_success_AF")
    keep_cs <- keep_hw
    hw <- hw[, ..keep_hw]
    cs <- cs[, ..keep_cs]
    setnames(hw, setdiff(names(hw), keys), paste0(setdiff(names(hw), keys), "_HW"))
    setnames(cs, setdiff(names(cs), keys), paste0(setdiff(names(cs), keys), "_CS"))
    point <- merge(hw, cs, by = keys, all = TRUE, sort = TRUE)
    if (anyNA(point[, .(AF_HW, AF_CS, N_deaths_HW, N_deaths_CS)])) {
      stop(sprintf("Heatwave and cold-spell AF summaries do not align for outcome %s.", outcome_name))
    }
    if (any(point$N_deaths_HW != point$N_deaths_CS)) {
      stop(sprintf("Heatwave and cold-spell full-year death denominators differ for outcome %s.", outcome_name))
    }
    point[, `:=`(
      Outcome = outcome_name,
      N_deaths = N_deaths_HW,
      AF_heatwave = AF_HW,
      AF_cold_spell = AF_CS,
      AF_extreme = AF_HW + AF_CS,
      Summary = fifelse(!is.na(Summary_HW), Summary_HW, Summary_CS),
      AF_extreme_CI_lower = NA_real_,
      AF_extreme_CI_upper = NA_real_,
      Bootstrap_success_extreme = 0L
    )]
    
    bhw <- copy(results$heatwave$bootstrap$AF)
    bcs <- copy(results$cold_spell$bootstrap$AF)
    boot <- data.table()
    if (nrow(bhw) > 0L && nrow(bcs) > 0L) {
      bhw <- bhw[, .(Bootstrap, Year, Summary_type, AF_HW = AF)]
      bcs <- bcs[, .(Bootstrap, Year, Summary_type, AF_CS = AF)]
      boot <- merge(bhw, bcs, by = c("Bootstrap", keys), all = FALSE, sort = FALSE)
      boot[, `:=`(
        Outcome = outcome_name,
        AF_extreme = AF_HW + AF_CS
      )]
      ci <- boot[, .(
        AF_extreme_CI_lower = as.numeric(quantile(AF_extreme, 0.025, na.rm = TRUE, names = FALSE)),
        AF_extreme_CI_upper = as.numeric(quantile(AF_extreme, 0.975, na.rm = TRUE, names = FALSE)),
        Bootstrap_success_extreme = sum(is.finite(AF_extreme))
      ), by = keys]
      point[ci, on = keys, `:=`(
        AF_extreme_CI_lower = i.AF_extreme_CI_lower,
        AF_extreme_CI_upper = i.AF_extreme_CI_upper,
        Bootstrap_success_extreme = i.Bootstrap_success_extreme
      )]
    }
    
    point <- point[, .(
      Outcome,
      Summary_type,
      Summary,
      Year,
      N_deaths,
      N_event_season_deaths_heatwave = N_event_season_deaths_HW,
      N_event_season_deaths_cold_spell = N_event_season_deaths_CS,
      N_exposed_deaths_heatwave = N_exposed_deaths_HW,
      N_exposed_deaths_cold_spell = N_exposed_deaths_CS,
      AF_heatwave,
      AF_heatwave_CI_lower = AF_CI_lower_HW,
      AF_heatwave_CI_upper = AF_CI_upper_HW,
      AF_cold_spell,
      AF_cold_spell_CI_lower = AF_CI_lower_CS,
      AF_cold_spell_CI_upper = AF_CI_upper_CS,
      AF_extreme,
      AF_extreme_CI_lower,
      AF_extreme_CI_upper,
      Bootstrap_success_extreme,
      Combination_definition = paste0(
        "AF_extreme = AF_heatwave + AF_cold_spell; heatwave and cold-spell seasons do not overlap"
      )
    )]
    setorder(point, Summary_type, Year)
    list(point = point, bootstrap = boot)
  }
  
  combined <- combine_two_event_components(results, OUTCOME_NAME)
  af_extreme_combined <- combined$point
  bootstrap_extreme_combined <- combined$bootstrap
  
  fwrite(rr_components, file.path(OUT_DIR, paste0(OUTCOME_NAME, "_extreme_event_RR_components.csv")))
  fwrite(af_components_long, file.path(OUT_DIR, paste0(OUTCOME_NAME, "_extreme_event_AF_components_long.csv")))
  if (nrow(af_extreme_combined) > 0L) {
    fwrite(af_extreme_combined, file.path(OUT_DIR, paste0(OUTCOME_NAME, "_extreme_event_AF_combined.csv")))
  }
  if (nrow(bootstrap_extreme_combined) > 0L) {
    fwrite(
      bootstrap_extreme_combined,
      file.path(OUT_DIR, paste0(OUTCOME_NAME, "_extreme_event_AF_combined_bootstrap.csv"))
    )
  }
  
  invisible(list(
    events = results,
    RR_results = rr_components,
    AF_results = af_components_long,
    AF_combined_extreme = af_extreme_combined,
    bootstrap_combined_extreme = bootstrap_extreme_combined
  ))
}

run_all_extreme_events_from_definition_table <- function(
    DATA_DIR,
    MMT_LOOKUP,
    DEFINITION_XLSX,
    OUT_DIR = file.path(getwd(), "extreme_event_case_crossover_AIPW_results"),
    SHEET = 1,
    BOOT_B = 500L,
    FILE_SUFFIX = ".rds",
    ...
) {
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  definitions <- read_extreme_event_definition_table(DEFINITION_XLSX, SHEET = SHEET)
  outcomes <- unique(definitions$outcome)
  if (is.null(names(MMT_LOOKUP))) stop("MMT_LOOKUP must be a named numeric vector.")
  missing_mmt <- setdiff(outcomes, names(MMT_LOOKUP))
  if (length(missing_mmt) > 0L) {
    stop(sprintf("MMT_LOOKUP is missing outcomes: %s", paste(missing_mmt, collapse = ", ")))
  }
  
  results <- list()
  failures <- list()
  for (outcome in outcomes) {
    data_path <- file.path(DATA_DIR, paste0(outcome, FILE_SUFFIX))
    message("################################################################################################")
    message(sprintf("Batch outcome: %s", outcome))
    message("################################################################################################")
    one <- tryCatch(
      run_outcome_extreme_events_from_definition_table(
        DATA_RDS = data_path,
        MMT = unname(MMT_LOOKUP[[outcome]]),
        OUTCOME_NAME = outcome,
        DEFINITION_XLSX = DEFINITION_XLSX,
        OUT_DIR = OUT_DIR,
        SHEET = SHEET,
        BOOT_B = BOOT_B,
        ...
      ),
      error = function(e) e
    )
    if (inherits(one, "error")) {
      failures[[outcome]] <- data.table(Outcome = outcome, Error = conditionMessage(one))
      warning(sprintf("Outcome %s failed: %s", outcome, conditionMessage(one)))
    } else {
      results[[outcome]] <- one
    }
    gc(FALSE)
  }
  
  rr_all <- if (length(results) > 0L) {
    rbindlist(lapply(results, `[[`, "RR_results"), fill = TRUE)
  } else data.table()
  af_components_all <- if (length(results) > 0L) {
    rbindlist(lapply(results, `[[`, "AF_results"), fill = TRUE)
  } else data.table()
  af_extreme_outcome <- if (length(results) > 0L) {
    rbindlist(lapply(results, `[[`, "AF_combined_extreme"), fill = TRUE)
  } else data.table()
  boot_extreme_outcome <- if (length(results) > 0L) {
    rbindlist(lapply(results, `[[`, "bootstrap_combined_extreme"), fill = TRUE)
  } else data.table()
  failure_table <- if (length(failures) > 0L) rbindlist(failures, fill = TRUE) else data.table()
  
  af_extreme_overall <- data.table()
  boot_extreme_overall <- data.table()
  if (nrow(af_extreme_outcome) > 0L) {
    annual_point <- af_extreme_outcome[Summary_type == "Annual"]
    components <- c("AF_heatwave", "AF_cold_spell", "AF_extreme")
    af_extreme_overall <- annual_point[, {
      denominator <- sum(N_deaths)
      values <- lapply(.SD, function(x) sum(N_deaths * x) / denominator)
      c(list(N_deaths = denominator), values)
    }, by = Year, .SDcols = components]
    af_extreme_overall[, `:=`(
      Summary_type = "Annual",
      Summary = as.character(Year),
      Scope = "Across all included mutually exclusive outcomes"
    )]
    
    if (nrow(boot_extreme_outcome) > 0L) {
      death_map <- annual_point[, .(Outcome, Year, N_deaths)]
      boot_work <- death_map[boot_extreme_outcome[Summary_type == "Annual"], on = .(Outcome, Year)]
      boot_extreme_overall <- boot_work[, {
        denominator <- sum(N_deaths)
        list(
          AF_heatwave = sum(N_deaths * AF_HW) / denominator,
          AF_cold_spell = sum(N_deaths * AF_CS) / denominator,
          AF_extreme = sum(N_deaths * AF_extreme) / denominator
        )
      }, by = .(Bootstrap, Year)]
      ci_annual <- boot_extreme_overall[, .(
        AF_heatwave_CI_lower = as.numeric(quantile(AF_heatwave, 0.025, na.rm = TRUE, names = FALSE)),
        AF_heatwave_CI_upper = as.numeric(quantile(AF_heatwave, 0.975, na.rm = TRUE, names = FALSE)),
        AF_cold_spell_CI_lower = as.numeric(quantile(AF_cold_spell, 0.025, na.rm = TRUE, names = FALSE)),
        AF_cold_spell_CI_upper = as.numeric(quantile(AF_cold_spell, 0.975, na.rm = TRUE, names = FALSE)),
        AF_extreme_CI_lower = as.numeric(quantile(AF_extreme, 0.025, na.rm = TRUE, names = FALSE)),
        AF_extreme_CI_upper = as.numeric(quantile(AF_extreme, 0.975, na.rm = TRUE, names = FALSE)),
        Bootstrap_success = sum(is.finite(AF_extreme))
      ), by = Year]
      af_extreme_overall[ci_annual, on = "Year", `:=`(
        AF_heatwave_CI_lower = i.AF_heatwave_CI_lower,
        AF_heatwave_CI_upper = i.AF_heatwave_CI_upper,
        AF_cold_spell_CI_lower = i.AF_cold_spell_CI_lower,
        AF_cold_spell_CI_upper = i.AF_cold_spell_CI_upper,
        AF_extreme_CI_lower = i.AF_extreme_CI_lower,
        AF_extreme_CI_upper = i.AF_extreme_CI_upper,
        Bootstrap_success = i.Bootstrap_success
      )]
    }
    
    mean_row <- af_extreme_overall[, .(
      Year = NA_integer_,
      N_deaths = sum(N_deaths),
      AF_heatwave = mean(AF_heatwave),
      AF_cold_spell = mean(AF_cold_spell),
      AF_extreme = mean(AF_extreme),
      Summary_type = "Mean_annual",
      Summary = sprintf("Mean annual AF, %d-%d", min(Year), max(Year)),
      Scope = "Across all included mutually exclusive outcomes"
    )]
    if (nrow(boot_extreme_overall) > 0L) {
      boot_mean <- boot_extreme_overall[, .(
        AF_heatwave = mean(AF_heatwave),
        AF_cold_spell = mean(AF_cold_spell),
        AF_extreme = mean(AF_extreme)
      ), by = Bootstrap]
      mean_row[, `:=`(
        AF_heatwave_CI_lower = as.numeric(quantile(boot_mean$AF_heatwave, 0.025, na.rm = TRUE, names = FALSE)),
        AF_heatwave_CI_upper = as.numeric(quantile(boot_mean$AF_heatwave, 0.975, na.rm = TRUE, names = FALSE)),
        AF_cold_spell_CI_lower = as.numeric(quantile(boot_mean$AF_cold_spell, 0.025, na.rm = TRUE, names = FALSE)),
        AF_cold_spell_CI_upper = as.numeric(quantile(boot_mean$AF_cold_spell, 0.975, na.rm = TRUE, names = FALSE)),
        AF_extreme_CI_lower = as.numeric(quantile(boot_mean$AF_extreme, 0.025, na.rm = TRUE, names = FALSE)),
        AF_extreme_CI_upper = as.numeric(quantile(boot_mean$AF_extreme, 0.975, na.rm = TRUE, names = FALSE)),
        Bootstrap_success = sum(is.finite(boot_mean$AF_extreme))
      )]
    }
    af_extreme_overall <- rbindlist(list(af_extreme_overall, mean_row), fill = TRUE)
    setorder(af_extreme_overall, Summary_type, Year)
  }
  
  if (nrow(rr_all) > 0L) fwrite(rr_all, file.path(OUT_DIR, "extreme_event_RR_all_outcomes.csv"))
  if (nrow(af_components_all) > 0L) {
    fwrite(af_components_all, file.path(OUT_DIR, "extreme_event_AF_components_all_outcomes_long.csv"))
  }
  if (nrow(af_extreme_outcome) > 0L) {
    fwrite(af_extreme_outcome, file.path(OUT_DIR, "extreme_event_AF_outcome_specific_combined.csv"))
  }
  if (nrow(af_extreme_overall) > 0L) {
    fwrite(af_extreme_overall, file.path(OUT_DIR, "extreme_event_AF_overall_combined.csv"))
  }
  if (nrow(boot_extreme_outcome) > 0L) {
    fwrite(boot_extreme_outcome, file.path(OUT_DIR, "extreme_event_AF_outcome_specific_combined_bootstrap.csv"))
  }
  if (nrow(boot_extreme_overall) > 0L) {
    fwrite(boot_extreme_overall, file.path(OUT_DIR, "extreme_event_AF_overall_combined_bootstrap.csv"))
  }
  if (nrow(failure_table) > 0L) fwrite(failure_table, file.path(OUT_DIR, "extreme_event_failed_outcomes.csv"))
  
  invisible(list(
    results = results,
    RR_results = rr_all,
    AF_components = af_components_all,
    AF_outcome_specific_combined = af_extreme_outcome,
    AF_overall_combined = af_extreme_overall,
    bootstrap_outcome_specific_combined = boot_extreme_outcome,
    bootstrap_overall_combined = boot_extreme_overall,
    failures = failure_table
  ))
}

####################################################################################################
# OPTIONAL POST-PROCESSING: INTEGRATE NON-OPTIMAL TEMPERATURE AND EXTREME-EVENT BURDENS
####################################################################################################
#
# Following Section B.23, this helper uses the prespecified additive point-estimate approximation:
#
#   AF_extreme = AF_heatwave + AF_cold_spell
#   AF_total   ≈ AF_nonoptimal + AF_extreme
#
# Heatwave and cold-spell histories occur in non-overlapping seasons. The non-optimal-temperature and
# extreme-event components are estimated and bootstrapped separately; therefore this helper combines
# point estimates only and does not claim a jointly estimated total AF or a joint confidence interval.
####################################################################################################

combine_nonoptimal_and_extreme_event_burdens <- function(
    OUTCOME_NAMES,
    NONOPTIMAL_CASE_FILES,
    HEATWAVE_CASE_FILES,
    COLD_SPELL_CASE_FILES,
    OUT_DIR = file.path(getwd(), "integrated_temperature_burden_results"),
    ANALYSIS_YEARS = 2013:2019,
    SAVE_CASE_SPECIFIC = TRUE
) {
  OUTCOME_NAMES <- as.character(OUTCOME_NAMES)
  if (length(OUTCOME_NAMES) == 0L || any(!nzchar(OUTCOME_NAMES))) {
    stop("OUTCOME_NAMES must contain at least one non-empty outcome name.")
  }
  
  normalize_file_map <- function(x, label) {
    if (is.list(x) && !is.character(x)) x <- unlist(x, use.names = TRUE)
    x <- as.character(x)
    if (is.null(names(x)) || any(!nzchar(names(x)))) {
      if (length(x) != length(OUTCOME_NAMES)) {
        stop(sprintf(
          "%s must be a named vector/list or have the same length as OUTCOME_NAMES.",
          label
        ))
      }
      names(x) <- OUTCOME_NAMES
    }
    missing_outcomes <- setdiff(OUTCOME_NAMES, names(x))
    if (length(missing_outcomes) > 0L) {
      stop(sprintf("%s is missing outcomes: %s", label, paste(missing_outcomes, collapse = ", ")))
    }
    x[OUTCOME_NAMES]
  }
  
  nonoptimal_files <- normalize_file_map(NONOPTIMAL_CASE_FILES, "NONOPTIMAL_CASE_FILES")
  heatwave_files <- normalize_file_map(HEATWAVE_CASE_FILES, "HEATWAVE_CASE_FILES")
  cold_spell_files <- normalize_file_map(COLD_SPELL_CASE_FILES, "COLD_SPELL_CASE_FILES")
  all_files <- c(nonoptimal_files, heatwave_files, cold_spell_files)
  missing_files <- all_files[!file.exists(all_files)]
  if (length(missing_files) > 0L) {
    stop(sprintf("The following case-specific files do not exist: %s", paste(missing_files, collapse = ", ")))
  }
  
  read_case_component <- function(path, required_columns, component_label) {
    x <- data.table::fread(path)
    missing_columns <- setdiff(required_columns, names(x))
    if (length(missing_columns) > 0L) {
      stop(sprintf(
        "%s file %s is missing columns: %s",
        component_label, path, paste(missing_columns, collapse = ", ")
      ))
    }
    x <- x[, ..required_columns]
    x[, set_id_original := as.character(set_id_original)]
    x[, case_year := as.integer(case_year)]
    if (anyNA(x$set_id_original) || any(!nzchar(x$set_id_original))) {
      stop(sprintf("%s contains missing or empty matched-set identifiers.", component_label))
    }
    if (anyDuplicated(x$set_id_original)) {
      stop(sprintf("%s contains duplicated matched-set identifiers.", component_label))
    }
    x[]
  }
  
  case_tables <- vector("list", length(OUTCOME_NAMES))
  names(case_tables) <- OUTCOME_NAMES
  for (outcome in OUTCOME_NAMES) {
    nt <- read_case_component(
      nonoptimal_files[[outcome]],
      c("set_id_original", "case_year", "log_RR_total"),
      sprintf("Non-optimal-temperature case file for %s", outcome)
    )
    setnames(nt, "log_RR_total", "L_nonoptimal")
    hw <- read_case_component(
      heatwave_files[[outcome]],
      c("set_id_original", "case_year", "log_RR_event"),
      sprintf("Heatwave case file for %s", outcome)
    )
    setnames(hw, "log_RR_event", "G_heatwave")
    cs <- read_case_component(
      cold_spell_files[[outcome]],
      c("set_id_original", "case_year", "log_RR_event"),
      sprintf("Cold-spell case file for %s", outcome)
    )
    setnames(cs, "log_RR_event", "G_cold_spell")
    
    one <- merge(nt, hw, by = c("set_id_original", "case_year"), all = TRUE, sort = FALSE)
    one <- merge(one, cs, by = c("set_id_original", "case_year"), all = TRUE, sort = FALSE)
    if (nrow(one) != nrow(nt) || anyNA(one[, .(L_nonoptimal, G_heatwave, G_cold_spell)])) {
      stop(sprintf(
        paste0(
          "Case-specific files for outcome %s do not contain exactly the same deaths. ",
          "Verify that Section A, heatwave, and cold-spell analyses used the same full-year outcome data."
        ),
        outcome
      ))
    }
    if (any(!(one$case_year %in% ANALYSIS_YEARS))) {
      stop(sprintf("Outcome %s contains case years outside ANALYSIS_YEARS.", outcome))
    }
    if (any(abs(one$G_heatwave) > 1e-12 & abs(one$G_cold_spell) > 1e-12)) {
      stop(sprintf(
        "Outcome %s contains deaths with both heatwave and cold-spell contributions; check seasonal coding.",
        outcome
      ))
    }
    
    one[, `:=`(
      AF_nonoptimal = 1 - exp(-L_nonoptimal),
      AF_heatwave_additional = 1 - exp(-G_heatwave),
      AF_cold_spell_additional = 1 - exp(-G_cold_spell)
    )]
    one[, `:=`(
      AF_extreme_additional = AF_heatwave_additional + AF_cold_spell_additional,
      AF_total_temperature = AF_nonoptimal + AF_heatwave_additional + AF_cold_spell_additional,
      Outcome = outcome,
      Total_is_two_model_additive_approximation = TRUE
    )]
    setcolorder(one, c(
      "Outcome", "set_id_original", "case_year",
      "L_nonoptimal", "G_heatwave", "G_cold_spell",
      "AF_nonoptimal", "AF_heatwave_additional", "AF_cold_spell_additional",
      "AF_extreme_additional", "AF_total_temperature",
      "Total_is_two_model_additive_approximation"
    ))
    case_tables[[outcome]] <- one
  }
  
  cases_all <- rbindlist(case_tables, use.names = TRUE, fill = TRUE)
  component_columns <- c(
    "AF_nonoptimal", "AF_heatwave_additional", "AF_cold_spell_additional",
    "AF_extreme_additional", "AF_total_temperature"
  )
  annual_outcome <- cases_all[, c(
    list(N_deaths = .N),
    lapply(.SD, mean)
  ), by = .(Outcome, Year = case_year), .SDcols = component_columns]
  setorder(annual_outcome, Outcome, Year)
  
  year_grid <- CJ(Outcome = OUTCOME_NAMES, Year = as.integer(ANALYSIS_YEARS), unique = TRUE)
  annual_outcome <- annual_outcome[year_grid, on = .(Outcome, Year)]
  if (anyNA(annual_outcome$N_deaths)) {
    stop("At least one requested outcome-year combination has no full-year deaths.")
  }
  
  mean_annual_outcome <- annual_outcome[, c(
    list(
      Year = NA_integer_,
      N_deaths = sum(N_deaths),
      Summary = sprintf("Mean annual burden, %d-%d", min(ANALYSIS_YEARS), max(ANALYSIS_YEARS))
    ),
    lapply(.SD, mean)
  ), by = Outcome, .SDcols = component_columns]
  
  annual_overall <- annual_outcome[, {
    total_deaths <- sum(N_deaths)
    values <- lapply(.SD, function(x) sum(N_deaths * x) / total_deaths)
    c(list(N_deaths = total_deaths), values)
  }, by = Year, .SDcols = component_columns]
  setorder(annual_overall, Year)
  
  mean_annual_overall <- data.table(
    Year = NA_integer_,
    N_deaths = sum(annual_overall$N_deaths),
    Summary = sprintf(
      "Mean annual overall burden, %d-%d",
      min(ANALYSIS_YEARS), max(ANALYSIS_YEARS)
    )
  )
  for (nm in component_columns) {
    mean_annual_overall[, (nm) := mean(annual_overall[[nm]])]
  }
  
  dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
  files <- list(
    case_specific = file.path(OUT_DIR, "temperature_burden_integrated_case_specific.csv"),
    annual_outcome = file.path(OUT_DIR, "temperature_burden_integrated_annual_outcome_specific.csv"),
    mean_annual_outcome = file.path(OUT_DIR, "temperature_burden_integrated_mean_annual_outcome_specific.csv"),
    annual_overall = file.path(OUT_DIR, "temperature_burden_integrated_annual_overall.csv"),
    mean_annual_overall = file.path(OUT_DIR, "temperature_burden_integrated_mean_annual_overall.csv")
  )
  if (SAVE_CASE_SPECIFIC) fwrite(cases_all, files$case_specific)
  fwrite(annual_outcome, files$annual_outcome)
  fwrite(mean_annual_outcome, files$mean_annual_outcome)
  fwrite(annual_overall, files$annual_overall)
  fwrite(mean_annual_overall, files$mean_annual_overall)
  
  invisible(list(
    case_specific = if (SAVE_CASE_SPECIFIC) cases_all else NULL,
    annual_outcome_specific = annual_outcome,
    mean_annual_outcome_specific = mean_annual_outcome,
    annual_overall = annual_overall,
    mean_annual_overall = mean_annual_overall,
    files = files
  ))
}

####################################################################################################
# NON-EXECUTING EXAMPLES
####################################################################################################

# Example 3: run both selected event definitions for one outcome from an Excel table.
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/LRI.rds",
#   MMT = 22.30,
#   OUTCOME_NAME = "LRI",
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx", #event
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/CMM.rds",
#   OUTCOME_NAME = "CMM",
#   MMT = 23.3,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/CKD.rds",
#   OUTCOME_NAME = "CKD",
#   MMT = 23.7,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/HHD.rds",
#   OUTCOME_NAME = "HHD",
#   MMT = 23.5,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/SHAIV.rds",
#   OUTCOME_NAME = "SHAIV",
#   MMT = -22.4,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/UI.rds",
#   OUTCOME_NAME = "UI",
#   MMT = -19.2,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/TI.rds",
#   OUTCOME_NAME = "TI",
#   MMT = -21.5,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/IHD.rds",
#   OUTCOME_NAME = "IHD",
#   MMT = 23.0,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/STROKE.rds",
#   OUTCOME_NAME = "STROKE",
#   MMT = 23.0,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/DM.rds",
#   OUTCOME_NAME = "DM",
#   MMT = 22.9,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx",
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
# 
# both_events <- run_outcome_extreme_events_from_definition_table(
#   DATA_RDS = "/xxx/COPD.rds",
#   OUTCOME_NAME = "COPD",
#   MMT = 23.1,
#   DEFINITION_XLSX = "/xxx/Best_Definition.xlsx", 
#   OUT_DIR = "/xxx/",
#   BOOT_B = 10L
# )
