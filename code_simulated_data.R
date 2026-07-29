# ============================================================================
# File: 01_generate_temperature_case_crossover_simulation.R
# Purpose:
#   Generate reproducible matched case-crossover simulation data for evaluating
#   a generalized augmented inverse probability weighting (AIPW) estimator of
#   side-specific heat- and cold-related mortality rate ratios.
#
# Design features implemented:
#   1. Independent matched risk sets with one case day and 2-4 control days.
#   2. Correlated time-varying covariates generated from matched-set and
#      sampled-day components.
#   3. Continuous heat- and cold-summary latent variables transformed through
#      hinge functions, thereby creating point masses at zero.
#   4. A retrospective sampling construction in which control days arise from
#      a baseline day distribution and the case day arises from its exponential
#      tilt. This construction is compatible with the conditional logistic
#      case-day probability under the correctly specified structural model.
#   5. Four primary nuisance-model specification scenarios: both nuisance
#      models correct, outcome model only correct, exposure model only correct,
#      and both nuisance models misspecified.
#   6. Calendar-year assignment for annual and mean-annual attributable-fraction
#      evaluation over 2013-2019.
#
# Output:
#   - One compressed RDS file per scenario and Monte Carlo replicate.
#   - A manifest CSV listing all generated files.
#   - A scenario-design CSV and an RDS configuration object.
#
# Reproducibility:
#   Set SIM_PROFILE to "quick", "moderate", or "full" before running.
#   The default is "moderate". Publication-scale simulations should use the
#   "full" profile or explicitly increase the configuration values below.
# ============================================================================

options(stringsAsFactors = FALSE)

required_packages <- c("data.table")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                              logical(1), quietly = TRUE)]
if (length(missing_packages) > 0L) {
  stop(
    "Missing required package(s): ", paste(missing_packages, collapse = ", "),
    ". Install them before running this script.",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
})

# ----------------------------------------------------------------------------
# 1. User-configurable settings
# ----------------------------------------------------------------------------

MASTER_SEED <- 20260712L
SIM_PROFILE <- tolower(Sys.getenv("SIM_PROFILE", unset = "full"))
PROJECT_ROOT <- "/xxx/ccc"
OUTPUT_DIR <- file.path(PROJECT_ROOT, "simulated_data")
DATA_DIR <- file.path(OUTPUT_DIR, "replicate_files")

profile_settings <- list(
  quick = list(
    n_replicates = 2L,
    n_matched_sets = 120L,
    n_case_candidates = 100L,
    n_outer_folds = 3L
  ),
  moderate = list(
    n_replicates = 20L,
    n_matched_sets = 500L,
    n_case_candidates = 300L,
    n_outer_folds = 5L
  ),
  full = list(
    n_replicates = 1000L,
    n_matched_sets = 500L,
    n_case_candidates = 300L,
    n_outer_folds = 5L
  )
)

if (!SIM_PROFILE %in% names(profile_settings)) {
  stop("SIM_PROFILE must be one of: quick, moderate, full.", call. = FALSE)
}

cfg <- profile_settings[[SIM_PROFILE]]
cfg$years <- 2013:2019
cfg$control_count_values <- 2:4
cfg$control_count_probabilities <- c(0.20, 0.60, 0.20)
cfg$sigma_heat <- 1.15
cfg$sigma_cold <- 1.20
cfg$simulation_mmt <- 0
cfg$master_seed <- MASTER_SEED
cfg$simulation_profile <- SIM_PROFILE
cfg$project_root <- PROJECT_ROOT
cfg$output_dir <- OUTPUT_DIR
cfg$data_dir <- DATA_DIR

# ----------------------------------------------------------------------------
# 2. Simulation scenarios
# ----------------------------------------------------------------------------

scenario_design <- data.table(
  scenario_id = c(
    "S1_both_correct",
    "S2_outcome_correct_exposure_misspecified",
    "S3_outcome_misspecified_exposure_correct",
    "S4_both_misspecified"
  ),
  
  scenario_label = c(
    "Both nuisance models correctly specified",
    "Outcome nuisance correct; exposure nuisance misspecified",
    "Outcome nuisance misspecified; exposure nuisance correct",
    "Both nuisance models misspecified"
  ),
  
  outcome_nuisance_spec = c(
    "correct",
    "correct",
    "misspecified",
    "misspecified"
  ),
  
  exposure_nuisance_spec = c(
    "correct",
    "misspecified",
    "correct",
    "misspecified"
  ),
  
  overlap = rep("standard", 4L),
  
  structural_model = rep("correct", 4L),
  
  beta_heat = rep(log(1.060), 4L),
  
  beta_cold = rep(log(1.045), 4L),
  
  g_scale = rep(1.00, 4L),
  
  heat_location_shift = rep(0.10, 4L),
  
  cold_location_shift = rep(0.00, 4L),
  
  shared_weather_loading = rep(0.30, 4L),
  
  within_set_exposure_scale = rep(1.00, 4L),
  
  sample_size_multiplier = rep(1.00, 4L),
  
  gamma_heat2 = rep(0.00, 4L),
  
  gamma_cold2 = rep(0.00, 4L),
  
  gamma_heat_cold = rep(0.00, 4L)
)
required_scenario_columns <- c(
  "scenario_id",
  "scenario_label",
  "outcome_nuisance_spec",
  "exposure_nuisance_spec",
  "overlap",
  "structural_model",
  "beta_heat",
  "beta_cold",
  "g_scale",
  "heat_location_shift",
  "cold_location_shift",
  "shared_weather_loading",
  "within_set_exposure_scale",
  "sample_size_multiplier",
  "gamma_heat2",
  "gamma_cold2",
  "gamma_heat_cold"
)

missing_scenario_columns <- setdiff(
  required_scenario_columns,
  names(scenario_design)
)

if (length(missing_scenario_columns) > 0L) {
  stop(
    "The scenario design is missing the following columns: ",
    paste(missing_scenario_columns, collapse = ", "),
    call. = FALSE
  )
}
# ----------------------------------------------------------------------------
# 3. Mathematical helper functions
# ----------------------------------------------------------------------------

expit <- function(x) {
  out <- numeric(length(x))
  positive <- x >= 0
  out[positive] <- 1 / (1 + exp(-x[positive]))
  ex <- exp(x[!positive])
  out[!positive] <- ex / (1 + ex)
  out
}

log_sum_exp_two <- function(log_a, log_b) {
  m <- pmax(log_a, log_b)
  m + log(exp(log_a - m) + exp(log_b - m))
}

safe_probability_sample <- function(log_weights) {
  if (length(log_weights) == 1L) {
    return(1L)
  }
  max_log_weight <- max(log_weights)
  weights <- exp(log_weights - max_log_weight)
  if (!all(is.finite(weights)) || sum(weights) <= 0) {
    stop("Non-finite importance weights were generated.", call. = FALSE)
  }
  sample.int(length(weights), size = 1L, prob = weights)
}

expected_positive_part_normal <- function(mu, sigma) {
  a <- mu / sigma
  sigma * dnorm(a) + mu * pnorm(a)
}

hinge_tilt_log_normalizer <- function(mu, sigma, beta) {
  log_negative_mass <- pnorm(-mu / sigma, log.p = TRUE)
  shifted_argument <- (mu + beta * sigma^2) / sigma
  log_positive_mass <- beta * mu + 0.5 * beta^2 * sigma^2 +
    pnorm(shifted_argument, log.p = TRUE)
  log_sum_exp_two(log_negative_mass, log_positive_mass)
}

expected_positive_part_tilted_normal <- function(mu, sigma, beta) {
  shifted_mean <- mu + beta * sigma^2
  shifted_argument <- shifted_mean / sigma
  log_multiplier <- beta * mu + 0.5 * beta^2 * sigma^2
  numerator <- exp(log_multiplier) *
    (shifted_mean * pnorm(shifted_argument) + sigma * dnorm(shifted_argument))
  denominator <- exp(hinge_tilt_log_normalizer(mu, sigma, beta))
  numerator / denominator
}

rtruncnorm_inverse_cdf <- function(n, mean, sd, lower = -Inf, upper = Inf) {
  if (length(mean) == 1L) mean <- rep(mean, n)
  if (length(sd) == 1L) sd <- rep(sd, n)
  if (length(lower) == 1L) lower <- rep(lower, n)
  if (length(upper) == 1L) upper <- rep(upper, n)
  
  lower_cdf <- pnorm((lower - mean) / sd)
  upper_cdf <- pnorm((upper - mean) / sd)
  lower_cdf <- pmin(pmax(lower_cdf, 1e-12), 1 - 1e-12)
  upper_cdf <- pmin(pmax(upper_cdf, 1e-12), 1 - 1e-12)
  
  invalid <- upper_cdf <= lower_cdf
  if (any(invalid)) {
    upper_cdf[invalid] <- pmin(1 - 1e-12, lower_cdf[invalid] + 1e-10)
  }
  
  u <- runif(n, min = lower_cdf, max = upper_cdf)
  mean + sd * qnorm(u)
}

rhinge_tilted_normal <- function(mu, sigma, beta) {
  n <- length(mu)
  log_negative_mass <- pnorm(-mu / sigma, log.p = TRUE)
  shifted_mean <- mu + beta * sigma^2
  log_positive_mass <- beta * mu + 0.5 * beta^2 * sigma^2 +
    pnorm(shifted_mean / sigma, log.p = TRUE)
  log_normalizer <- log_sum_exp_two(log_negative_mass, log_positive_mass)
  probability_positive <- exp(log_positive_mass - log_normalizer)
  
  positive_region <- runif(n) < probability_positive
  values <- numeric(n)
  
  if (any(!positive_region)) {
    idx <- which(!positive_region)
    values[idx] <- rtruncnorm_inverse_cdf(
      n = length(idx),
      mean = mu[idx],
      sd = sigma,
      lower = -Inf,
      upper = 0
    )
  }
  
  if (any(positive_region)) {
    idx <- which(positive_region)
    values[idx] <- rtruncnorm_inverse_cdf(
      n = length(idx),
      mean = shifted_mean[idx],
      sd = sigma,
      lower = 0,
      upper = Inf
    )
  }
  
  values
}

# ----------------------------------------------------------------------------
# 4. Data-generating functions
# ----------------------------------------------------------------------------

true_covariate_function <- function(x1, x2, x3, holiday, g_scale) {
  g_scale * (
    0.35 * sin(x1) -
      0.12 * x1^2 +
      0.25 * x2 +
      0.20 * x1 * x2 +
      0.20 * x3 +
      0.30 * holiday
  )
}

generate_covariates <- function(n, set_u1, set_u2) {
  shared_error <- rnorm(n)
  x1 <- 0.70 * set_u1 + 0.75 * shared_error + rnorm(n, sd = 0.45)
  x2 <- 0.55 * set_u2 + 0.30 * shared_error + rnorm(n, sd = 0.75)
  x3 <- 0.45 * set_u1 - 0.20 * set_u2 + 0.55 * shared_error +
    rnorm(n, sd = 0.55)
  holiday_probability <- expit(-1.25 + 0.25 * set_u1 - 0.15 * set_u2)
  holiday <- rbinom(n, size = 1L, prob = holiday_probability)
  
  data.table(x1 = x1, x2 = x2, x3 = x3, holiday = holiday)
}

compute_exposure_locations <- function(covariates, set_u1, set_u2, scenario) {
  heat_location <- scenario$heat_location_shift +
    0.60 * covariates$x1 -
    0.20 * covariates$x2 +
    scenario$shared_weather_loading * covariates$x3 +
    0.12 * covariates$holiday +
    0.20 * set_u1
  
  cold_location <- scenario$cold_location_shift -
    0.18 * covariates$x1 +
    0.38 * covariates$x2 +
    scenario$shared_weather_loading * covariates$x3 -
    0.10 * covariates$holiday +
    0.20 * set_u2
  
  list(heat = heat_location, cold = cold_location)
}

generate_baseline_days <- function(n, set_u1, set_u2, scenario, cfg) {
  covariates <- generate_covariates(n, set_u1, set_u2)
  locations <- compute_exposure_locations(covariates, set_u1, set_u2, scenario)
  sigma_heat <- cfg$sigma_heat * scenario$within_set_exposure_scale
  sigma_cold <- cfg$sigma_cold * scenario$within_set_exposure_scale
  
  latent_heat <- rnorm(n, mean = locations$heat, sd = sigma_heat)
  latent_cold <- rnorm(n, mean = locations$cold, sd = sigma_cold)
  
  covariates[, `:=`(
    mu_heat = locations$heat,
    mu_cold = locations$cold,
    latent_heat = latent_heat,
    latent_cold = latent_cold,
    heat = pmax(latent_heat, 0),
    cold = pmax(latent_cold, 0),
    g_true = true_covariate_function(x1, x2, x3, holiday, scenario$g_scale),
    set_u1 = set_u1,
    set_u2 = set_u2
  )]
  
  covariates
}

generate_case_day_linear_structural <- function(set_u1, set_u2, scenario, cfg) {
  candidates <- generate_covariates(cfg$n_case_candidates, set_u1, set_u2)
  locations <- compute_exposure_locations(candidates, set_u1, set_u2, scenario)
  sigma_heat <- cfg$sigma_heat * scenario$within_set_exposure_scale
  sigma_cold <- cfg$sigma_cold * scenario$within_set_exposure_scale
  
  g_values <- true_covariate_function(
    candidates$x1,
    candidates$x2,
    candidates$x3,
    candidates$holiday,
    scenario$g_scale
  )
  
  log_case_weight <- g_values +
    hinge_tilt_log_normalizer(
      locations$heat, sigma_heat, scenario$beta_heat
    ) +
    hinge_tilt_log_normalizer(
      locations$cold, sigma_cold, scenario$beta_cold
    )
  
  selected <- safe_probability_sample(log_case_weight)
  selected_covariates <- candidates[selected]
  selected_mu_heat <- locations$heat[selected]
  selected_mu_cold <- locations$cold[selected]
  
  latent_heat <- rhinge_tilted_normal(
    mu = selected_mu_heat,
    sigma = sigma_heat,
    beta = scenario$beta_heat
  )
  latent_cold <- rhinge_tilted_normal(
    mu = selected_mu_cold,
    sigma = sigma_cold,
    beta = scenario$beta_cold
  )
  
  selected_covariates[, `:=`(
    mu_heat = selected_mu_heat,
    mu_cold = selected_mu_cold,
    latent_heat = latent_heat,
    latent_cold = latent_cold,
    heat = pmax(latent_heat, 0),
    cold = pmax(latent_cold, 0),
    g_true = true_covariate_function(
      x1, x2, x3, holiday, scenario$g_scale
    ),
    set_u1 = set_u1,
    set_u2 = set_u2
  )]
  
  selected_covariates
}

generate_case_day_nonlinear_structural <- function(set_u1, set_u2, scenario, cfg) {
  candidates <- generate_baseline_days(
    n = cfg$n_case_candidates,
    set_u1 = set_u1,
    set_u2 = set_u2,
    scenario = scenario,
    cfg = cfg
  )
  
  structural_score <-
    scenario$beta_heat * candidates$heat +
    scenario$beta_cold * candidates$cold +
    scenario$gamma_heat2 * candidates$heat^2 +
    scenario$gamma_cold2 * candidates$cold^2 +
    scenario$gamma_heat_cold * candidates$heat * candidates$cold
  
  selected <- safe_probability_sample(candidates$g_true + structural_score)
  candidates[selected]
}

generate_one_matched_set <- function(set_number, year, fold, scenario, cfg) {
  set_u1 <- rnorm(1)
  set_u2 <- 0.45 * set_u1 + sqrt(1 - 0.45^2) * rnorm(1)
  n_controls <- sample(
    cfg$control_count_values,
    size = 1L,
    prob = cfg$control_count_probabilities
  )
  
  controls <- generate_baseline_days(
    n = n_controls,
    set_u1 = set_u1,
    set_u2 = set_u2,
    scenario = scenario,
    cfg = cfg
  )
  
  if (scenario$structural_model == "correct") {
    case_day <- generate_case_day_linear_structural(set_u1, set_u2, scenario, cfg)
  } else {
    case_day <- generate_case_day_nonlinear_structural(set_u1, set_u2, scenario, cfg)
  }
  
  case_day[, case := 1L]
  controls[, case := 0L]
  matched_set <- rbindlist(list(case_day, controls), use.names = TRUE)
  matched_set <- matched_set[sample.int(.N)]
  
  # Store scalar values outside data.table evaluation to avoid ambiguous
  # name resolution and zero-length replacement errors.
  
  n_sampled_days <- nrow(matched_set)
  
  set_number_value <- as.integer(set_number)
  year_value <- as.integer(year)
  fold_value <- as.integer(fold)
  n_controls_value <- as.integer(n_controls)
  
  sigma_heat_value <- as.numeric(
    cfg$sigma_heat * scenario$within_set_exposure_scale
  )
  
  sigma_cold_value <- as.numeric(
    cfg$sigma_cold * scenario$within_set_exposure_scale
  )
  
  simulation_mmt_value <- as.numeric(cfg$simulation_mmt)
  true_beta_heat_value <- as.numeric(scenario$beta_heat)
  true_beta_cold_value <- as.numeric(scenario$beta_cold)
  true_gamma_heat2_value <- as.numeric(scenario$gamma_heat2)
  true_gamma_cold2_value <- as.numeric(scenario$gamma_cold2)
  true_gamma_heat_cold_value <- as.numeric(
    scenario$gamma_heat_cold
  )
  
  # Verify that every matched-set-level parameter is a scalar.
  scalar_values <- list(
    set_number = set_number_value,
    year = year_value,
    fold = fold_value,
    n_controls = n_controls_value,
    sigma_heat = sigma_heat_value,
    sigma_cold = sigma_cold_value,
    simulation_mmt = simulation_mmt_value,
    true_beta_heat = true_beta_heat_value,
    true_beta_cold = true_beta_cold_value,
    true_gamma_heat2 = true_gamma_heat2_value,
    true_gamma_cold2 = true_gamma_cold2_value,
    true_gamma_heat_cold = true_gamma_heat_cold_value
  )
  
  invalid_scalar_names <- names(scalar_values)[
    vapply(
      scalar_values,
      function(x) length(x) != 1L,
      logical(1)
    )
  ]
  
  if (length(invalid_scalar_names) > 0L) {
    stop(
      "The following matched-set parameters are missing or not scalar: ",
      paste(invalid_scalar_names, collapse = ", "),
      call. = FALSE
    )
  }
  
  # Add columns explicitly using data.table::set().
  # This avoids repeated non-standard evaluation within :=.
  data.table::set(
    matched_set,
    j = "set_number",
    value = rep.int(set_number_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "year",
    value = rep.int(year_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "fold",
    value = rep.int(fold_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "day_index",
    value = seq_len(n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "n_controls",
    value = rep.int(n_controls_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "sigma_heat",
    value = rep.int(sigma_heat_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "sigma_cold",
    value = rep.int(sigma_cold_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "simulation_mmt",
    value = rep.int(simulation_mmt_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "true_beta_heat",
    value = rep.int(true_beta_heat_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "true_beta_cold",
    value = rep.int(true_beta_cold_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "true_gamma_heat2",
    value = rep.int(true_gamma_heat2_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "true_gamma_cold2",
    value = rep.int(true_gamma_cold2_value, n_sampled_days)
  )
  
  data.table::set(
    matched_set,
    j = "true_gamma_heat_cold",
    value = rep.int(
      true_gamma_heat_cold_value,
      n_sampled_days
    )
  )
  
  matched_set
}

generate_one_replicate <- function(scenario, replicate_id, cfg) {
  replicate_seed <- as.integer(
    cfg$master_seed +
      100000L * match(scenario$scenario_id, scenario_design$scenario_id) +
      replicate_id
  )
  set.seed(replicate_seed)
  
  n_sets <- as.integer(round(
    cfg$n_matched_sets * scenario$sample_size_multiplier
  ))
  years <- sample(rep(cfg$years, length.out = n_sets), size = n_sets)
  folds <- sample(rep(seq_len(cfg$n_outer_folds), length.out = n_sets), size = n_sets)
  
  set_list <- vector("list", n_sets)
  for (s in seq_len(n_sets)) {
    set_list[[s]] <- generate_one_matched_set(
      set_number = s,
      year = years[s],
      fold = folds[s],
      scenario = scenario,
      cfg = cfg
    )
  }
  
  day_data <- rbindlist(set_list, use.names = TRUE)
  day_data[, `:=`(
    scenario_id = scenario$scenario_id,
    scenario_label = scenario$scenario_label,
    outcome_nuisance_spec = scenario$outcome_nuisance_spec,
    exposure_nuisance_spec = scenario$exposure_nuisance_spec,
    overlap = scenario$overlap,
    structural_model = scenario$structural_model,
    replicate = replicate_id,
    set_id = sprintf("%s_R%04d_S%06d", scenario$scenario_id, replicate_id, set_number),
    row_id = sprintf(
      "%s_R%04d_S%06d_D%02d",
      scenario$scenario_id, replicate_id, set_number, day_index
    )
  )]
  
  setcolorder(
    day_data,
    c(
      "scenario_id", "scenario_label", "replicate", "set_id", "set_number",
      "year", "fold", "day_index", "row_id", "case", "n_controls",
      "heat", "cold", "latent_heat", "latent_cold", "mu_heat", "mu_cold",
      "x1", "x2", "x3", "holiday", "set_u1", "set_u2", "g_true",
      "sigma_heat", "sigma_cold", "simulation_mmt", "true_beta_heat", "true_beta_cold",
      "true_gamma_heat2", "true_gamma_cold2", "true_gamma_heat_cold",
      "outcome_nuisance_spec", "exposure_nuisance_spec", "overlap",
      "structural_model"
    )
  )
  
  day_data[]
}

# ----------------------------------------------------------------------------
# 5. Generate and save all simulation replicates
# ----------------------------------------------------------------------------

if (!dir.exists(DATA_DIR)) {
  dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
}

manifest_rows <- list()
manifest_index <- 0L

message("Simulation profile: ", SIM_PROFILE)
message("Scenarios: ", nrow(scenario_design))
message("Replicates per scenario: ", cfg$n_replicates)
message("Base matched sets per replicate: ", cfg$n_matched_sets)
message("Output directory: ", normalizePath(OUTPUT_DIR, mustWork = FALSE))

for (scenario_index in seq_len(nrow(scenario_design))) {
  scenario <- as.list(scenario_design[scenario_index])
  scenario_dir <- file.path(DATA_DIR, scenario$scenario_id)
  if (!dir.exists(scenario_dir)) {
    dir.create(scenario_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  message("Generating scenario: ", scenario$scenario_id)
  
  for (replicate_id in seq_len(cfg$n_replicates)) {
    day_data <- generate_one_replicate(scenario, replicate_id, cfg)
    file_name <- sprintf("replicate_%04d.rds", replicate_id)
    file_path <- file.path(scenario_dir, file_name)
    saveRDS(day_data, file = file_path, compress = "xz")
    
    manifest_index <- manifest_index + 1L
    manifest_rows[[manifest_index]] <- data.table(
      scenario_id = scenario$scenario_id,
      scenario_label = scenario$scenario_label,
      replicate = replicate_id,
      file_path = normalizePath(file_path, mustWork = FALSE),
      n_matched_sets = uniqueN(day_data$set_id),
      n_sampled_days = nrow(day_data),
      n_case_days = sum(day_data$case),
      mean_controls_per_set = mean(day_data[case == 1L, n_controls]),
      proportion_zero_heat = mean(day_data$heat == 0),
      proportion_zero_cold = mean(day_data$cold == 0),
      heat_cold_correlation = suppressWarnings(cor(day_data$heat, day_data$cold)),
      replicate_seed = as.integer(
        cfg$master_seed + 100000L * scenario_index + replicate_id
      )
    )
    
    if (replicate_id %% max(1L, floor(cfg$n_replicates / 10L)) == 0L ||
        replicate_id == cfg$n_replicates) {
      message(
        "  Completed replicate ", replicate_id, " of ", cfg$n_replicates
      )
    }
  }
}

manifest <- rbindlist(manifest_rows, use.names = TRUE)

fwrite(manifest, file.path(OUTPUT_DIR, "simulation_manifest.csv"))
fwrite(scenario_design, file.path(OUTPUT_DIR, "scenario_design.csv"))
saveRDS(
  list(
    configuration = cfg,
    scenario_design = scenario_design,
    generated_at = Sys.time(),
    r_version = R.version.string
  ),
  file.path(OUTPUT_DIR, "simulation_configuration.rds")
)

writeLines(
  c(
    "Temperature generalized-AIPW simulation data",
    "",
    paste0("Simulation profile: ", SIM_PROFILE),
    paste0("Number of scenarios: ", nrow(scenario_design)),
    paste0("Replicates per scenario: ", cfg$n_replicates),
    paste0("Matched sets per replicate: ", cfg$n_matched_sets),
    paste0("Outer folds: ", cfg$n_outer_folds),
    paste0("Calendar years: ", paste(cfg$years, collapse = ", ")),
    "",
    "Run 02_analyze_temperature_case_crossover_simulation.R after generation."
  ),
  con = file.path(OUTPUT_DIR, "README.txt")
)

message("Simulation-data generation completed successfully.")
message("Manifest: ", file.path(OUTPUT_DIR, "simulation_manifest.csv"))