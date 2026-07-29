# ============================================================================
# File: 01_generate_extreme_event_case_crossover_simulation.R
# Purpose:
#   Generate matched case-crossover Monte Carlo data for validating the
#   generalized AIPW estimator of an additional mortality rate ratio associated
#   with a binary recent extreme-temperature event-history exposure.
#
# Primary design:
#   - Four nuisance-model specification scenarios.
#   - 1,000 independent replicates per scenario by default.
#   - 500 seasonal matched sets per replicate.
#   - One case day and 2-4 control days per matched set.
#   - Five matched-set-level outer cross-fitting folds.
#   - True event-history RR = 1.10.
#   - True background heat- and cold-side RRs = 1.060 and 1.045.
#   - Full-calendar-year burden denominator created by adding off-season deaths
#     with event-history exposure fixed at zero.
#
# Output root required by the study:
#   /xxx/ccc
#
# Optional environment variables:
#   EXTREME_SIM_MODE       = full (default) or quick
#   EXTREME_SIM_WORKERS    = number of parallel workers; default 1
#   EXTREME_SIM_OVERWRITE  = TRUE/FALSE; default FALSE
# ============================================================================

options(stringsAsFactors = FALSE)

required_packages <- c("data.table")
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
})

# ----------------------------------------------------------------------------
# 1. Fixed project path and simulation settings
# ----------------------------------------------------------------------------

PROJECT_ROOT <- "/xxx/ccc"
EXTREME_SIM_ROOT <- file.path(PROJECT_ROOT, "extreme_event_simulation")
OUTPUT_DIR <- file.path(EXTREME_SIM_ROOT, "simulated_data")
REPLICATE_DIR <- file.path(OUTPUT_DIR, "replicate_files")

SIM_MODE <- tolower(Sys.getenv("EXTREME_SIM_MODE", unset = "full"))
N_WORKERS <- suppressWarnings(as.integer(
  Sys.getenv("EXTREME_SIM_WORKERS", unset = "1")
))
OVERWRITE <- toupper(
  Sys.getenv("EXTREME_SIM_OVERWRITE", unset = "FALSE")
) %in% c("TRUE", "T", "1", "YES", "Y")

if (!is.finite(N_WORKERS) || N_WORKERS < 1L) N_WORKERS <- 1L

profile_settings <- list(
  quick = list(
    n_replicates = 2L,
    n_matched_sets = 80L,
    n_outer_folds = 5L
  ),
  full = list(
    n_replicates = 1000L,
    n_matched_sets = 500L,
    n_outer_folds = 5L
  )
)

if (!SIM_MODE %in% names(profile_settings)) {
  stop("EXTREME_SIM_MODE must be 'quick' or 'full'.", call. = FALSE)
}

cfg <- profile_settings[[SIM_MODE]]
cfg$master_seed <- 20260713L
cfg$simulation_mode <- SIM_MODE
cfg$years <- 2013:2019
cfg$control_count_values <- 2:4
cfg$control_count_probabilities <- c(0.20, 0.60, 0.20)
cfg$simulation_mmt <- 0
cfg$true_alpha_event <- log(1.10)
cfg$true_beta_heat <- log(1.060)
cfg$true_beta_cold <- log(1.045)
cfg$sigma_heat <- 1.15
cfg$sigma_cold <- 1.20
cfg$season_fraction_of_year <- 5 / 12
cfg$project_root <- PROJECT_ROOT
cfg$extreme_sim_root <- EXTREME_SIM_ROOT
cfg$output_dir <- OUTPUT_DIR
cfg$replicate_dir <- REPLICATE_DIR
cfg$n_workers <- N_WORKERS
cfg$overwrite <- OVERWRITE

# ----------------------------------------------------------------------------
# 2. Four nuisance-model specification scenarios
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
    "correct", "correct", "misspecified", "misspecified"
  ),
  exposure_nuisance_spec = c(
    "correct", "misspecified", "correct", "misspecified"
  )
)

# The data-generating mechanism is identical across the four scenarios. Only
# the fitted nuisance-model specifications differ during analysis.

# ----------------------------------------------------------------------------
# 3. Mathematical helper functions
# ----------------------------------------------------------------------------

expit <- function(x) {
  output <- numeric(length(x))
  positive <- x >= 0
  output[positive] <- 1 / (1 + exp(-x[positive]))
  exp_x <- exp(x[!positive])
  output[!positive] <- exp_x / (1 + exp_x)
  output
}

sample_softmax_index <- function(linear_predictor) {
  if (length(linear_predictor) == 1L) return(1L)
  centered <- linear_predictor - max(linear_predictor)
  weights <- exp(centered)
  if (!all(is.finite(weights)) || sum(weights) <= 0) {
    stop("Invalid case-day softmax weights were generated.", call. = FALSE)
  }
  sample.int(length(weights), size = 1L, prob = weights)
}

true_covariate_function <- function(x1, x2, x3, holiday) {
  0.35 * sin(x1) -
    0.12 * x1^2 +
    0.25 * x2 +
    0.20 * x1 * x2 +
    0.20 * x3 +
    0.30 * holiday
}

# ----------------------------------------------------------------------------
# 4. Generate one matched risk set
# ----------------------------------------------------------------------------

generate_one_matched_set <- function(
    set_number_value,
    year_value,
    fold_value,
    scenario_row,
    cfg) {
  
  set_context1 <- rnorm(1)
  set_context2 <- 0.45 * set_context1 + sqrt(1 - 0.45^2) * rnorm(1)
  
  n_controls_value <- sample(
    cfg$control_count_values,
    size = 1L,
    prob = cfg$control_count_probabilities
  )
  n_sampled_days <- n_controls_value + 1L
  
  shared_day_error <- rnorm(n_sampled_days)
  
  x1 <- 0.70 * set_context1 +
    0.75 * shared_day_error +
    rnorm(n_sampled_days, sd = 0.45)
  
  x2 <- 0.55 * set_context2 +
    0.30 * shared_day_error +
    rnorm(n_sampled_days, sd = 0.75)
  
  x3_probability <- expit(
    -0.20 +
      0.40 * set_context1 -
      0.25 * set_context2 +
      0.35 * shared_day_error
  )
  x3 <- rbinom(n_sampled_days, size = 1L, prob = x3_probability)
  
  holiday_probability <- expit(
    -1.25 + 0.25 * set_context1 - 0.15 * set_context2
  )
  holiday <- rbinom(
    n_sampled_days,
    size = 1L,
    prob = holiday_probability
  )
  
  heat_location <- 0.10 +
    0.60 * x1 -
    0.20 * x2 +
    0.30 * x3 +
    0.12 * holiday +
    0.20 * set_context1
  
  cold_location <- -0.18 * x1 +
    0.38 * x2 +
    0.30 * x3 -
    0.10 * holiday +
    0.20 * set_context2
  
  latent_heat <- rnorm(
    n_sampled_days,
    mean = heat_location,
    sd = cfg$sigma_heat
  )
  latent_cold <- rnorm(
    n_sampled_days,
    mean = cold_location,
    sd = cfg$sigma_cold
  )
  
  heat <- pmax(latent_heat - cfg$simulation_mmt, 0)
  # latent_cold is generated directly on the positive cold-deviation scale.
  cold <- pmax(latent_cold, 0)
  
  # The binary event-history exposure depends on background temperature,
  # nonlinear measured covariates, and observed matched-set context.
  event_linear_predictor <- -1.90 +
    0.45 * heat -
    0.12 * cold +
    0.30 * sin(x1) -
    0.10 * x1^2 +
    0.18 * x2 +
    0.18 * x1 * x2 +
    0.25 * x3 +
    0.25 * holiday +
    0.25 * set_context1 -
    0.10 * set_context2
  
  event_probability <- expit(event_linear_predictor)
  event_history <- rbinom(
    n_sampled_days,
    size = 1L,
    prob = event_probability
  )
  
  g_true <- true_covariate_function(x1, x2, x3, holiday)
  
  mortality_score <- cfg$true_alpha_event * event_history +
    cfg$true_beta_heat * heat +
    cfg$true_beta_cold * cold +
    g_true
  
  case_index <- sample_softmax_index(mortality_score)
  case_indicator <- integer(n_sampled_days)
  case_indicator[case_index] <- 1L
  
  matched_set <- data.table(
    case = case_indicator,
    event = as.integer(event_history),
    event_probability = event_probability,
    heat = heat,
    cold = cold,
    latent_heat = latent_heat,
    latent_cold = latent_cold,
    heat_location = heat_location,
    cold_location = cold_location,
    x1 = x1,
    x2 = x2,
    x3 = as.integer(x3),
    holiday = as.integer(holiday),
    set_context1 = set_context1,
    set_context2 = set_context2,
    g_true = g_true
  )
  
  # Randomize row order within the matched set.
  matched_set <- matched_set[sample.int(.N)]
  
  data.table::set(
    matched_set,
    j = "set_number",
    value = rep.int(as.integer(set_number_value), nrow(matched_set))
  )
  data.table::set(
    matched_set,
    j = "year",
    value = rep.int(as.integer(year_value), nrow(matched_set))
  )
  data.table::set(
    matched_set,
    j = "fold",
    value = rep.int(as.integer(fold_value), nrow(matched_set))
  )
  data.table::set(
    matched_set,
    j = "n_controls",
    value = rep.int(as.integer(n_controls_value), nrow(matched_set))
  )
  
  matched_set[, day_index := seq_len(.N)]
  matched_set[, simulation_mmt := cfg$simulation_mmt]
  matched_set[, true_alpha_event := cfg$true_alpha_event]
  matched_set[, true_beta_heat := cfg$true_beta_heat]
  matched_set[, true_beta_cold := cfg$true_beta_cold]
  matched_set[, scenario_id := scenario_row$scenario_id]
  matched_set[, scenario_label := scenario_row$scenario_label]
  matched_set[, outcome_nuisance_spec := scenario_row$outcome_nuisance_spec]
  matched_set[, exposure_nuisance_spec := scenario_row$exposure_nuisance_spec]
  
  matched_set[]
}

# ----------------------------------------------------------------------------
# 5. Generate one complete simulation replicate
# ----------------------------------------------------------------------------

generate_one_replicate <- function(scenario_row, scenario_index, replicate_id, cfg) {
  replicate_seed <- as.integer(
    cfg$master_seed + 100000L * scenario_index + replicate_id
  )
  set.seed(replicate_seed)
  
  n_sets <- cfg$n_matched_sets
  
  year_values <- sample(
    rep(cfg$years, length.out = n_sets),
    size = n_sets,
    replace = FALSE
  )
  fold_values <- sample(
    rep(seq_len(cfg$n_outer_folds), length.out = n_sets),
    size = n_sets,
    replace = FALSE
  )
  
  matched_set_list <- vector("list", n_sets)
  for (set_number_value in seq_len(n_sets)) {
    matched_set_list[[set_number_value]] <- generate_one_matched_set(
      set_number_value = set_number_value,
      year_value = year_values[set_number_value],
      fold_value = fold_values[set_number_value],
      scenario_row = scenario_row,
      cfg = cfg
    )
  }
  
  day_data <- rbindlist(matched_set_list, use.names = TRUE, fill = TRUE)
  day_data[, replicate := as.integer(replicate_id)]
  day_data[, set_id := sprintf(
    "%s_R%04d_S%06d",
    scenario_row$scenario_id,
    replicate_id,
    set_number
  )]
  day_data[, row_id := sprintf(
    "%s_R%04d_S%06d_D%02d",
    scenario_row$scenario_id,
    replicate_id,
    set_number,
    day_index
  )]
  
  setcolorder(
    day_data,
    c(
      "scenario_id", "scenario_label", "replicate", "set_id",
      "set_number", "year", "fold", "day_index", "row_id", "case",
      "n_controls", "event", "event_probability", "heat", "cold",
      "latent_heat", "latent_cold", "heat_location", "cold_location",
      "x1", "x2", "x3", "holiday", "set_context1", "set_context2",
      "g_true", "simulation_mmt", "true_alpha_event", "true_beta_heat",
      "true_beta_cold", "outcome_nuisance_spec",
      "exposure_nuisance_spec"
    )
  )
  
  # Construct the full-calendar-year burden population. Seasonal case deaths
  # carry their observed event-history status; added off-season deaths have E=0.
  seasonal_deaths <- day_data[case == 1L, .(
    scenario_id,
    replicate,
    year,
    event,
    source_set_id = set_id,
    death_period = "event_analysis_season"
  )]
  
  off_season_list <- lapply(cfg$years, function(year_value) {
    n_seasonal_year <- seasonal_deaths[year == year_value, .N]
    n_off_season <- as.integer(round(
      n_seasonal_year *
        (1 - cfg$season_fraction_of_year) /
        cfg$season_fraction_of_year
    ))
    
    if (n_off_season <= 0L) return(NULL)
    
    data.table(
      scenario_id = scenario_row$scenario_id,
      replicate = as.integer(replicate_id),
      year = as.integer(year_value),
      event = 0L,
      source_set_id = NA_character_,
      death_period = "off_season"
    )
  })
  
  burden_data <- rbindlist(
    c(list(seasonal_deaths), off_season_list),
    use.names = TRUE,
    fill = TRUE
  )
  burden_data[, death_id := sprintf(
    "%s_R%04d_B%07d",
    scenario_row$scenario_id,
    replicate_id,
    seq_len(.N)
  )]
  setcolorder(
    burden_data,
    c(
      "scenario_id", "replicate", "death_id", "year", "event",
      "death_period", "source_set_id"
    )
  )
  
  set_variation <- day_data[, .(
    event_variation = uniqueN(event) > 1L
  ), by = set_id]
  
  metadata <- data.table(
    scenario_id = scenario_row$scenario_id,
    scenario_label = scenario_row$scenario_label,
    replicate = as.integer(replicate_id),
    replicate_seed = replicate_seed,
    n_matched_sets = uniqueN(day_data$set_id),
    n_sampled_days = nrow(day_data),
    n_case_days = sum(day_data$case),
    n_control_days = sum(day_data$case == 0L),
    mean_controls_per_set = mean(day_data[case == 1L, n_controls]),
    sampled_day_event_prevalence = mean(day_data$event),
    case_day_event_prevalence = mean(day_data[case == 1L, event]),
    control_day_event_prevalence = mean(day_data[case == 0L, event]),
    proportion_sets_with_event_variation = mean(set_variation$event_variation),
    n_full_year_deaths = nrow(burden_data),
    n_off_season_deaths = sum(burden_data$death_period == "off_season"),
    true_alpha_event = cfg$true_alpha_event,
    true_rr_event = exp(cfg$true_alpha_event),
    true_beta_heat = cfg$true_beta_heat,
    true_beta_cold = cfg$true_beta_cold
  )
  
  list(
    day_data = day_data,
    burden_data = burden_data,
    metadata = metadata
  )
}

# ----------------------------------------------------------------------------
# 6. Save one replicate safely and support restart
# ----------------------------------------------------------------------------

generate_and_save_one <- function(
    scenario_row,
    scenario_index,
    replicate_id,
    cfg,
    scenario_dir) {
  
  file_name <- sprintf("replicate_%04d.rds", replicate_id)
  file_path <- file.path(scenario_dir, file_name)
  
  if (file.exists(file_path) && !isTRUE(cfg$overwrite)) {
    existing <- tryCatch(readRDS(file_path), error = function(e) NULL)
    if (!is.null(existing) &&
        is.list(existing) &&
        !is.null(existing$metadata) &&
        nrow(existing$metadata) == 1L) {
      manifest_row <- copy(existing$metadata)
      manifest_row[, file_path := normalizePath(file_path, mustWork = FALSE)]
      manifest_row[, generation_status := "existing_file_reused"]
      return(manifest_row)
    }
  }
  
  generated <- generate_one_replicate(
    scenario_row = scenario_row,
    scenario_index = scenario_index,
    replicate_id = replicate_id,
    cfg = cfg
  )
  
  temporary_path <- paste0(
    file_path,
    ".tmp_",
    Sys.getpid(),
    "_",
    sample.int(1000000L, 1L)
  )
  
  saveRDS(generated, file = temporary_path, compress = "gzip")
  if (file.exists(file_path)) unlink(file_path)
  renamed <- file.rename(temporary_path, file_path)
  if (!renamed) {
    unlink(temporary_path)
    stop("Failed to move temporary RDS file to: ", file_path, call. = FALSE)
  }
  
  manifest_row <- copy(generated$metadata)
  manifest_row[, file_path := normalizePath(file_path, mustWork = FALSE)]
  manifest_row[, generation_status := "generated"]
  manifest_row[]
}

# ----------------------------------------------------------------------------
# 7. Main generation loop
# ----------------------------------------------------------------------------

for (directory_path in c(EXTREME_SIM_ROOT, OUTPUT_DIR, REPLICATE_DIR)) {
  if (!dir.exists(directory_path)) {
    dir.create(directory_path, recursive = TRUE, showWarnings = FALSE)
  }
}

message("Extreme-event simulation mode: ", SIM_MODE)
message("Project root: ", PROJECT_ROOT)
message("Replicates per scenario: ", cfg$n_replicates)
message("Matched sets per replicate: ", cfg$n_matched_sets)
message("Outer folds: ", cfg$n_outer_folds)
message("Parallel workers: ", cfg$n_workers)
message("Overwrite existing replicate files: ", cfg$overwrite)

manifest_list <- vector("list", nrow(scenario_design))

for (scenario_index in seq_len(nrow(scenario_design))) {
  scenario_row <- as.list(scenario_design[scenario_index])
  scenario_dir <- file.path(REPLICATE_DIR, scenario_row$scenario_id)
  if (!dir.exists(scenario_dir)) {
    dir.create(scenario_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  message(
    "Generating scenario ",
    scenario_index,
    "/",
    nrow(scenario_design),
    ": ",
    scenario_row$scenario_id
  )
  
  replicate_ids <- seq_len(cfg$n_replicates)
  
  if (.Platform$OS.type == "unix" && cfg$n_workers > 1L) {
    scenario_results <- parallel::mclapply(
      replicate_ids,
      FUN = function(replicate_id) {
        generate_and_save_one(
          scenario_row = scenario_row,
          scenario_index = scenario_index,
          replicate_id = replicate_id,
          cfg = cfg,
          scenario_dir = scenario_dir
        )
      },
      mc.cores = cfg$n_workers,
      mc.preschedule = FALSE,
      mc.set.seed = FALSE
    )
  } else {
    scenario_results <- lapply(replicate_ids, function(replicate_id) {
      result <- generate_and_save_one(
        scenario_row = scenario_row,
        scenario_index = scenario_index,
        replicate_id = replicate_id,
        cfg = cfg,
        scenario_dir = scenario_dir
      )
      
      progress_interval <- max(1L, floor(cfg$n_replicates / 10L))
      if (replicate_id %% progress_interval == 0L ||
          replicate_id == cfg$n_replicates) {
        message(
          "  Completed replicate ",
          replicate_id,
          " of ",
          cfg$n_replicates
        )
      }
      result
    })
  }
  
  manifest_list[[scenario_index]] <- rbindlist(
    scenario_results,
    use.names = TRUE,
    fill = TRUE
  )
}

manifest <- rbindlist(manifest_list, use.names = TRUE, fill = TRUE)
setorder(manifest, scenario_id, replicate)

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
    "Extreme-temperature event-history generalized-AIPW simulation data",
    "",
    paste0("Simulation mode: ", SIM_MODE),
    paste0("Scenarios: ", nrow(scenario_design)),
    paste0("Replicates per scenario: ", cfg$n_replicates),
    paste0("Seasonal matched sets per replicate: ", cfg$n_matched_sets),
    paste0("Outer cross-fitting folds: ", cfg$n_outer_folds),
    paste0("True event-history RR: ", exp(cfg$true_alpha_event)),
    paste0("Calendar years: ", paste(cfg$years, collapse = ", ")),
    "",
    "Next run: 02_analyze_extreme_event_case_crossover_simulation.R"
  ),
  con = file.path(OUTPUT_DIR, "README.txt")
)

message("Extreme-event simulation-data generation completed successfully.")
message("Manifest: ", file.path(OUTPUT_DIR, "simulation_manifest.csv"))