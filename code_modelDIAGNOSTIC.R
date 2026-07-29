####################################################################################################
# REVIEWER-READY CORE MODEL-DIAGNOSTIC AGGREGATION SCRIPT
# Generalized AIPW analyses of non-optimal daily temperature and extreme-temperature events
####################################################################################################
#
# PURPOSE
# -------
# This standalone script reads only the diagnostic outputs required for the final publication
# tables. It does not refit any statistical model.
#
# The script creates two Excel workbooks:
#
#   1. Model_diagnostics_non_optimal_temperature.xlsx
#   2. Model_diagnostics_extreme_temperature_events.xlsx
#
# The non-optimal-temperature table contains:
#   - mortality outcome;
#   - number of matched sets;
#   - number of case-control comparisons; and
#   - first percentile of the fitted case-first probabilities.
#
# The extreme-temperature-event table additionally contains:
#   - event type;
#   - event definition; and
#   - percentage of informative comparisons with a non-zero event-history contrast.
#
# Convergence of the cross-fitted conditional logistic outcome models, convergence of the
# generalized AIPW estimating equations, final score norms, proportions of fitted case-first
# probabilities below 0.05, BFGS fallback use for extreme-event models, and bootstrap completion
# are checked internally and summarized in the table notes rather than displayed as separate columns.
#
# The two analysis families are stored in separate result directories. The non-optimal-temperature
# aggregation reads only from NONOPTIMAL_RESULTS_DIR, whereas the heatwave and cold-spell aggregation
# reads only from EXTREME_RESULTS_DIR. CSV-schema checks are retained as an additional safeguard.
# When duplicate files are found for the same analysis key within a directory, the most recently
# modified file is retained.
####################################################################################################

options(stringsAsFactors = FALSE)

required_packages <- c("data.table", "openxlsx")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    sprintf(
      "Install the following R packages before running this script: %s",
      paste(missing_packages, collapse = ", ")
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(openxlsx)
})

# ================================================================================================
# 1. Paths and reporting settings
# ================================================================================================

NONOPTIMAL_RESULTS_DIR <-
  "/xxx/xxx"

EXTREME_RESULTS_DIR <-
  "/xxx/xxx"

OUTPUT_DIR <-
  "/xxx/xxx"

NONOPTIMAL_OUTPUT <- file.path(
  OUTPUT_DIR,
  "Model_diagnostics_non_optimal_temperature.xlsx"
)

EXTREME_OUTPUT <- file.path(
  OUTPUT_DIR,
  "Model_diagnostics_extreme_temperature_events.xlsx"
)

OUTCOME_ORDER <- c(
  "CKD", "CMM", "COPD", "DM", "HHD", "IHD", "LRI", "Stroke", "UI", "TI", "SHAIV"
)

# A final score norm below this threshold is regarded as numerical convergence when an explicit
# Target_converged field is unavailable in a saved diagnostic file.
SCORE_TOLERANCE_FOR_REPORTING <- 1e-6

# Number of decimal places displayed in the publication tables.
CASE_PROBABILITY_DIGITS <- 3L
INFORMATIVE_PERCENT_DIGITS <- 2L

for (results_directory in c(NONOPTIMAL_RESULTS_DIR, EXTREME_RESULTS_DIR)) {
  if (!dir.exists(results_directory)) {
    stop(sprintf("Results directory does not exist: %s", results_directory), call. = FALSE)
  }
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ================================================================================================
# 2. General utilities
# ================================================================================================

check_columns <- function(x, required, label) {
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L) {
    stop(
      sprintf("%s is missing required columns: %s", label, paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
}

as_logical_safe <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x) || is.integer(x)) return(!is.na(x) & x != 0)
  value <- tolower(trimws(as.character(x)))
  out <- rep(NA, length(value))
  out[value %in% c("true", "t", "yes", "y", "1")] <- TRUE
  out[value %in% c("false", "f", "no", "n", "0")] <- FALSE
  out
}

format_integer <- function(x) {
  format(as.integer(x), big.mark = ",", scientific = FALSE, trim = TRUE)
}

format_decimal <- function(x, digits = 3L) {
  formatC(as.numeric(x), format = "f", digits = digits)
}

format_score <- function(x) {
  x <- as.numeric(x)
  if (!is.finite(x)) return("not available")
  if (abs(x) < 0.00005) return("zero at the reported numerical precision")
  format(x, digits = 3L, scientific = TRUE)
}

apply_outcome_order <- function(x, event_table = FALSE) {
  out <- copy(x)
  additional_outcomes <- sort(setdiff(unique(out$Outcome), OUTCOME_ORDER))
  complete_order <- c(OUTCOME_ORDER, additional_outcomes)
  out[, Outcome_order := match(Outcome, complete_order)]
  
  if (event_table) {
    out[, Event_order := fifelse(
      `Event type` == "Heatwave", 1L,
      fifelse(`Event type` == "Cold spell", 2L, 3L)
    )]
    setorder(out, Outcome_order, Event_order, `Event definition`)
    out[, c("Outcome_order", "Event_order") := NULL]
  } else {
    setorder(out, Outcome_order)
    out[, Outcome_order := NULL]
  }
  out
}

publication_outcome_label <- function(x) {
  fifelse(x == "STROKE", "Stroke", x)
}

# Read all files with a specified suffix, identify the analysis family from the CSV schema, and
# retain the most recently modified file for each analysis key.
read_latest_by_suffix <- function(
    directory,
    suffix,
    label,
    analysis_family = c("nonoptimal", "extreme"),
    key_columns
) {
  analysis_family <- match.arg(analysis_family)
  
  files <- list.files(directory, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
  files <- files[endsWith(basename(files), suffix)]
  files <- files[file.exists(files) & file.info(files)$size > 0L]
  
  if (length(files) == 0L) {
    stop(
      sprintf("No %s files ending in '%s' were found under %s", label, suffix, directory),
      call. = FALSE
    )
  }
  
  tables <- list()
  for (file in files) {
    one <- tryCatch(
      fread(file, showProgress = FALSE),
      error = function(e) {
        warning(sprintf("Could not read %s: %s", file, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(one) || nrow(one) == 0L) next
    
    is_extreme <- all(c("Event_type", "Event_definition") %in% names(one))
    keep <- if (analysis_family == "extreme") is_extreme else !is_extreme
    if (!keep) next
    
    info <- file.info(file)
    one[, `:=`(
      Source_file = basename(file),
      Source_path = normalizePath(file, winslash = "/", mustWork = FALSE),
      Source_modified_time = as.POSIXct(info$mtime)
    )]
    tables[[length(tables) + 1L]] <- one
  }
  
  if (length(tables) == 0L) {
    stop(
      sprintf("No %s %s files were identified under %s.", analysis_family, label, directory),
      call. = FALSE
    )
  }
  
  combined <- rbindlist(tables, use.names = TRUE, fill = TRUE)
  check_columns(combined, key_columns, sprintf("Combined %s %s files", analysis_family, label))
  
  # A saved CSV may contain more than one row for an analysis key. The duplicate-file selection is
  # therefore performed using the source-file modification time rather than by dropping data rows.
  source_map <- unique(combined[, c(key_columns, "Source_path", "Source_modified_time"), with = FALSE])
  setorderv(source_map, c(key_columns, "Source_modified_time"), c(rep(1L, length(key_columns)), -1L))
  latest_sources <- source_map[, .SD[1L], by = key_columns]
  
  selected <- merge(
    combined,
    latest_sources[, c(key_columns, "Source_path"), with = FALSE],
    by = c(key_columns, "Source_path"),
    all = FALSE,
    sort = FALSE
  )
  
  n_source_files <- uniqueN(selected$Source_path)
  message(sprintf(
    "Selected %d latest %s %s file(s).",
    n_source_files, analysis_family, label
  ))
  
  selected
}

summarize_bootstrap_completion <- function(rr, key_columns) {
  check_columns(
    rr,
    c(key_columns, "Bootstrap_target", "Bootstrap_success"),
    "RR/bootstrap results"
  )
  
  rr[, .(
    Bootstrap_target = suppressWarnings(max(as.numeric(Bootstrap_target), na.rm = TRUE)),
    Bootstrap_success = suppressWarnings(min(as.numeric(Bootstrap_success), na.rm = TRUE))
  ), by = key_columns]
}

# ================================================================================================
# 3. Non-optimal daily temperature diagnostics
# ================================================================================================

build_nonoptimal_table <- function(directory) {
  key <- "Outcome"
  
  aipw <- read_latest_by_suffix(
    directory = directory,
    suffix = "_diagnostic_AIPW_summary.csv",
    label = "AIPW summary",
    analysis_family = "nonoptimal",
    key_columns = key
  )
  
  outcome <- read_latest_by_suffix(
    directory = directory,
    suffix = "_diagnostic_outcome_folds.csv",
    label = "outcome-fold diagnostic",
    analysis_family = "nonoptimal",
    key_columns = key
  )
  
  rr <- read_latest_by_suffix(
    directory = directory,
    suffix = "_RR_results.csv",
    label = "RR/bootstrap result",
    analysis_family = "nonoptimal",
    key_columns = key
  )
  
  check_columns(
    aipw,
    c(
      "Outcome", "N_matched_sets", "N_case_control_pairs", "Score_norm",
      "pi_case_first_p01", "Proportion_pi_case_first_lt_0_05"
    ),
    "Non-optimal-temperature AIPW summaries"
  )
  check_columns(
    outcome,
    c("Outcome", "Fold", "Converged"),
    "Non-optimal-temperature outcome-fold diagnostics"
  )
  
  if (aipw[, anyDuplicated(Outcome)] > 0L) {
    stop(
      paste0(
        "More than one AIPW summary row remains for at least one non-optimal-temperature outcome. ",
        "Inspect the selected latest summary files."
      ),
      call. = FALSE
    )
  }
  
  bootstrap <- summarize_bootstrap_completion(rr, key)
  
  table <- aipw[, .(
    Outcome,
    `Matched sets` = as.integer(N_matched_sets),
    `Case-control comparisons` = as.integer(N_case_control_pairs),
    `Case-first probability, 1st percentile` = round(
      as.numeric(pi_case_first_p01),
      CASE_PROBABILITY_DIGITS
    )
  )]
  
  table[, Outcome := publication_outcome_label(Outcome)]
  table <- apply_outcome_order(table, event_table = FALSE)
  
  # Internal checks summarized in the publication table note.
  fold_status <- outcome[, .(
    N_folds = .N,
    N_converged = sum(as_logical_safe(Converged) %in% TRUE, na.rm = TRUE)
  ), by = Outcome]
  
  if ("Target_converged" %in% names(aipw)) {
    target_converged <- as_logical_safe(aipw$Target_converged)
  } else {
    target_converged <- is.finite(aipw$Score_norm) &
      abs(as.numeric(aipw$Score_norm)) <= SCORE_TOLERANCE_FOR_REPORTING
  }
  
  note_values <- list(
    n_folds = sum(fold_status$N_folds),
    n_folds_converged = sum(fold_status$N_converged),
    n_models = nrow(aipw),
    n_targets_converged = sum(target_converged %in% TRUE, na.rm = TRUE),
    max_score_norm = suppressWarnings(max(abs(as.numeric(aipw$Score_norm)), na.rm = TRUE)),
    max_low_probability_percent = suppressWarnings(
      100 * max(as.numeric(aipw$Proportion_pi_case_first_lt_0_05), na.rm = TRUE)
    ),
    bootstrap_success = sum(bootstrap$Bootstrap_success, na.rm = TRUE),
    bootstrap_target = sum(bootstrap$Bootstrap_target, na.rm = TRUE),
    all_bootstrap_successful = all(
      is.finite(bootstrap$Bootstrap_target) &
        is.finite(bootstrap$Bootstrap_success) &
        bootstrap$Bootstrap_success >= bootstrap$Bootstrap_target
    )
  )
  
  note <- paste0(
    "All ", note_values$n_folds_converged, " of ", note_values$n_folds,
    " cross-fitted conditional logistic outcome-model folds converged. All ",
    note_values$n_targets_converged, " of ", note_values$n_models,
    " outcome-specific generalized AIPW estimating equations converged, and the maximum final ",
    "score norm was ", format_score(note_values$max_score_norm), ". The proportion of fitted ",
    "case-first probabilities below 0.05 was ",
    format_decimal(note_values$max_low_probability_percent, 2L),
    "% in the most extreme outcome. ",
    if (note_values$all_bootstrap_successful) {
      paste0(
        "All ", format_integer(note_values$bootstrap_success), " planned bootstrap replicates ",
        "were completed successfully. "
      )
    } else {
      paste0(
        format_integer(note_values$bootstrap_success), " of ",
        format_integer(note_values$bootstrap_target),
        " planned bootstrap replicates were completed successfully. "
      )
    },
    "Case-first probability denotes the fitted conditional probability that the first member of an ",
    "ordered case-control comparison was the case day. Outcome abbreviations are defined in Table S1. ",
    "Detailed definitions and calculation procedures for the model diagnostic metrics are provided ",
    "in Section A.25 of the Supplementary Methods. AIPW=augmented inverse probability weighting."
  )
  
  list(table = table, note = note)
}

# ================================================================================================
# 4. Extreme-temperature-event diagnostics
# ================================================================================================

build_extreme_table <- function(directory) {
  key <- c("Outcome", "Event_type", "Event_definition")
  
  aipw <- read_latest_by_suffix(
    directory = directory,
    suffix = "_diagnostic_AIPW_summary.csv",
    label = "AIPW summary",
    analysis_family = "extreme",
    key_columns = key
  )
  
  outcome <- read_latest_by_suffix(
    directory = directory,
    suffix = "_diagnostic_outcome_folds.csv",
    label = "outcome-fold diagnostic",
    analysis_family = "extreme",
    key_columns = key
  )
  
  rr <- read_latest_by_suffix(
    directory = directory,
    suffix = "_RR_results.csv",
    label = "RR/bootstrap result",
    analysis_family = "extreme",
    key_columns = key
  )
  
  check_columns(
    aipw,
    c(
      key, "N_matched_sets", "N_case_control_comparisons",
      "Proportion_informative_comparisons", "Target_converged", "Used_BFGS_fallback",
      "Score_norm", "Case_probability_p01", "Proportion_case_probability_lt_0_05"
    ),
    "Extreme-temperature-event AIPW summaries"
  )
  check_columns(
    outcome,
    c(key, "Fold", "Converged"),
    "Extreme-temperature-event outcome-fold diagnostics"
  )
  
  duplicate_keys <- aipw[, .N, by = key][N > 1L]
  if (nrow(duplicate_keys) > 0L) {
    stop(
      paste0(
        "More than one AIPW summary row remains for at least one extreme-event model. ",
        "Inspect the selected latest summary files."
      ),
      call. = FALSE
    )
  }
  
  bootstrap <- summarize_bootstrap_completion(rr, key)
  
  table <- aipw[, .(
    Outcome,
    `Event type` = fifelse(
      Event_type == "heatwave", "Heatwave",
      fifelse(Event_type == "cold_spell", "Cold spell", Event_type)
    ),
    `Event definition` = Event_definition,
    `Matched sets` = as.integer(N_matched_sets),
    `Case-control comparisons` = as.integer(N_case_control_comparisons),
    `Informative comparisons, %` = round(
      100 * as.numeric(Proportion_informative_comparisons),
      INFORMATIVE_PERCENT_DIGITS
    ),
    `Case-first probability, 1st percentile` = round(
      as.numeric(Case_probability_p01),
      CASE_PROBABILITY_DIGITS
    )
  )]
  
  table[, Outcome := publication_outcome_label(Outcome)]
  table <- apply_outcome_order(table, event_table = TRUE)
  
  # Internal checks summarized in the publication table note.
  fold_status <- outcome[, .(
    N_folds = .N,
    N_converged = sum(as_logical_safe(Converged) %in% TRUE, na.rm = TRUE)
  ), by = key]
  
  target_converged <- as_logical_safe(aipw$Target_converged)
  bfgs_used <- as_logical_safe(aipw$Used_BFGS_fallback)
  
  note_values <- list(
    n_folds = sum(fold_status$N_folds),
    n_folds_converged = sum(fold_status$N_converged),
    n_models = nrow(aipw),
    n_targets_converged = sum(target_converged %in% TRUE, na.rm = TRUE),
    n_without_bfgs = sum(bfgs_used %in% FALSE, na.rm = TRUE),
    max_score_norm = suppressWarnings(max(abs(as.numeric(aipw$Score_norm)), na.rm = TRUE)),
    max_low_probability_percent = suppressWarnings(
      100 * max(as.numeric(aipw$Proportion_case_probability_lt_0_05), na.rm = TRUE)
    ),
    bootstrap_success = sum(bootstrap$Bootstrap_success, na.rm = TRUE),
    bootstrap_target = sum(bootstrap$Bootstrap_target, na.rm = TRUE),
    all_bootstrap_successful = all(
      is.finite(bootstrap$Bootstrap_target) &
        is.finite(bootstrap$Bootstrap_success) &
        bootstrap$Bootstrap_success >= bootstrap$Bootstrap_target
    )
  )
  
  note <- paste0(
    "Heatwave and cold-spell models were fitted separately. All ",
    note_values$n_folds_converged, " of ", note_values$n_folds,
    " cross-fitted conditional logistic outcome-model folds converged. All ",
    note_values$n_targets_converged, " of ", note_values$n_models,
    " generalized AIPW estimating equations converged; ", note_values$n_without_bfgs,
    " were solved without BFGS fallback. The maximum final score norm was ",
    format_score(note_values$max_score_norm), ". The proportion of fitted case-first probabilities ",
    "below 0.05 was ", format_decimal(note_values$max_low_probability_percent, 2L),
    "% in the most extreme model. ",
    if (note_values$all_bootstrap_successful) {
      paste0(
        "All ", format_integer(note_values$bootstrap_success), " planned bootstrap replicates ",
        "were completed successfully. "
      )
    } else {
      paste0(
        format_integer(note_values$bootstrap_success), " of ",
        format_integer(note_values$bootstrap_target),
        " planned bootstrap replicates were completed successfully. "
      )
    },
    "Informative comparisons were defined as case-control comparisons with a non-zero event-history ",
    "contrast. Case-first probability denotes the fitted conditional probability that the first member ",
    "of an ordered case-control comparison was the case day. Event definitions are expressed as event ",
    "type, temperature-percentile threshold, and minimum consecutive duration; for example, ",
    "HW_P97p5_D3 denotes a heatwave above the 97.5th percentile lasting at least 3 consecutive days. ",
    "Outcome abbreviations are defined in Table S1. Detailed definitions and calculation procedures ",
    "for the model diagnostic metrics are provided in Section B.25 of the Supplementary Methods. ",
    "AIPW=augmented inverse probability weighting."
  )
  
  list(table = table, note = note)
}

# ================================================================================================
# 5. Publication-ready Excel output
# ================================================================================================

write_publication_table <- function(
    table,
    output_file,
    title,
    note,
    integer_columns,
    decimal_formats
) {
  workbook <- createWorkbook(creator = "Reviewer-ready generalized AIPW diagnostic script")
  sheet <- "Model diagnostics"
  addWorksheet(workbook, sheet, gridLines = FALSE)
  
  title_style <- createStyle(
    fontSize = 12,
    textDecoration = "bold",
    valign = "center"
  )
  header_style <- createStyle(
    textDecoration = "bold",
    halign = "center",
    valign = "center",
    wrapText = TRUE,
    border = c("Top", "Bottom"),
    borderStyle = "medium"
  )
  body_style <- createStyle(
    valign = "center",
    wrapText = TRUE,
    border = "Bottom",
    borderStyle = "hair"
  )
  note_style <- createStyle(
    fontSize = 9,
    textDecoration = "italic",
    valign = "top",
    wrapText = TRUE
  )
  integer_style <- createStyle(numFmt = "#,##0")
  
  n_columns <- ncol(table)
  n_rows <- nrow(table)
  
  writeData(workbook, sheet, title, startRow = 1L, startCol = 1L)
  mergeCells(workbook, sheet, cols = 1:n_columns, rows = 1L)
  addStyle(
    workbook, sheet, title_style,
    rows = 1L, cols = 1:n_columns,
    gridExpand = TRUE, stack = TRUE
  )
  
  writeData(
    workbook, sheet, table,
    startRow = 3L,
    startCol = 1L,
    colNames = TRUE,
    rowNames = FALSE,
    withFilter = FALSE,
    headerStyle = header_style
  )
  
  if (n_rows > 0L) {
    addStyle(
      workbook, sheet, body_style,
      rows = 4:(n_rows + 3L), cols = 1:n_columns,
      gridExpand = TRUE, stack = TRUE
    )
  }
  
  for (column_name in integer_columns) {
    if (column_name %in% names(table) && n_rows > 0L) {
      column_index <- match(column_name, names(table))
      addStyle(
        workbook, sheet, integer_style,
        rows = 4:(n_rows + 3L), cols = column_index,
        gridExpand = TRUE, stack = TRUE
      )
    }
  }
  
  for (column_name in names(decimal_formats)) {
    if (column_name %in% names(table) && n_rows > 0L) {
      column_index <- match(column_name, names(table))
      decimal_style <- createStyle(numFmt = decimal_formats[[column_name]])
      addStyle(
        workbook, sheet, decimal_style,
        rows = 4:(n_rows + 3L), cols = column_index,
        gridExpand = TRUE, stack = TRUE
      )
    }
  }
  
  note_row <- n_rows + 5L
  writeData(workbook, sheet, paste0("Note: ", note), startRow = note_row, startCol = 1L)
  mergeCells(workbook, sheet, cols = 1:n_columns, rows = note_row)
  addStyle(
    workbook, sheet, note_style,
    rows = note_row, cols = 1:n_columns,
    gridExpand = TRUE, stack = TRUE
  )
  
  freezePane(workbook, sheet, firstActiveRow = 4L, firstActiveCol = 2L)
  setRowHeights(workbook, sheet, rows = 1L, heights = 24)
  setRowHeights(workbook, sheet, rows = 3L, heights = 42)
  setRowHeights(workbook, sheet, rows = note_row, heights = 90)
  
  setColWidths(workbook, sheet, cols = 1:n_columns, widths = "auto")
  setColWidths(workbook, sheet, cols = 1L, widths = 14)
  if ("Event type" %in% names(table)) {
    setColWidths(workbook, sheet, cols = match("Event type", names(table)), widths = 13)
  }
  if ("Event definition" %in% names(table)) {
    setColWidths(workbook, sheet, cols = match("Event definition", names(table)), widths = 20)
  }
  if ("Case-first probability, 1st percentile" %in% names(table)) {
    setColWidths(
      workbook, sheet,
      cols = match("Case-first probability, 1st percentile", names(table)),
      widths = 22
    )
  }
  if ("Informative comparisons, %" %in% names(table)) {
    setColWidths(
      workbook, sheet,
      cols = match("Informative comparisons, %", names(table)),
      widths = 19
    )
  }
  
  pageSetup(
    workbook, sheet,
    orientation = "landscape",
    paperSize = 9,
    fitToWidth = 1,
    fitToHeight = 0
  )
  printSetup <- try(
    setPrintArea(workbook, sheet, cols = 1:n_columns, rows = 1:note_row),
    silent = TRUE
  )
  
  saveWorkbook(workbook, output_file, overwrite = TRUE)
  invisible(output_file)
}

# ================================================================================================
# 6. Build and export the final tables
# ================================================================================================

nonoptimal <- build_nonoptimal_table(NONOPTIMAL_RESULTS_DIR)
extreme <- build_extreme_table(EXTREME_RESULTS_DIR)

write_publication_table(
  table = nonoptimal$table,
  output_file = NONOPTIMAL_OUTPUT,
  title = paste0(
    "Core model diagnostics for generalized augmented inverse probability weighting analyses ",
    "of non-optimal daily temperature"
  ),
  note = nonoptimal$note,
  integer_columns = c("Matched sets", "Case-control comparisons"),
  decimal_formats = c(
    "Case-first probability, 1st percentile" = "0.000"
  )
)

write_publication_table(
  table = extreme$table,
  output_file = EXTREME_OUTPUT,
  title = paste0(
    "Core model diagnostics for generalized augmented inverse probability weighting analyses ",
    "of heatwaves and cold spells"
  ),
  note = extreme$note,
  integer_columns = c("Matched sets", "Case-control comparisons"),
  decimal_formats = c(
    "Informative comparisons, %" = "0.00",
    "Case-first probability, 1st percentile" = "0.000"
  )
)

message("Core diagnostic aggregation completed successfully.")
message(sprintf("Non-optimal-temperature workbook: %s", NONOPTIMAL_OUTPUT))
message(sprintf("Extreme-temperature-event workbook: %s", EXTREME_OUTPUT))