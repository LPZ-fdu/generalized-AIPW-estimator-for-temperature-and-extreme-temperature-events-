# Generalized AIPW for Temperature-Related Mortality in Time-Stratified Case-Crossover Studies

## Overview

This repository contains the R code accompanying the manuscript:

> **Cause-specific mortality burden of non-optimal daily temperature and extreme temperature events in China: A nationwide assessment using causal machine learning**

The study developed a generalized augmented inverse probability weighting (AIPW) framework for individual-level, time-stratified case-crossover studies. The framework combines:

- a prospective conditional outcome model;
- a retrospective exposure conditional-mean model;
- matched-set-level cross-fitting; and
- a generalized doubly robust estimating equation.

The code supports analyses of:

1. continuous heat- and cold-side deviations from the outcome-specific minimum mortality temperature;
2. binary recent histories of heatwaves and cold spells;
3. mortality risks and attributable burdens;
4. Monte Carlo simulations evaluating estimator performance and double robustness; and
5. model diagnostics for the empirical analyses.

The empirical application used nationally representative mortality surveillance data from mainland China during 2013–2019. Individual-level mortality data are not included in this repository.

---

## Methodological framework

### Non-optimal daily temperature

For each mortality outcome, daily temperature is divided into two continuous exposure components relative to the outcome-specific minimum mortality temperature:

- a heat-side deviation above the minimum mortality temperature; and
- a cold-side deviation below the minimum mortality temperature.

In the primary analysis, heat exposure is summarised over lag 0–3 days and cold exposure over lag 0–14 days. The corresponding coefficients are estimated jointly using the generalized AIPW estimating equation.

Because the heat and cold summaries use different lag windows, both components can be positive for the same death. Total attributable burden is therefore calculated from their joint relative-rate contribution. A two-component Shapley decomposition is used when separate heat- and cold-related burdens are required.

### Extreme-temperature events

Heatwaves and cold spells are evaluated as recent event histories after adjustment for the background association with daily temperature.

The primary event-history windows are:

- heatwaves: at least one qualifying event day within lag 0–10 days;
- cold spells: at least one qualifying event day within lag 0–21 days.

Heatwave analyses are conducted during the warm season and cold-spell analyses during the cool season. The resulting rate ratio represents the additional association of recent extreme-event exposure beyond the prespecified background heat- and cold-side temperature components.

### Double robustness

Under the stated identification assumptions and a correctly specified structural effect model, the generalized AIPW estimator is consistent when either:

1. the prospective conditional outcome nuisance model is correctly specified; or
2. the retrospective exposure conditional-mean nuisance model is correctly specified.

The estimator is not expected to retain this protection when both nuisance models are misspecified.

---

## Repository structure

| File | Description |
|---|---|
| `code_AIPW.R` | Main empirical analysis for non-optimal daily temperature. Estimates heat- and cold-related mortality coefficients, relative risks, attributable fractions, annual burdens, and bootstrap uncertainty intervals. |
| `code_AIPW_event.R` | Main empirical analysis for heatwaves and cold spells. Estimates the additional mortality association and attributable burden of recent extreme-event histories after adjustment for background daily temperature. |
| `code_modelDIAGNOSTIC.R` | Summarises diagnostic information from the non-optimal-temperature and extreme-event analyses, including model convergence, estimating-equation performance, fitted-probability stability, cross-fitting results, and bootstrap completion. |
| `code_simulated_data.R` | Generates simulated one-case multiple-control matched risk sets for the non-optimal-temperature analysis. The simulated data include correlated heat- and cold-side exposures, time-varying covariates, structural mortality effects, and alternative nuisance-model specification scenarios. |
| `code_simulated_analysis.R` | Applies the competing estimators to the simulated non-optimal-temperature datasets and summarises coefficient bias, attributable-fraction bias, confidence-interval coverage, numerical convergence, and related performance measures. |
| `code_simulated_data_event.R` | Generates simulated matched risk sets for the extreme-event analysis, including a binary recent event-history exposure, background heat- and cold-side temperature components, time-varying covariates, and alternative nuisance-model specification scenarios. |
| `code_simulated_analysis_event.R` | Evaluates the generalized AIPW and comparison estimators for the simulated extreme-event datasets and summarises bias, coverage, convergence, and attributable-burden performance. |

---

## Recommended workflow

### Empirical analyses

The main empirical scripts can be run independently once the required input data and analysis parameters have been prepared.

1. Run `code_AIPW.R` for non-optimal daily temperature.
2. Run `code_AIPW_event.R` separately for heatwaves and cold spells.
3. Run `code_modelDIAGNOSTIC.R` after the empirical analyses have finished.

The diagnostic script expects the result directories produced by the two main empirical workflows.

### Simulation analyses

The simulation data-generation and analysis scripts are deliberately separated.

For non-optimal daily temperature:

```text
code_simulated_data.R
        ↓
code_simulated_analysis.R
