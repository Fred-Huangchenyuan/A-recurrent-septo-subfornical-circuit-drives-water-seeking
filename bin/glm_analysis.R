# This script runs event-level GLM model selection for ca_signal and saves model metrics
# plus coefficient-stability summaries across predictor subsets.
# @author Huang Chenyuan, Xu Lingyu
# @date 2026-07-18

library(data.table)
library(tidyverse)

set.seed(234)
setwd("") # Run this script from the project root directory.
output_dir <- file.path("") # Change this to the directory where the output files will be saved
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


# 1. Read and prepare files.
DATA_PATH <- file.path(data_dir, "harmonized_data.csv")
combined <- fread(DATA_PATH)
allowed <- c("day1","day2","day3","final")
bad <- setdiff(unique(combined$day_number), allowed)
if (length(bad) > 0) {
  stop(sprintf("day_number unclear: %s", paste(bad, collapse = ", ")))
}

required_cols <- c(
  "ca_signal", "dist_to_water", "speed", "digging_number",
  "day_number", "digging_status", "drinking_status", "cos_angle", "sin_angle", "event"
)
missing_cols <- setdiff(required_cols, names(combined))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

combined <- combined %>%
  as.data.frame() %>%
  mutate(
    day_1     = as.integer(ifelse(day_number == "day1",  1, 0)),
    day_2     = as.integer(ifelse(day_number == "day2",  1, 0)),
    day_3     = as.integer(ifelse(day_number == "day3",  1, 0)),
    day_final = as.integer(ifelse(day_number == "final", 1, 0))
  )

combined_before <- combined %>%
  filter(digging_status == 0 & drinking_status == 0) %>%
  filter(complete.cases(dplyr::select(., all_of(required_cols))))

# NOTE : no standardization of continuous predictors -- raw scale kept.

num_iterations <- 100

all_predictors <- c("dist_to_water", "speed", "digging_number",
                    "day_2", "day_3", "day_final", "cos_angle", "sin_angle")

# 38 model combinations.
model_combinations <- list(
  all_predictors,
  "digging_number",
  "dist_to_water",
  "speed",
  "day_2",
  "day_3",
  "day_final",
  "cos_angle",
  "sin_angle",
  c("dist_to_water", "speed"),
  c("dist_to_water", "day_2"),
  c("dist_to_water", "day_3"),
  c("dist_to_water", "day_final"),
  c("dist_to_water", "cos_angle", "sin_angle"),
  c("speed", "day_2"),
  c("speed", "day_3"),
  c("speed", "day_final"),
  c("speed", "cos_angle", "sin_angle"),
  c("day_2", "day_3"),
  c("day_2", "day_final"),
  c("day_3", "day_final"),
  c("dist_to_water", "speed", "day_2"),
  c("dist_to_water", "speed", "day_3"),
  c("dist_to_water", "speed", "day_final"),
  c("dist_to_water", "day_2", "day_3"),
  c("dist_to_water", "day_2", "day_final"),
  c("dist_to_water", "day_3", "day_final"),
  c("speed", "day_2", "day_3"),
  c("speed", "day_2", "day_final"),
  c("speed", "day_3", "day_final"),
  c("day_2", "day_3", "day_final"),
  c("dist_to_water", "speed", "day_2", "day_3"),
  c("dist_to_water", "speed", "day_2", "day_final"),
  c("dist_to_water", "speed", "day_3", "day_final"),
  c("dist_to_water", "day_2", "day_3", "day_final"),
  c("speed", "day_2", "day_3", "day_final"),
  c("dist_to_water", "speed", "day_2", "day_3", "day_final"),
  c("dist_to_water", "speed", "day_2", "day_3", "day_final", "cos_angle", "sin_angle")
)
stopifnot(length(model_combinations) == 38)

create_formula <- function(pred_vars) {
  paste("ca_signal ~", paste(pred_vars, collapse = " + "))
}

group_labels     <- sort(unique(combined_before$event))
num_train_groups <- round(0.8 * length(group_labels))

# 2. Repeated event-level train/test splits for model comparison.
for (model_idx in seq_along(model_combinations)) {
  
  set.seed(234)
  predictors <- model_combinations[[model_idx]]
  
  metric_cols   <- c("mse_test", "rsq", "aic_train")
  metric_matrix <- matrix(NA, nrow = num_iterations, ncol = length(metric_cols),
                          dimnames = list(NULL, metric_cols))
  
  coef_cols   <- c("(Intercept)", predictors)
  coef_matrix <- matrix(NA, nrow = num_iterations, ncol = length(coef_cols),
                        dimnames = list(NULL, coef_cols))
  
  for (i in 1:num_iterations) {
    # Event-level split avoids mixing frames from the same event across train and test.
    train_groups <- sample(group_labels, num_train_groups, replace = FALSE)
    train_idx    <- combined_before$event %in% train_groups
    train_data   <- combined_before[train_idx, ]
    test_data    <- combined_before[!train_idx, ]
    
    model_formula <- as.formula(create_formula(predictors))
    full_model    <- glm(model_formula, data = train_data,
                         family = gaussian(link = "identity"))
    
    pred   <- predict(full_model, newdata = test_data)
    metric_matrix[i, "mse_test"]  <- mean((pred - test_data$ca_signal)^2)
    # R2 = 1 - deviance/null.deviance (train, deviance-based)
    metric_matrix[i, "rsq"]       <- 1 - full_model$deviance / full_model$null.deviance
    metric_matrix[i, "aic_train"] <- AIC(full_model)
    
    cf <- coef(full_model)
    for (nm in coef_cols) if (nm %in% names(cf)) coef_matrix[i, nm] <- cf[nm]
  }
  
  # ===== metrics summary (Mean / SD / SEM, incl. AIC SEM) =====
  metric_summary <- data.frame(
    Metric = c("MSE_test", "R2", "AIC_train"),
    Mean   = colMeans(metric_matrix, na.rm = TRUE),
    SD     = apply(metric_matrix, 2, sd, na.rm = TRUE),
    SEM    = apply(metric_matrix, 2, function(x) sd(x, na.rm = TRUE)/sqrt(sum(!is.na(x)))),
    row.names = NULL
  )
  
  # Full-sample p-values are frame-level and exploratory because frames within event are correlated.
  full_fit  <- glm(as.formula(create_formula(predictors)),
                   data = combined_before,
                   family = gaussian(link = "identity"))
  cs        <- summary(full_fit)$coefficients   # cols: Estimate, Std. Error, t value, Pr(>|t|)
  p_full    <- setNames(rep(NA_real_, length(coef_cols)), coef_cols)
  est_full  <- setNames(rep(NA_real_, length(coef_cols)), coef_cols)
  for (nm in coef_cols) {
    if (nm %in% rownames(cs)) {
      est_full[nm] <- cs[nm, "Estimate"]
      p_full[nm]   <- cs[nm, "Pr(>|t|)"]
    }
  }
  
  # ===== coefficient summary (resampling stability + full-sample p) =====
  coef_summary <- data.frame(
    Term             = coef_cols,
    Coef_Mean        = colMeans(coef_matrix, na.rm = TRUE),     # mean over 100 resamples
    Coef_SD          = apply(coef_matrix, 2, sd, na.rm = TRUE),
    Coef_SEM         = apply(coef_matrix, 2, function(x) sd(x, na.rm = TRUE)/sqrt(sum(!is.na(x)))),
    CI_lower         = apply(coef_matrix, 2, quantile, probs = 0.025, na.rm = TRUE),
    CI_upper         = apply(coef_matrix, 2, quantile, probs = 0.975, na.rm = TRUE),
    Sign_consistency = apply(coef_matrix, 2, function(x) {
      x <- x[!is.na(x)]
      if (length(x) == 0) return(NA)
      max(mean(x > 0), mean(x < 0))
    }),
    Coef_full   = est_full[coef_cols],   # full-sample estimate
    p_full      = p_full[coef_cols],     # <-- classic p-value on full sample
    row.names = NULL
  )
  coef_summary$CI_excludes_0 <- with(coef_summary, (CI_lower > 0) | (CI_upper < 0))
  coef_summary$sig_full      <- coef_summary$p_full < 0.05   # exploratory frame-level significance flag
  
  model_name <- paste0("model_", model_idx)
  write.csv(metric_summary,
            file = file.path(output_dir, paste0(model_name, "_metrics.csv")),
            row.names = FALSE)
  write.csv(coef_summary,
            file = file.path(output_dir, paste0(model_name, "_coefficients.csv")),
            row.names = FALSE)
}

# 3. Overall comparison table across all models.
all_terms     <- c("(Intercept)", unique(unlist(model_combinations)))
coef_cols_out <- paste0("Coef_",    all_terms)
sem_cols_out  <- paste0("CoefSEM_", all_terms)
p_cols_out    <- paste0("pfull_",   all_terms)

summary_df <- data.frame(
  Model_ID = seq_along(model_combinations),
  Formula  = sapply(model_combinations, function(x) paste(x, collapse = " + ")),
  MSE_test_Mean = NA, MSE_test_SEM = NA,
  R2_Mean  = NA, R2_SEM  = NA,
  AIC_train_Mean = NA, AIC_train_SEM = NA,
  stringsAsFactors = FALSE
)
for (cc in c(coef_cols_out, sem_cols_out, p_cols_out)) summary_df[[cc]] <- NA_real_

for (model_idx in seq_along(model_combinations)) {
  f <- file.path(output_dir, paste0("model_", model_idx, "_metrics.csv"))
  if (file.exists(f)) {
    m <- read.csv(f)
    summary_df[model_idx, "MSE_test_Mean"]  <- m$Mean[m$Metric == "MSE_test"]
    summary_df[model_idx, "MSE_test_SEM"]   <- m$SEM [m$Metric == "MSE_test"]
    summary_df[model_idx, "R2_Mean"]        <- m$Mean[m$Metric == "R2"]
    summary_df[model_idx, "R2_SEM"]         <- m$SEM [m$Metric == "R2"]
    summary_df[model_idx, "AIC_train_Mean"] <- m$Mean[m$Metric == "AIC_train"]
    summary_df[model_idx, "AIC_train_SEM"]  <- m$SEM [m$Metric == "AIC_train"]
  }
  # per-term coef (resampling mean), its SEM, and full-sample p (NA if unused)
  cf_file <- file.path(output_dir, paste0("model_", model_idx, "_coefficients.csv"))
  if (file.exists(cf_file)) {
    cdf <- read.csv(cf_file, check.names = FALSE)
    for (k in seq_along(all_terms)) {
      r <- which(cdf$Term == all_terms[k])
      if (length(r) == 1) {
        summary_df[model_idx, coef_cols_out[k]] <- cdf$Coef_Mean[r]
        summary_df[model_idx, sem_cols_out[k]]  <- cdf$Coef_SEM[r]
        summary_df[model_idx, p_cols_out[k]]    <- cdf$p_full[r]
      }
    }
  }
}
# Rank primarily by held-out event MSE; train R2 is kept as a descriptive fit metric.
summary_df <- summary_df[order(summary_df$MSE_test_Mean, -summary_df$R2_Mean), ]

write.csv(summary_df,
          file = file.path(output_dir, "models_comparison_summary.csv"),
          row.names = FALSE)
