# This script takes the posterior + flow-parameter tables from the Python fit and turns them into
# the core figures/tables for the paper: PCA embedding, energy/slowness landscape,
# attractor geometry, line-attractor overlays, and phase-vs-state enrichment tests.
# @author Huang Chenyuan, Xu Lingyu
# @date 2026-07-18

library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(plotly)
library(htmlwidgets)
library(FNN)
library(MASS)
library(viridisLite)
library(grid)

setwd("") # Run this script from the project root directory.

# 1. Configurations of the parameter and input/output for the analysis.
DEFAULT_CONFIG <- list(
  behavior_file = "./data/harmonized_data.csv",
  posterior_file = "trained_rslds_posterior_table.csv",
  flow_file = "trained_rslds_flow_parameters.csv",
  selected_events = NULL,
  auto_event_count = 1,
  n_grid = 160,
  n_grid_e = 160,
  knn_state_k = 10,
  knn_flow_k = 20,
  loess_span_w = 0.25,
  loess_span_energy = 0.4,
  energy_clip = c(0, 0.99),
  # Heuristic dynamics thresholds for classifying slow, stable, and near-singular states.
  slow_thr = 0.95,
  stable_thr = 1.00,
  cond_thr = 1e8,
  field_mode = "attractor",
  data_quantile = 0.99,
  # Heuristic distance cutoff for hiding grid regions far from the observed latent cloud.
  mask_dist_q = 0.95,
  arrow_stride = 12,
  arrow_scale = 0.80,
  min_arrow_len = 0.2,
  arrow_mag_quantile = 0.05,
  show_energy_background = FALSE,
  force_plane_as_line = TRUE,
  use_curved_attractor_line = TRUE,
  line_attractor_states = integer(0),
  event_candidates = c("event_id", "trial_id", "trial", "event", "session")
)

RUN_CONFIGS <- list(
  list(
    label = "Day 1",
    result_dir = "./new_result_44_masked_speed_test_123_day1",
    output_prefix = "flow_123_44_day1_v23",
    selected_events = c(350), # The representative event for Day 1, used for the main figures.
    line_attractor_states = c(2) # Change according to the actual line-attractor states for Day 1.
  ),
  list(
    label = "Day 2",
    result_dir = "./new_result_44_masked_speed_test_123_day2",
    output_prefix = "flow_123_44_day2_v23",
    selected_events = c(103), # The representative event for Day 2, used for the main figures.
    line_attractor_states = c(2) # Change according to the actual line-attractor states for Day 2.
  ),
  list(
    label = "Day 3",
    result_dir = "./new_result_44_masked_speed_test_123_day3",
    output_prefix = "flow_123_44_day3_v23",
    selected_events = c(123), # The representative event for Day 3, used for the main figures.
    line_attractor_states = c(0) # Change according to the actual line-attractor states for Day 3.
  ),
  list(
    label = "Final",
    result_dir = "./new_result_44_masked_speed_test_123_final",
    output_prefix = "flow_123_44_final_v23",
    selected_events = c(329), # The representative event for the final day, used for the main figures.
    line_attractor_states = c(2) # Change according to the actual line-attractor states for the final day.
  )
)

coolwarm_colorscale <- list(
  c(0.00, "#3b4cc0"),
  c(0.15, "#6788ee"),
  c(0.30, "#9abbff"),
  c(0.45, "#c9d7f0"),
  c(0.55, "#edd1c2"),
  c(0.70, "#f7a889"),
  c(0.85, "#e26952"),
  c(1.00, "#b40426")
)

build_paths <- function(config) {
  list(
    posterior = file.path(config$result_dir, config$posterior_file),
    flow = file.path(config$result_dir, config$flow_file),
    html_3d = paste0(config$output_prefix, ".html"),
    pdf_2d = paste0(config$output_prefix, "_2d.pdf"),
    svg_2d = paste0(config$output_prefix, "_2d.svg"),
    png_2d = paste0(config$output_prefix, "_2d.png"),
    diagnostics_csv = paste0(config$output_prefix, "_diagnostics.csv"),
    state_summary_csv = paste0(config$output_prefix, "_state_summary.csv"),
    line_score_csv = paste0(config$output_prefix, "_line_attractor_scores.csv"),
    geometry_score_csv = paste0(config$output_prefix, "_geometry_scores.csv"),
    phase_state_overall_csv = paste0(config$output_prefix, "_phase_state_overall_fisher.csv"),
    phase_state_pairwise_csv = paste0(config$output_prefix, "_phase_state_pairwise_fisher.csv"),
    phase_state_table_csv = paste0(config$output_prefix, "_phase_state_contingency.csv"),
    state_encoding_or_png = paste0(config$output_prefix, "_state_encoding_oddsratio.png"),
    state_encoding_or_pdf = paste0(config$output_prefix, "_state_encoding_oddsratio.pdf"),
    state_encoding_or_svg = paste0(config$output_prefix, "_state_encoding_oddsratio.svg")
  )
}

# 2. Data loading and preparation functions.
load_rslds_data <- function(config) {
  paths <- build_paths(config)
  list(
    post = as.data.frame(data.table::fread(paths$posterior)),
    fpar = as.data.frame(data.table::fread(paths$flow)),
    behavior = as.data.frame(data.table::fread(config$behavior_file)),
    paths = paths
  )
}

infer_event_col <- function(post, candidates) {
  # Posterior tables name the event column differently across versions.
  intersect(candidates, colnames(post))[1]
}

make_join_key <- function(x) {
  # Format timestamps to a fixed string for robust event-time joins.
  x_num <- suppressWarnings(as.numeric(x))
  ifelse(is.finite(x_num), sprintf("%.10f", x_num), as.character(x))
}

make_behavior_status_label <- function(timestamp, digging_status, ever_dug) {
  # Coarse per-frame behavior label, only used for coloring the paths.
  dplyr::case_when(
    timestamp < 0 ~ "timestamp < 0",
    digging_status == 1L ~ "digging",
    timestamp >= 0 & ever_dug ~ "post-dig, no digging",
    TRUE ~ "timestamp >= 0, no digging"
  )
}

make_event_phase_label <- function(timestamp, digging_status, ever_dug) {
  # Establish per-frame phase label used in the phase-state enrichment tests.
  dplyr::case_when(
    !is.na(timestamp) & timestamp >= -2 & timestamp < 0 ~ "pre_approach",
    digging_status == 1L ~ "digging",
    !is.na(timestamp) & timestamp >= 0 & !ever_dug & digging_status != 1L ~ "approach",
    ever_dug & digging_status != 1L ~ "after_digging",
    TRUE ~ NA_character_
  )
}

build_behavior_lookup <- function(behavior) {
  # Build a unique event-timestamp lookup for joining behavior labels to posterior frames.
  behavior %>%
    mutate(
      event_key = as.character(event),
      timestamp_key = make_join_key(timestamp),
      timestamp = suppressWarnings(as.numeric(timestamp)),
      digging_status = suppressWarnings(as.integer(digging_status)),
      drinking_status = suppressWarnings(as.integer(drinking_status))
    ) %>%
    dplyr::select(event_key, timestamp_key, timestamp, digging_status, drinking_status) %>%
    distinct(event_key, timestamp_key, .keep_all = TRUE)
}

make_behavior_palette <- function() {
  c(
    "timestamp < 0" = "#000000",
    "timestamp >= 0, no digging" = "#0B6E4F",
    "digging" = "#8B1E1E",
    "post-dig, no digging" = "#0B3C8C"
  )
}

make_state_palette <- function(state_levels) {
  base_colors <- c(
    "#0B6E4F",
    "#0B3C8C",
    "#8B1E1E",
    "#2CA02C",
    "#FF7F00",
    "#9467BD",
    "#17BECF",
    "#8C564B"
  )

  if (length(state_levels) > length(base_colors)) {
    base_colors <- c(base_colors, viridisLite::turbo(length(state_levels) - length(base_colors)))
  }

  setNames(base_colors[seq_along(state_levels)], as.character(state_levels))
}

# 3. Model building and analysis functions.
build_state_models <- function(fpar, state_levels) {
  # Pull each state's A matrix + b vector back out of the flow-parameter rows.
  model_list <- vector("list", length(state_levels))
  names(model_list) <- as.character(state_levels)

  for (state in state_levels) {
    row <- fpar %>% filter(state_index == state)
    model_list[[as.character(state)]] <- list(
      A = matrix(
        c(
          row$A_1_1, row$A_1_2, row$A_1_3, row$A_1_4,
          row$A_2_1, row$A_2_2, row$A_2_3, row$A_2_4,
          row$A_3_1, row$A_3_2, row$A_3_3, row$A_3_4,
          row$A_4_1, row$A_4_2, row$A_4_3, row$A_4_4
        ),
        nrow = 4,
        byrow = TRUE
      ),
      b = c(row$bias_latent_1, row$bias_latent_2, row$bias_latent_3, row$bias_latent_4)
    )
  }

  model_list
}

safe_find_interval <- function(x, grid_seq) {
  # Find safe grid-cell indices for bilinear interpolation without edge overflow.
  idx <- findInterval(x, grid_seq, all.inside = TRUE)
  pmin(idx, length(grid_seq) - 1)
}

compute_fixed_points <- function(state_models, cond_thr) {
  # Calculate fixed point per state = solve (I - A) x = b.
  result <- lapply(names(state_models), function(state_name) {
    model <- state_models[[state_name]]
    matrix_term <- diag(4) - model$A
    cond_value <- kappa(matrix_term)
    fp <- tryCatch(as.numeric(solve(matrix_term, model$b)), error = function(e) rep(NA_real_, 4))

    data.frame(
      state = as.integer(state_name),
      fp_l1 = fp[1],
      fp_l2 = fp[2],
      fp_l3 = fp[3],
      fp_l4 = fp[4],
      cond_I_minus_A = cond_value,
      singular = cond_value > cond_thr
    )
  })

  dplyr::bind_rows(result)
}

project_with_pc <- function(pc, x, source_cols = NULL) {
  # Project latent coords into the shared PCA space.
  x_df <- as.data.frame(x)
  if (!is.null(source_cols)) {
    x_df <- x_df[, source_cols, drop = FALSE]
  }

  x_mat <- as.matrix(x_df)
  centered_mat <- sweep(x_mat, 2, pc$center, FUN = "-")
  projected <- centered_mat %*% pc$rotation
  colnames(projected) <- colnames(pc$x)
  projected
}

prepare_analysis_context <- function(post, fpar, behavior, config) {
  # Bundle posterior, flow parameters, behavior lookup, PCA projection, and state dynamics.
  latent_mat <- as.matrix(post[, c("latent_1", "latent_2", "latent_3", "latent_4")])
  state_levels <- sort(unique(post$state_most_likely))
  state_models <- build_state_models(fpar, state_levels)
  pc <- prcomp(latent_mat, center = TRUE, scale. = FALSE)
  fixed_points <- compute_fixed_points(state_models, config$cond_thr)
  fixed_points_complete <- fixed_points %>% filter(stats::complete.cases(fp_l1, fp_l2, fp_l3, fp_l4))
  fp_pc <- if (nrow(fixed_points_complete) > 0) {
    project_with_pc(pc, fixed_points_complete[, c("fp_l1", "fp_l2", "fp_l3", "fp_l4")])[, 1:3, drop = FALSE]
  } else {
    matrix(numeric(0), ncol = 3)
  }

  list(
    post = post,
    fpar = fpar,
    X = latent_mat,
    zs = post$state_most_likely,
    state_levels = state_levels,
    state_models = state_models,
    state_palette = make_state_palette(state_levels),
    behavior_lookup = build_behavior_lookup(behavior),
    behavior_palette = make_behavior_palette(),
    pc = pc,
    U = pc$x[, 1],
    V = pc$x[, 2],
    W = pc$x[, 3],
    x_mean = colMeans(latent_mat),
    event_col = infer_event_col(post, config$event_candidates),
    fixed_points = fixed_points,
    fp_pc = fp_pc,
    config = config
  )
}

build_energy_surface <- function(ctx) {
  # Build a smoothed PC1-PC2 energy/slowness surface from fitted latent dynamics.
  config <- ctx$config
  u_seq <- seq(min(ctx$U), max(ctx$U), length.out = config$n_grid)
  v_seq <- seq(min(ctx$V), max(ctx$V), length.out = config$n_grid)
  grid2d <- expand.grid(u = u_seq, v = v_seq)

  train_df <- data.frame(u = ctx$U, v = ctx$V, w = ctx$W)
  w_model <- loess(
    w ~ u + v,
    data = train_df,
    span = config$loess_span_w,
    control = loess.control(surface = "direct")
  )
  w_hat <- predict(w_model, newdata = grid2d)
  w_hat[is.na(w_hat)] <- 0

  back_project <- function(u, v, w) {
    as.numeric(
      ctx$x_mean +
        ctx$pc$rotation[, 1] * u +
        ctx$pc$rotation[, 2] * v +
        ctx$pc$rotation[, 3] * w
    )
  }

  grid3d <- t(vapply(
    seq_len(nrow(grid2d)),
    function(i) back_project(grid2d$u[i], grid2d$v[i], w_hat[i]),
    numeric(4)
  ))

  knn_res <- FNN::get.knnx(ctx$X, grid3d, k = config$knn_state_k)
  state_at <- vapply(seq_len(nrow(knn_res$nn.index)), function(i) {
    ix <- knn_res$nn.index[i, ]
    neighbor_states <- ctx$zs[ix]
    as.integer(names(sort(table(neighbor_states), decreasing = TRUE))[1])
  }, integer(1))

  flow_speed_at <- function(x3, state_id) {
    model <- ctx$state_models[[as.character(state_id)]]
    dx <- as.numeric(model$A %*% x3 + model$b - x3)
    sqrt(sum(dx^2))
  }

  z_speed <- vapply(
    seq_len(nrow(grid3d)),
    function(i) flow_speed_at(grid3d[i, ], state_at[i]),
    numeric(1)
  )
  z_mat <- matrix(log(z_speed + 1e-8), config$n_grid, config$n_grid)

  smooth_df <- data.frame(u = grid2d$u, v = grid2d$v, z = as.vector(z_mat))
  smooth_fit <- loess(
    z ~ u + v,
    data = smooth_df,
    span = config$loess_span_energy,
    control = loess.control(surface = "direct")
  )
  z_sm <- matrix(predict(smooth_fit, newdata = grid2d), config$n_grid, config$n_grid)

  qlo <- quantile(z_sm, config$energy_clip[1], na.rm = TRUE)
  qhi <- quantile(z_sm, config$energy_clip[2], na.rm = TRUE)
  z_sm <- pmin(pmax(z_sm, qlo), qhi)

  interp_z <- function(u, v) {
    iu <- safe_find_interval(u, u_seq)
    iv <- safe_find_interval(v, v_seq)
    du <- (u - u_seq[iu]) / (u_seq[iu + 1] - u_seq[iu])
    dv <- (v - v_seq[iv]) / (v_seq[iv + 1] - v_seq[iv])

    (1 - du) * (1 - dv) * z_sm[cbind(iu, iv)] +
      du * (1 - dv) * z_sm[cbind(iu + 1, iv)] +
      (1 - du) * dv * z_sm[cbind(iu, iv + 1)] +
      du * dv * z_sm[cbind(iu + 1, iv + 1)]
  }

  list(
    u_seq = u_seq,
    v_seq = v_seq,
    grid2d = grid2d,
    grid3d = grid3d,
    state_at = state_at,
    z_speed = z_speed,
    z_mat = z_mat,
    z_sm = z_sm,
    qlo = qlo,
    qhi = qhi,
    interp_z = interp_z,
    fp_z_on_surf = if (nrow(ctx$fp_pc) > 0) interp_z(ctx$fp_pc[, 1], ctx$fp_pc[, 2]) + 0.15 else numeric(0),
    z_axis_range = c(qlo, qhi + 0.2)
  )
}

compute_plot_limits <- function(contexts, data_quantile) {
  # Compute plotting limits for the PCA projection based on the specified data quantile.
  all_u <- unlist(lapply(contexts, function(ctx) ctx$U), use.names = FALSE)
  all_v <- unlist(lapply(contexts, function(ctx) ctx$V), use.names = FALSE)
  u_lim <- quantile(all_u, c((1 - data_quantile) / 2, 1 - (1 - data_quantile) / 2), na.rm = TRUE)
  v_lim <- quantile(all_v, c((1 - data_quantile) / 2, 1 - (1 - data_quantile) / 2), na.rm = TRUE)

  list(
    u = u_lim + c(-0.05, 0.05) * diff(u_lim),
    v = v_lim + c(-0.15, 0.15) * diff(v_lim)
  )
}

resolve_selected_events <- function(ctx) {
  if (!is.null(ctx$config$selected_events) && length(ctx$config$selected_events) > 0) {
    return(ctx$config$selected_events)
  }

  event_counts <- sort(table(ctx$post[[ctx$event_col]]), decreasing = TRUE)
  head(names(event_counts), ctx$config$auto_event_count)
}

compute_line_attractor_segments <- function(ctx, states) {
  # Draw a straight line-attractor segment for selected states using their slowest stable mode.
  segments <- lapply(states, function(state_id) {
    model <- ctx$state_models[[as.character(state_id)]]
    fp_row <- ctx$fixed_points %>% filter(state == state_id)
    eig <- eigen(model$A)
    abs_lam <- Mod(eig$values)
    # Heuristic: keep only slow but stable modes as line-attractor candidates.
    cand_idx <- which(abs_lam > ctx$config$slow_thr & abs_lam <= ctx$config$stable_thr + 1e-6)
    if (length(cand_idx) == 0 || nrow(fp_row) == 0) {
      return(NULL)
    }
    pick <- cand_idx[which(abs(Im(eig$values[cand_idx])) < 1e-8)[1]]
    if (is.na(pick)) {
      pick <- cand_idx[1]
    }

    direction <- Re(eig$vectors[, pick])
    direction <- direction / sqrt(sum(direction^2))
    fp <- as.numeric(fp_row[1, c("fp_l1", "fp_l2", "fp_l3", "fp_l4")])
    if (!all(is.finite(fp))) {
      return(NULL)
    }
    state_points <- ctx$X[ctx$zs == state_id, , drop = FALSE]
    centered <- sweep(state_points, 2, fp, FUN = "-")
    proj <- as.numeric(centered %*% direction)
    seg_range <- as.numeric(stats::quantile(proj, c(0.1, 0.9), na.rm = TRUE))
    endpoint_mat <- rbind(fp + seg_range[1] * direction, fp + seg_range[2] * direction)
    endpoint_pc <- project_with_pc(ctx$pc, endpoint_mat)[, 1:2, drop = FALSE]

    data.frame(
      state = state_id,
      u0 = endpoint_pc[1, 1],
      v0 = endpoint_pc[1, 2],
      u1 = endpoint_pc[2, 1],
      v1 = endpoint_pc[2, 2],
      lambda_mod = signif(Mod(eig$values[pick]), 4),
      is_complex = abs(Im(eig$values[pick])) >= 1e-8,
      color = ctx$state_palette[[as.character(state_id)]],
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(segments)
}

compute_line_attractor_curve <- function(ctx, states, n_pts = 60, span = 0.5, trim_q = c(0.05, 0.95)) {
  # Draw a smoothed curved line-attractor overlay using slow-mode ordering and LOESS.
  curves <- lapply(states, function(state_id) {
    model <- ctx$state_models[[as.character(state_id)]]
    fp_row <- ctx$fixed_points %>% filter(state == state_id)
    eig <- eigen(model$A)
    abs_lam <- Mod(eig$values)
    cand_idx <- which(abs_lam > ctx$config$slow_thr & abs_lam <= ctx$config$stable_thr + 1e-6)
    if (length(cand_idx) == 0 || nrow(fp_row) == 0) {
      return(NULL)
    }
    pick <- cand_idx[which(abs(Im(eig$values[cand_idx])) < 1e-8)[1]]
    if (is.na(pick)) {
      pick <- cand_idx[1]
    }

    direction <- Re(eig$vectors[, pick])
    direction <- direction / sqrt(sum(direction^2))
    fp <- as.numeric(fp_row[1, c("fp_l1", "fp_l2", "fp_l3", "fp_l4")])
    if (!all(is.finite(fp))) {
      return(NULL)
    }
    state_pts <- ctx$X[ctx$zs == state_id, , drop = FALSE]
    if (nrow(state_pts) < 5) {
      return(NULL)
    }
    centered <- sweep(state_pts, 2, fp, FUN = "-")
    t_proj <- as.numeric(centered %*% direction)
    pc_pts <- project_with_pc(ctx$pc, state_pts)[, 1:2, drop = FALSE]
    t_lim <- as.numeric(stats::quantile(t_proj, trim_q, na.rm = TRUE))
    keep <- is.finite(t_proj) & t_proj >= t_lim[1] & t_proj <= t_lim[2]
    if (sum(keep) < 5 || diff(range(t_proj[keep])) == 0) {
      return(NULL)
    }
    t_seq <- seq(min(t_proj[keep]), max(t_proj[keep]), length.out = n_pts)
    lo_u <- stats::loess(pc_pts[keep, 1] ~ t_proj[keep], span = span, degree = 1)
    lo_v <- stats::loess(pc_pts[keep, 2] ~ t_proj[keep], span = span, degree = 1)

    data.frame(
      state = state_id,
      ord = seq_along(t_seq),
      u = as.numeric(stats::predict(lo_u, newdata = t_seq)),
      v = as.numeric(stats::predict(lo_v, newdata = t_seq)),
      color = ctx$state_palette[[as.character(state_id)]],
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(curves)
}

compute_transition_matrix <- function(post, state_levels, event_col) {
  # Return raw transition counts and row-normalized transition probabilities.
  z <- post$state_most_likely
  state_names <- as.character(state_levels)
  trans_count <- matrix(0, length(state_levels), length(state_levels), dimnames = list(state_names, state_names))

  for (ev in unique(post[[event_col]])) {
    zi <- z[post[[event_col]] == ev]
    if (length(zi) < 2) {
      next
    }

    for (t in seq_len(length(zi) - 1)) {
      from <- as.character(zi[t])
      to <- as.character(zi[t + 1])
      trans_count[from, to] <- trans_count[from, to] + 1
    }
  }

  list(count = trans_count, prob = trans_count / pmax(rowSums(trans_count), 1))
}

run_state_diagnostics <- function(ctx) {
  # Compute state occupancy and local attractor-type diagnostics from fitted dynamics.
  state_levels <- ctx$state_levels
  z <- ctx$zs
  transition <- compute_transition_matrix(ctx$post, state_levels, ctx$event_col)
  trans_prob <- transition$prob

  diag_df <- lapply(names(ctx$state_models), function(state_name) {
    model <- ctx$state_models[[state_name]]
    state_id <- as.integer(state_name)
    matrix_term <- diag(4) - model$A
    cond_value <- kappa(matrix_term)
    fp <- tryCatch(as.numeric(solve(matrix_term, model$b)), error = function(e) rep(NA_real_, 4))
    eig_vals <- eigen(model$A)$values
    abs_eig <- Mod(eig_vals)
    n_slow <- sum(abs_eig > ctx$config$slow_thr & abs_eig <= ctx$config$stable_thr + 1e-6)
    n_unstable <- sum(abs_eig > ctx$config$stable_thr + 1e-6)

    type <- if (n_unstable > 0) {
      "unstable / saddle"
    } else if (n_slow == 0) {
      "point attractor"
    } else if (n_slow == 1) {
      "line attractor"
    } else if (n_slow == 2) {
      "plane attractor"
    } else {
      "marginal (all dirs slow)"
    }

    type_forced <- if (isTRUE(ctx$config$force_plane_as_line) && n_slow >= 1 && n_unstable == 0) {
      "line attractor"
    } else {
      type
    }

    data.frame(
      state = state_id,
      cond_I_minus_A = signif(cond_value, 4),
      singular = cond_value > ctx$config$cond_thr,
      fp_l1 = fp[1],
      fp_l2 = fp[2],
      fp_l3 = fp[3],
      fp_l4 = fp[4],
      lam1_mod = signif(abs_eig[1], 4),
      lam2_mod = signif(abs_eig[2], 4),
      lam3_mod = signif(abs_eig[3], 4),
      lam4_mod = signif(abs_eig[4], 4),
      max_abs_lam = signif(max(abs_eig), 4),
      n_slow_dirs = n_slow,
      n_unstable = n_unstable,
      type = type,
      type_forced = type_forced
    )
  }) %>% bind_rows()

  frac_time_tbl <- prop.table(table(factor(z, levels = state_levels)))
  self_prob <- diag(trans_prob)
  eig <- eigen(t(trans_prob))
  stationary_idx <- which.min(abs(eig$values - 1))
  pi_stationary <- Re(eig$vectors[, stationary_idx])
  pi_stationary <- pi_stationary / sum(pi_stationary)

  summary_state <- data.frame(
    state = state_levels,
    frac_time = round(as.numeric(frac_time_tbl), 4),
    self_prob = round(as.numeric(self_prob), 3),
    pi_stationary = round(as.numeric(pi_stationary), 4)
  ) %>%
    left_join(diag_df[, c("state", "max_abs_lam", "type", "type_forced")], by = "state") %>%
    arrange(desc(frac_time))

  list(
    diag_df = diag_df,
    summary_state = summary_state,
    trans_prob = trans_prob,
    trans_count = transition$count
  )
}

compute_line_attractor_score <- function(ctx, eps = 1e-12) {
  # line_attractor_score = log2(tau_slowest / tau_second_slowest), tau = 1 / |log(|lambda|)|.
  lapply(names(ctx$state_models), function(state_name) {
    eig_vals <- eigen(ctx$state_models[[state_name]]$A)$values
    lam_mod <- Mod(eig_vals)
    lam_mod_clip <- pmin(pmax(lam_mod, eps), 1 - eps)
    tau <- abs(1 / log(lam_mod_clip))
    tau_sorted <- sort(tau, decreasing = TRUE)
    dominant_idx <- order(tau, decreasing = TRUE)[1]
    second_idx <- order(tau, decreasing = TRUE)[2]
    line_score <- log2(tau_sorted[1] / tau_sorted[2])

    data.frame(
      state = as.integer(state_name),
      eig1_mod = round(lam_mod[1], 6),
      eig2_mod = round(lam_mod[2], 6),
      eig3_mod = round(lam_mod[3], 6),
      eig4_mod = round(lam_mod[4], 6),
      tau1 = round(tau[1], 6),
      tau2 = round(tau[2], 6),
      tau3 = round(tau[3], 6),
      tau4 = round(tau[4], 6),
      max_tau = round(tau_sorted[1], 6),
      second_tau = round(tau_sorted[2], 6),
      line_attractor_score = round(line_score, 6),
      slowest_mode_idx = dominant_idx,
      second_slowest_mode_idx = second_idx,
      stringsAsFactors = FALSE
    )
  }) %>% bind_rows()
}

compute_geometry_score <- function(ctx) {
  # Geometry check: do the state's points actually lie on a thin 1D line?
  lapply(ctx$state_levels, function(state_id) {
    state_pts <- ctx$X[ctx$zs == state_id, , drop = FALSE]
    geom_pc <- prcomp(state_pts, center = TRUE, scale. = FALSE)
    variances <- geom_pc$sdev^2
    prop_var <- variances / sum(variances)
    pc1_pc2_ratio <- variances[1] / variances[2]

    data.frame(
      state = as.integer(state_id),
      n_state_points = nrow(state_pts),
      geom_pc1_var = round(variances[1], 6),
      geom_pc2_var = round(variances[2], 6),
      geom_pc3_var = round(variances[3], 6),
      geom_pc4_var = round(variances[4], 6),
      geom_pc1_var_explained = round(prop_var[1], 6),
      geometry_score = round(pc1_pc2_ratio, 6),
      stringsAsFactors = FALSE
    )
  }) %>% bind_rows()
}

build_trajectory_df <- function(ctx, surface = NULL, event_nums, use_surface_z = TRUE) {
  # Create per-event path in PCA space, with behavior + phase labels joined on.
  event_nums <- unlist(event_nums, use.names = FALSE)
  traj_list <- list()

  if (use_surface_z && is.null(surface)) {
    stop("surface must be provided when use_surface_z = TRUE")
  }

  for (ev in event_nums) {
    idx <- which(ctx$post[[ctx$event_col]] == ev)
    post_slice <- ctx$post[idx, , drop = FALSE]
    timestamp_key <- if ("original_timestamp" %in% colnames(post_slice)) {
      make_join_key(post_slice$original_timestamp)
    } else {
      rep(NA_character_, nrow(post_slice))
    }

    behavior_info <- data.frame(
      event_key = as.character(post_slice[[ctx$event_col]]),
      timestamp_key = timestamp_key,
      ord = seq_along(idx),
      stringsAsFactors = FALSE
    ) %>%
      left_join(ctx$behavior_lookup, by = c("event_key", "timestamp_key")) %>%
      arrange(ord) %>%
      mutate(
        digging_status = ifelse(is.na(digging_status), 0L, digging_status),
        ever_dug = cummax(digging_status == 1L),
        status_label = make_behavior_status_label(timestamp, digging_status, ever_dug),
        phase_label = make_event_phase_label(timestamp, digging_status, ever_dug)
      )

    traj_x <- as.matrix(ctx$post[idx, c("latent_1", "latent_2", "latent_3", "latent_4")])
    traj_pc <- project_with_pc(ctx$pc, traj_x)[, 1:3, drop = FALSE]
    traj_u <- traj_pc[, 1]
    traj_v <- traj_pc[, 2]
    traj_z <- traj_pc[, 3]

    if (use_surface_z) {
      traj_u <- pmin(pmax(traj_u, min(surface$u_seq)), max(surface$u_seq))
      traj_v <- pmin(pmax(traj_v, min(surface$v_seq)), max(surface$v_seq))
      traj_z <- surface$interp_z(traj_u, traj_v) + 0.02
    }

    gap_group <- cumsum(c(TRUE, diff(seq_along(idx)) > 1))

    traj_list[[as.character(ev)]] <- data.frame(
      event = as.character(ev),
      ord = seq_along(idx),
      timestamp = behavior_info$timestamp,
      u = traj_u,
      v = traj_v,
      z = traj_z,
      w_raw = traj_pc[, 3],
      state = ctx$zs[idx],
      digging_status = behavior_info$digging_status,
      drinking_status = behavior_info$drinking_status,
      status_label = behavior_info$status_label,
      phase_label = behavior_info$phase_label,
      is_valid = is.finite(traj_u) & is.finite(traj_v) & is.finite(traj_z),
      gap_group = gap_group
    )
  }

  bind_rows(traj_list)
}

build_status_segment_df <- function(traj_df) {
  # Split each valid trajectory into same-status segments so behavior changes can be colored separately.
  segment_list <- list()

  for (event_name in unique(traj_df$event)) {
    event_df <- traj_df %>% filter(event == event_name, is_valid) %>% arrange(ord)
    for (group_id in unique(event_df$gap_group)) {
      group_df <- event_df %>% filter(gap_group == group_id)
      if (nrow(group_df) < 2) {
        next
      }

      edge_status <- group_df$status_label[-nrow(group_df)]
      edge_run <- cumsum(c(TRUE, edge_status[-1] != edge_status[-length(edge_status)]))

      for (edge_id in unique(edge_run)) {
        edge_idx <- which(edge_run == edge_id)
        segment_df <- group_df[min(edge_idx):(max(edge_idx) + 1), , drop = FALSE]
        segment_df$segment_id <- paste(event_name, group_id, edge_id, sep = "_")
        segment_df$segment_status <- edge_status[min(edge_idx)]
        segment_list[[length(segment_list) + 1]] <- segment_df
      }
    }
  }

  dplyr::bind_rows(segment_list)
}

build_time_label_df <- function(traj_df, every = 50) {
  # Drop label on every Nth frame (t50, t100, ...).
  if (is.null(traj_df) || nrow(traj_df) == 0 || every <= 0) {
    return(data.frame())
  }

  bind_rows(lapply(unique(traj_df$event), function(event_name) {
    event_df <- traj_df %>% filter(event == event_name, is_valid) %>% arrange(ord)
    frame_id <- event_df$ord - 1
    keep <- frame_id > 0 & frame_id %% every == 0
    if (!any(keep)) {
      return(NULL)
    }

    event_df[keep, , drop = FALSE] %>%
      mutate(time_label = paste0("t", frame_id[keep]))
  }))
}

run_phase_state_fisher_tests <- function(ctx, states_to_test = 0:3, n_sim = 1e5) {
  # Overall RxC uses Monte-Carlo Fisher; pairwise rows are exact 2x2 enrichment tests.
  phase_levels <- c("pre_approach", "approach", "digging", "after_digging")

  traj_df <- build_trajectory_df(
    ctx = ctx,
    surface = NULL,
    event_nums = unique(ctx$post[[ctx$event_col]]),
    use_surface_z = FALSE
  ) %>%
    filter(
      !is.na(phase_label),
      state %in% states_to_test,
      phase_label %in% phase_levels
    ) %>%
    mutate(
      phase_label = factor(phase_label, levels = phase_levels),
      state = factor(state, levels = states_to_test)
    )

  contingency_tbl <- table(traj_df$phase_label, traj_df$state)
  overall_test <- fisher.test(contingency_tbl, simulate.p.value = TRUE, B = n_sim)

  per_pair <- lapply(phase_levels, function(ph) {
    lapply(states_to_test, function(st) {
      in_phase <- traj_df$phase_label == ph
      in_state <- traj_df$state == st
      mat <- matrix(
        c(
          sum(in_phase & in_state),
          sum(in_phase & !in_state),
          sum(!in_phase & in_state),
          sum(!in_phase & !in_state)
        ),
        nrow = 2,
        byrow = TRUE,
        dimnames = list(
          phase = c(ph, paste0("not_", ph)),
          state = c(as.character(st), paste0("not_", st))
        )
      )
      ft <- fisher.test(mat)

      data.frame(
        phase = ph,
        state = as.integer(as.character(st)),
        n_phase_state = mat[1, 1],
        n_phase_not_state = mat[1, 2],
        n_not_phase_state = mat[2, 1],
        n_not_phase_not_state = mat[2, 2],
        odds_ratio = unname(ft$estimate),
        p_value = ft$p.value,
        stringsAsFactors = FALSE
      )
    }) %>% bind_rows()
  }) %>% bind_rows()

  per_pair <- per_pair %>%
    mutate(
      p_adj_bh = p.adjust(p_value, method = "BH"),
      enriched = ifelse(!is.na(odds_ratio) & odds_ratio > 1, TRUE, FALSE)
    ) %>%
    arrange(p_adj_bh, p_value)

  list(
    data = traj_df,
    contingency_table = contingency_tbl,
    overall_test = overall_test,
    pairwise_results = per_pair
  )
}

add_xyz_inset_3d <- function(p, surface, axis_limits = NULL, show_xyz_inset = TRUE) {
  if (!isTRUE(show_xyz_inset)) {
    return(p)
  }

  origin <- c(x = 0, y = 0, z = 0)
  axis_length <- c(x = 0.82, y = 0.82, z = 0.96)
  label_pad <- c(x = 0.12, y = 0.12, z = 0.15)
  axis_specs <- list(
    list(label = "PC1", color = "#c0392b", tip = c(origin["x"] + axis_length["x"], origin["y"], origin["z"]), text = c(origin["x"] + axis_length["x"] + label_pad["x"], origin["y"] - 0.06, origin["z"])),
    list(label = "PC2", color = "#1f9d55", tip = c(origin["x"], origin["y"] + axis_length["y"], origin["z"]), text = c(origin["x"] - 0.06, origin["y"] + axis_length["y"] + label_pad["y"], origin["z"])),
    list(label = "energy", color = "#2c7fb8", tip = c(origin["x"], origin["y"], origin["z"] + axis_length["z"]), text = c(origin["x"] - 0.06, origin["y"] - 0.06, origin["z"] + axis_length["z"] + label_pad["z"]))
  )

  p <- p %>% add_trace(
    x = origin["x"],
    y = origin["y"],
    z = origin["z"],
    type = "scatter3d",
    mode = "markers",
    marker = list(size = 3.5, color = "#444444"),
    hoverinfo = "skip",
    showlegend = FALSE,
    scene = "scene2",
    inherit = FALSE
  )

  for (axis_spec in axis_specs) {
    p <- p %>%
      add_trace(
        x = c(origin["x"], axis_spec$tip[1]),
        y = c(origin["y"], axis_spec$tip[2]),
        z = c(origin["z"], axis_spec$tip[3]),
        type = "scatter3d",
        mode = "lines",
        line = list(color = axis_spec$color, width = 10),
        hoverinfo = "skip",
        showlegend = FALSE,
        scene = "scene2",
        inherit = FALSE
      ) %>%
      add_trace(
        x = axis_spec$text[1],
        y = axis_spec$text[2],
        z = axis_spec$text[3],
        type = "scatter3d",
        mode = "text",
        text = paste0("<b>", axis_spec$label, "</b>"),
        textfont = list(size = 13, color = axis_spec$color, family = "Arial"),
        hoverinfo = "skip",
        showlegend = FALSE,
        scene = "scene2",
        inherit = FALSE
      )
  }

  htmlwidgets::onRender(
    p,
    "function(el, x) {
      var gd = document.getElementById(el.id);
      if (!gd) return;
      var syncingInset = false;

      function cloneCamera(camera) {
        return camera ? JSON.parse(JSON.stringify(camera)) : null;
      }

      function readMainCamera(evt) {
        if (evt && evt['scene.camera']) return cloneCamera(evt['scene.camera']);
        var fullLayoutCamera = gd._fullLayout && gd._fullLayout.scene ? gd._fullLayout.scene.camera : null;
        return cloneCamera(fullLayoutCamera || (gd.layout && gd.layout.scene ? gd.layout.scene.camera : null));
      }

      function syncInsetCamera(camera) {
        var nextCamera = camera || readMainCamera();
        if (!nextCamera || syncingInset) return;
        syncingInset = true;
        Plotly.relayout(gd, {'scene2.camera': nextCamera}).then(function() {
          syncingInset = false;
        }).catch(function() {
          syncingInset = false;
        });
      }

      gd.on('plotly_afterplot', function() {
        syncInsetCamera();
      });

      gd.on('plotly_relayouting', function(e) {
        if (!e || syncingInset) return;
        if (e['scene.camera'] || e['scene.camera.eye'] || e['scene.camera.up'] || e['scene.camera.center']) {
          syncInsetCamera(readMainCamera(e));
        }
      });

      gd.on('plotly_relayout', function(e) {
        if (!e || syncingInset) return;
        if (e['scene.camera'] || e['scene.camera.eye'] || e['scene.camera.up'] || e['scene.camera.center']) {
          syncInsetCamera(readMainCamera(e));
        }
      });

      requestAnimationFrame(function() {
        syncInsetCamera();
      });
    }"
  )
}

plot_event_3d <- function(ctx, surface, event_nums,
                          line_width = 1,
                          time_marker_every = 50,
                          arrow_every = 8,
                          cone_size_ref = 0.2,
                          cone_vec_len = 0.12,
                          show_time_markers = FALSE,
                          show_trough_label = TRUE,
                          show_endpoints = TRUE,
                          show_direction_cones = TRUE,
                          show_xyz_inset = TRUE,
                          time_label_z_lift = 0.08,
                          axis_limits = NULL) {
  # Construct main interactive figure: energy surface + event trajectories in 3D.
  traj_df <- build_trajectory_df(ctx, surface, event_nums, use_surface_z = TRUE)
  time_label_df <- build_time_label_df(traj_df, every = time_marker_every)
  cone_list <- list()
  shown_statuses <- character(0)
  z_label_offset <- time_label_z_lift * diff(surface$z_axis_range)

  p <- plot_ly() %>%
    add_surface(
      x = surface$u_seq,
      y = surface$v_seq,
      z = t(surface$z_sm),
      cmin = surface$qlo,
      cmax = surface$qhi,
      colorscale = coolwarm_colorscale,
      contours = list(z = list(show = FALSE)),
      opacity = 0.6,
      colorbar = list(title = list(text = "energy", font = list(size = 13)), tickfont = list(size = 10), len = 0.55, thickness = 14, x = 1.02, y = 0.54, xanchor = "left"),
      lighting = list(ambient = 0.65, diffuse = 0.8, specular = 0.15, roughness = 0.55, fresnel = 0.2),
      showscale = TRUE
    )

  if (show_trough_label) {
    min_idx <- which(surface$z_sm == min(surface$z_sm, na.rm = TRUE), arr.ind = TRUE)[1, ]
    p <- p %>% add_trace(
      x = surface$u_seq[min_idx[1]],
      y = surface$v_seq[min_idx[2]],
      z = surface$z_sm[min_idx[1], min_idx[2]] - 0.05,
      type = "scatter3d",
      mode = "text",
      text = "trough",
      textfont = list(size = 14, color = "#444444", family = "Arial"),
      showlegend = FALSE,
      hoverinfo = "skip"
    )
  }

  traj_segments <- build_status_segment_df(traj_df)
  for (segment_id in unique(traj_segments$segment_id)) {
    segment_df <- traj_segments[traj_segments$segment_id == segment_id, , drop = FALSE]
    status_label <- segment_df$segment_status[1]
    status_color <- unname(ctx$behavior_palette[[status_label]])
    show_status_legend <- !(status_label %in% shown_statuses)
    if (show_status_legend) {
      shown_statuses <- c(shown_statuses, status_label)
    }

    p <- p %>% add_trace(
      x = segment_df$u,
      y = segment_df$v,
      z = segment_df$z + 0.03,
      type = "scatter3d",
      mode = "lines",
      line = list(color = status_color, width = line_width),
      name = paste0("status ", status_label),
      legendgroup = paste0("status_", status_label),
      showlegend = show_status_legend,
      hovertext = sprintf("event %s, frame %d, timestamp %.3f, status %s", segment_df$event, segment_df$ord - 1, segment_df$timestamp, status_label),
      hoverinfo = "text"
    )
  }

  for (event_name in unique(traj_df$event)) {
    event_df <- traj_df %>% filter(event == event_name, is_valid) %>% arrange(ord)
    for (group_id in unique(event_df$gap_group)) {
      group_df <- event_df %>% filter(gap_group == group_id)
      if (nrow(group_df) < 2) {
        next
      }

      if (show_direction_cones) {
        arrow_idx <- seq(1, nrow(group_df) - 1, by = arrow_every)
        cone_df <- data.frame(
          x = group_df$u[arrow_idx],
          y = group_df$v[arrow_idx],
          z = group_df$z[arrow_idx] + 0.03,
          u = group_df$u[arrow_idx + 1] - group_df$u[arrow_idx],
          v = group_df$v[arrow_idx + 1] - group_df$v[arrow_idx],
          w = group_df$z[arrow_idx + 1] - group_df$z[arrow_idx]
        )
        mag <- sqrt(cone_df$u^2 + cone_df$v^2 + cone_df$w^2)
        keep <- is.finite(mag) & mag > 0
        cone_df <- cone_df[keep, , drop = FALSE]
        mag <- mag[keep]
        if (nrow(cone_df) > 0) {
          cone_df$u <- cone_df$u / mag * cone_vec_len
          cone_df$v <- cone_df$v / mag * cone_vec_len
          cone_df$w <- cone_df$w / mag * cone_vec_len
          cone_list[[length(cone_list) + 1]] <- cone_df
        }
      }

      if (show_endpoints) {
        first_valid <- group_df[1, , drop = FALSE]
        last_valid <- group_df[nrow(group_df), , drop = FALSE]
        p <- p %>%
          add_markers(x = first_valid$u, y = first_valid$v, z = first_valid$z + 0.03, marker = list(size = 10, color = "white", line = list(color = "black", width = 2.4), symbol = "circle"), name = paste0("start (ev ", event_name, ")"), showlegend = FALSE, hoverinfo = "skip") %>%
          add_markers(x = last_valid$u, y = last_valid$v, z = last_valid$z + 0.03, marker = list(size = 10, color = "black", line = list(color = "black", width = 1.2), symbol = "circle"), name = paste0("end (ev ", event_name, ")"), showlegend = FALSE, hoverinfo = "skip")
      }
    }
  }

  if (show_time_markers && nrow(time_label_df) > 0) {
    p <- p %>%
      add_trace(data = time_label_df, x = ~u, y = ~v, z = ~z + 0.03, type = "scatter3d", mode = "markers", marker = list(size = 6, color = "white", line = list(color = "black", width = 1.6), symbol = "circle"), showlegend = FALSE, hovertext = ~sprintf("event %s, %s", event, time_label), hoverinfo = "text") %>%
      add_trace(data = time_label_df, x = ~u, y = ~v, z = ~z + z_label_offset, type = "scatter3d", mode = "text", text = ~paste0("<b>", time_label, "</b>"), textposition = "top center", textfont = list(size = 11, color = "#111111", family = "Arial"), showlegend = FALSE, hovertext = ~sprintf("event %s, %s", event, time_label), hoverinfo = "text")
  }

  if (show_direction_cones && length(cone_list) > 0) {
    cone_df_all <- bind_rows(cone_list)
    p <- p %>% add_trace(x = cone_df_all$x, y = cone_df_all$y, z = cone_df_all$z, u = cone_df_all$u, v = cone_df_all$v, w = cone_df_all$w, type = "cone", colorscale = list(c(0, "black"), c(1, "black")), showscale = FALSE, sizemode = "raw", sizeref = cone_size_ref, anchor = "tail", hoverinfo = "skip", showlegend = FALSE)
  }

  title_text <- sprintf("%s attractor view", ctx$config$label)
  xaxis_cfg <- list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, showticklabels = FALSE, title = list(text = ""))
  yaxis_cfg <- list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, showticklabels = FALSE, title = list(text = ""))
  if (!is.null(axis_limits)) {
    xaxis_cfg$range <- unname(axis_limits$u)
    yaxis_cfg$range <- unname(axis_limits$v)
  }

  p <- add_xyz_inset_3d(p, surface, axis_limits = axis_limits, show_xyz_inset = show_xyz_inset)

  p %>% layout(
    title = list(text = title_text, font = list(size = 14)),
    scene = list(xaxis = xaxis_cfg, yaxis = yaxis_cfg, zaxis = list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, showticklabels = FALSE, title = list(text = ""), range = unname(surface$z_axis_range)), domain = list(x = c(0.00, 0.88), y = c(0.06, 1.00)), camera = list(eye = list(x = 1.5, y = -1.7, z = 1.0), up = list(x = 0, y = 0, z = 1)), aspectmode = "cube"),
    scene2 = list(domain = list(x = c(0.03, 0.20), y = c(0.02, 0.24)), xaxis = list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, title = list(text = ""), range = c(-0.15, 1.15)), yaxis = list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, title = list(text = ""), range = c(-0.15, 1.15)), zaxis = list(visible = FALSE, showgrid = FALSE, zeroline = FALSE, showbackground = FALSE, title = list(text = ""), range = c(-0.15, 1.25)), bgcolor = "rgba(255,255,255,0)", aspectmode = "cube", camera = list(eye = list(x = 1.5, y = -1.7, z = 1.0), up = list(x = 0, y = 0, z = 1))),
    legend = list(orientation = "h", x = 0.44, xanchor = "center", y = -0.05, yanchor = "top", bgcolor = "rgba(255,255,255,0.88)", bordercolor = "rgba(0,0,0,0.10)", borderwidth = 1, font = list(size = 10)),
    paper_bgcolor = "white",
    plot_bgcolor = "white",
    margin = list(l = 0, r = 95, b = 70, t = 45)
  )
}

plot_energy_2d <- function(ctx,
                           surface,
                           event_nums = NULL,
                           n_grid_e = 250,
                           arrow_stride = 12,
                           arrow_scale = 1.00,
                           min_arrow_len = 0.2,
                           show_energy = FALSE,
                           energy_clip = c(0, 0.99),
                           data_quantile = 0.99,
                           mask_dist_q = 0.95,
                           show_traj = TRUE,
                           label_endpoints = TRUE,
                           line_attractor_df = NULL,
                           line_attractor_curve_df = NULL,
                           plot_limits = NULL,
                           arrow_mag_quantile = 0.05,
                           show_time_labels = TRUE,
                           time_label_every = 50,
                           traj_offset_u = 0.004,
                           traj_offset_v = -0.012,
                           field_mode = c("downhill", "attractor")) {
  # Construct main static 2D flow field in PC1-PC2, with optional trajectory and line-attractor overlay.
  field_mode <- match.arg(field_mode)
  has_colour_scale <- FALSE

  if (is.null(plot_limits)) {
    u_lim <- quantile(ctx$U, c((1 - data_quantile) / 2, 1 - (1 - data_quantile) / 2))
    v_lim <- quantile(ctx$V, c((1 - data_quantile) / 2, 1 - (1 - data_quantile) / 2))
    u_lim <- u_lim + c(-0.05, 0.05) * diff(u_lim)
    v_lim <- v_lim + c(-0.15, 0.15) * diff(v_lim)
  } else {
    u_lim <- plot_limits$u
    v_lim <- plot_limits$v
  }

  kd <- MASS::kde2d(ctx$U, ctx$V, n = n_grid_e, lims = c(u_lim, v_lim))
  energy_df <- expand.grid(u = kd$x, v = kd$y)
  energy_df$E <- -log(as.vector(kd$z) + 1e-12)
  qlo_e <- quantile(energy_df$E, energy_clip[1], na.rm = TRUE)
  qhi_e <- quantile(energy_df$E, energy_clip[2], na.rm = TRUE)
  energy_df$E <- pmin(pmax(energy_df$E, qlo_e), qhi_e)
  energy_mat <- matrix(energy_df$E, nrow = length(kd$x), ncol = length(kd$y))

  data_pts <- cbind(ctx$U, ctx$V)
  nn_energy <- FNN::get.knnx(data_pts, as.matrix(energy_df[, c("u", "v")]), k = 1)
  dist_thr <- quantile(nn_energy$nn.dist, mask_dist_q)
  energy_df$E[nn_energy$nn.dist[, 1] > dist_thr] <- NA
  energy_mat[nn_energy$nn.dist[, 1] > dist_thr] <- NA_real_

  du_step <- mean(diff(kd$x))
  dv_step <- mean(diff(kd$y))
  grad_u <- matrix(NA_real_, nrow = nrow(energy_mat), ncol = ncol(energy_mat))
  grad_v <- matrix(NA_real_, nrow = nrow(energy_mat), ncol = ncol(energy_mat))
  grad_u[1, ] <- (energy_mat[2, ] - energy_mat[1, ]) / du_step
  grad_u[nrow(energy_mat), ] <- (energy_mat[nrow(energy_mat), ] - energy_mat[nrow(energy_mat) - 1, ]) / du_step
  grad_u[2:(nrow(energy_mat) - 1), ] <- (energy_mat[3:nrow(energy_mat), ] - energy_mat[1:(nrow(energy_mat) - 2), ]) / (2 * du_step)
  grad_v[, 1] <- (energy_mat[, 2] - energy_mat[, 1]) / dv_step
  grad_v[, ncol(energy_mat)] <- (energy_mat[, ncol(energy_mat)] - energy_mat[, ncol(energy_mat) - 1]) / dv_step
  grad_v[, 2:(ncol(energy_mat) - 1)] <- (energy_mat[, 3:ncol(energy_mat)] - energy_mat[, 1:(ncol(energy_mat) - 2)]) / (2 * dv_step)

  if (identical(field_mode, "downhill")) {
    arrow_df <- expand.grid(u = kd$x, v = kd$y) %>%
      mutate(E = as.vector(energy_mat), du = -as.vector(grad_u), dv = -as.vector(grad_v), state = NA_integer_)
    idx0 <- seq_len(nrow(arrow_df)) - 1
    ix <- idx0 %% n_grid_e
    iy <- idx0 %/% n_grid_e
    keep <- (ix %% arrow_stride == 0) & (iy %% arrow_stride == 0)
    arrow_sub <- arrow_df[keep, ] %>% filter(u >= u_lim[1], u <= u_lim[2], v >= v_lim[1], v <= v_lim[2], is.finite(E), is.finite(du), is.finite(dv))
  } else {
    knn_w <- FNN::get.knnx(ctx$X, surface$grid3d, k = ctx$config$knn_flow_k)
    flow_uv <- t(vapply(seq_len(nrow(surface$grid3d)), function(i) {
      neighbor_states <- ctx$zs[knn_w$nn.index[i, ]]
      p_k <- prop.table(table(factor(neighbor_states, levels = ctx$state_levels)))
      x3 <- surface$grid3d[i, ]
      dx <- rep(0, 3)
      for (state_id in ctx$state_levels) {
        weight <- p_k[as.character(state_id)]
        if (weight == 0) {
          next
        }
        model <- ctx$state_models[[as.character(state_id)]]
        dx <- dx + weight * as.numeric(model$A %*% x3 + model$b - x3)
      }
      c(sum(dx * ctx$pc$rotation[, 1]), sum(dx * ctx$pc$rotation[, 2]))
    }, numeric(2)))

    arrow_df <- data.frame(u = surface$grid2d$u, v = surface$grid2d$v, E = surface$z_speed, du = flow_uv[, 1], dv = flow_uv[, 2], state = surface$state_at)
    idx0 <- seq_len(nrow(arrow_df)) - 1
    ix <- idx0 %% ctx$config$n_grid
    iy <- idx0 %/% ctx$config$n_grid
    keep <- (ix %% arrow_stride == 0) & (iy %% arrow_stride == 0)
    arrow_sub <- arrow_df[keep, ] %>% filter(u >= u_lim[1], u <= u_lim[2], v >= v_lim[1], v <= v_lim[2])
    nn_arrow <- FNN::get.knnx(data_pts, as.matrix(arrow_sub[, c("u", "v")]), k = 1)
    arrow_sub <- arrow_sub[nn_arrow$nn.dist[, 1] <= dist_thr, , drop = FALSE]
  }

  mag <- sqrt(arrow_sub$du^2 + arrow_sub$dv^2)
  if (arrow_mag_quantile > 0 && any(is.finite(mag))) {
    mag_cut <- stats::quantile(mag[is.finite(mag)], probs = arrow_mag_quantile, na.rm = TRUE)
    keep_mag <- is.finite(mag) & mag >= mag_cut
    arrow_sub <- arrow_sub[keep_mag, , drop = FALSE]
    mag <- mag[keep_mag]
  }

  if (nrow(arrow_sub) == 0 || !any(is.finite(mag) & mag > 0)) {
    arrow_sub <- data.frame(u = numeric(0), v = numeric(0), xend = numeric(0), yend = numeric(0))
  } else {

    grid_density <- if (identical(field_mode, "attractor")) ctx$config$n_grid else n_grid_e
    base_scale_factor <- arrow_scale * (diff(u_lim) / (grid_density / arrow_stride)) / max(mag, na.rm = TRUE)
    dx0 <- arrow_sub$du * base_scale_factor
    dy0 <- arrow_sub$dv * base_scale_factor
    len0 <- sqrt(dx0^2 + dy0^2)
    tiny <- is.finite(len0) & len0 > 0 & len0 < min_arrow_len
    stretch <- min_arrow_len / len0[tiny]
    dx0[tiny] <- dx0[tiny] * stretch
    dy0[tiny] <- dy0[tiny] * stretch
    arrow_sub$xend <- arrow_sub$u + dx0
    arrow_sub$yend <- arrow_sub$v + dy0
  }

  p <- ggplot()

  if (show_energy) {
    p <- p +
      geom_raster(data = energy_df, aes(u, v, fill = E), interpolate = TRUE, na.rm = TRUE) +
      scale_fill_gradientn(colours = c("#3b4cc0", "#6788ee", "#9abbff", "#c9d7f0", "#edd1c2", "#f7a889", "#e26952", "#b40426"), name = "energy", na.value = "white")
  }

  p <- p +
    geom_segment(data = arrow_sub, aes(x = u, y = v, xend = xend, yend = yend), arrow = arrow(length = unit(0.05, "inches"), type = "closed"), colour = "gray20", alpha = 0.85, linewidth = 0.22, lineend = "round")

  if (!is.null(line_attractor_curve_df) && nrow(line_attractor_curve_df) > 0) {
    has_colour_scale <- TRUE
    p <- p + geom_path(data = line_attractor_curve_df, aes(x = u, y = v, group = factor(state), colour = color), linewidth = 0.9, linetype = "22", alpha = 0.95, show.legend = FALSE)
  } else if (!is.null(line_attractor_df) && nrow(line_attractor_df) > 0) {
    has_colour_scale <- TRUE
    p <- p + geom_segment(data = line_attractor_df, aes(x = u0, y = v0, xend = u1, yend = v1, colour = color), linewidth = 0.9, linetype = "22", alpha = 0.95, show.legend = FALSE)
  }

  if (show_traj && !is.null(event_nums)) {
    traj_df <- build_trajectory_df(ctx, surface, event_nums, use_surface_z = TRUE) %>% filter(is_valid)
    traj_segments <- build_status_segment_df(traj_df) %>% mutate(color = unname(ctx$behavior_palette[segment_status]))
    traj_u_offset <- traj_offset_u * diff(u_lim)
    traj_v_offset <- traj_offset_v * diff(v_lim)
    traj_segments <- traj_segments %>% mutate(u_plot = u + traj_u_offset, v_plot = v + traj_v_offset)
    time_label_df <- build_time_label_df(traj_df, every = time_label_every) %>% mutate(u_plot = u + traj_u_offset, v_plot = v + traj_v_offset)
    has_colour_scale <- has_colour_scale || nrow(traj_segments) > 0

    p <- p +
      geom_path(data = traj_segments, aes(u_plot, v_plot, group = segment_id), colour = "white", linewidth = 1.15, alpha = 0.90, lineend = "round") +
      geom_path(data = traj_segments, aes(u_plot, v_plot, group = segment_id, colour = color), linewidth = 0.55, lineend = "round")

    if (label_endpoints) {
      starts <- traj_df %>% group_by(event) %>% slice_min(order_by = ord, n = 1, with_ties = FALSE) %>% mutate(u_plot = u + traj_u_offset, v_plot = v + traj_v_offset)
      ends <- traj_df %>% group_by(event) %>% slice_max(order_by = ord, n = 1, with_ties = FALSE) %>% mutate(u_plot = u + traj_u_offset, v_plot = v + traj_v_offset)
      p <- p +
        geom_point(data = starts, aes(u_plot, v_plot), shape = 21, fill = "white", colour = "black", size = 3.2, stroke = 1.1) +
        geom_point(data = ends, aes(u_plot, v_plot), shape = 4, colour = "black", size = 3.5, stroke = 1.4)
    }

    if (show_time_labels && nrow(time_label_df) > 0) {
      p <- p +
        geom_point(data = time_label_df, aes(u_plot, v_plot), shape = 21, fill = "white", colour = "black", size = 2.4, stroke = 1.0) +
        geom_text(data = time_label_df, aes(u_plot, v_plot, label = time_label), size = 2.6, nudge_y = 0.016 * diff(v_lim), colour = "white", fontface = "bold") +
        geom_text(data = time_label_df, aes(u_plot, v_plot, label = time_label), size = 2.15, nudge_y = 0.016 * diff(v_lim), colour = "#111111", fontface = "bold")
    }
  }

  if (has_colour_scale) {
    p <- p + scale_colour_identity(guide = "none")
  }

  p +
    coord_cartesian(xlim = u_lim, ylim = v_lim, expand = FALSE) +
    labs(title = sprintf("%s %s field in PC1-PC2", ctx$config$label, if (identical(field_mode, "attractor")) "attractor" else "occupancy downhill"), x = expression(PC[1]), y = expression(PC[2])) +
    theme_classic(base_size = 13) +
    theme(aspect.ratio = 1, panel.background = element_rect(fill = "white", colour = NA), panel.grid = element_blank(), legend.position = "none")
}

make_significance_label <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ "ns"
  )
}

compute_phase_overall_pvalues <- function(contingency_table) {
  phases <- rownames(contingency_table)
  res <- lapply(phases, function(ph) {
    in_ph <- as.numeric(contingency_table[ph, ])
    out_ph <- as.numeric(colSums(contingency_table[setdiff(phases, ph), , drop = FALSE]))
    mat <- rbind(in_ph, out_ph)
    p <- tryCatch(stats::chisq.test(mat)$p.value, error = function(e) NA_real_)
    data.frame(phase = ph, p_overall = p, stringsAsFactors = FALSE)
  })
  out <- dplyr::bind_rows(res)
  out$p_overall_bh <- p.adjust(out$p_overall, method = "BH")
  out
}

make_phase_strip_labels <- function(phase_levels, overall_p) {
  labels <- vapply(phase_levels, function(ph) {
    row <- overall_p[overall_p$phase == ph, , drop = FALSE]
    if (nrow(row) == 0 || is.na(row$p_overall_bh)) {
      return(ph)
    }
    sprintf("%s\n(states differ p=%.2g)", ph, row$p_overall_bh)
  }, character(1))
  setNames(labels, phase_levels)
}

plot_state_encoding_oddsratio <- function(pairwise_results, contingency_table, state_palette, label) {
  # Construct per-phase bars of log2(odds ratio) to show which state is enriched in each behavior.
  phase_levels <- c("pre_approach", "approach", "digging", "after_digging")
  overall_p <- compute_phase_overall_pvalues(contingency_table)
  strip_labels <- make_phase_strip_labels(phase_levels, overall_p)
  df <- pairwise_results %>%
    mutate(
      phase = factor(phase, levels = phase_levels),
      state = factor(state),
      log2_or = dplyr::case_when(
        is.na(odds_ratio) ~ NA_real_,
        odds_ratio == 0 ~ -Inf,
        is.infinite(odds_ratio) ~ Inf,
        TRUE ~ log2(odds_ratio)
      ),
      sig = make_significance_label(p_adj_bh)
    ) %>%
    filter(!is.na(phase))

  finite_abs <- abs(df$log2_or[is.finite(df$log2_or)])
  cap <- if (length(finite_abs) > 0) max(2, ceiling(max(finite_abs, na.rm = TRUE))) else 5
  df <- df %>%
    mutate(
      log2_or_plot = dplyr::case_when(
        is.na(log2_or) ~ 0,
        log2_or > cap ~ cap,
        log2_or < -cap ~ -cap,
        TRUE ~ log2_or
      ),
      star_y = log2_or_plot + ifelse(log2_or_plot >= 0, 0.06 * cap, -0.06 * cap)
    )

  ggplot(df, aes(state, log2_or_plot, fill = state)) +
    geom_col(width = 0.7, colour = "gray20", linewidth = 0.2) +
    geom_hline(yintercept = 0, linewidth = 0.4, colour = "gray40") +
    geom_text(aes(y = star_y, label = sig), size = 3.2, fontface = "bold") +
    facet_wrap(~phase, nrow = 1, labeller = labeller(phase = strip_labels)) +
    scale_fill_manual(values = state_palette) +
    coord_cartesian(ylim = c(-cap * 1.15, cap * 1.15)) +
    labs(
      title = sprintf("%s: per behavior phase, which state stands out", label),
      subtitle = "Bars = log2(odds ratio); >0 = state enriched in this phase. Stars: BH Fisher (* <.05, ** <.01, *** <.001)",
      x = "hidden state",
      y = expression(log[2] ~ "(odds ratio)")
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none", panel.grid.minor = element_blank())
}

save_plotly_widget <- function(widget, output_path) {
  htmlwidgets::saveWidget(widget, output_path, selfcontained = FALSE)
}

# 4. Run a single analysis given loaded data and config.
run_single_analysis <- function(loaded, config) {
  # Main pipeline for a single analysis run.
  ctx <- prepare_analysis_context(loaded$post, loaded$fpar, loaded$behavior, config)
  surface <- build_energy_surface(ctx)
  diagnostics <- run_state_diagnostics(ctx)
  line_attr_scores <- compute_line_attractor_score(ctx)
  geometry_scores <- compute_geometry_score(ctx)

  diagnostics$diag_df <- diagnostics$diag_df %>%
    left_join(line_attr_scores %>% dplyr::select(state, max_tau, second_tau, line_attractor_score), by = "state") %>%
    left_join(geometry_scores %>% dplyr::select(state, n_state_points, geom_pc1_var_explained, geometry_score), by = "state")
  diagnostics$summary_state <- diagnostics$summary_state %>%
    left_join(line_attr_scores %>% dplyr::select(state, max_tau, second_tau, line_attractor_score), by = "state") %>%
    left_join(geometry_scores %>% dplyr::select(state, n_state_points, geom_pc1_var_explained, geometry_score), by = "state")

  phase_state_fisher <- run_phase_state_fisher_tests(ctx, states_to_test = sort(intersect(ctx$state_levels, 0:3)))
  event_nums <- resolve_selected_events(ctx)

  eff_line_states <- if (isTRUE(config$force_plane_as_line)) {
    auto_plane_states <- diagnostics$diag_df$state[diagnostics$diag_df$type %in% c("plane attractor", "marginal (all dirs slow)")]
    unique(c(config$line_attractor_states, auto_plane_states))
  } else {
    config$line_attractor_states
  }

  line_attractor_df <- compute_line_attractor_segments(ctx, eff_line_states)
  line_attractor_curve_df <- if (isTRUE(config$use_curved_attractor_line) && length(eff_line_states) > 0) {
    compute_line_attractor_curve(ctx, eff_line_states)
  } else {
    NULL
  }

  # Only keep a single line overlay in 2D, and only for the Final panel.
  is_final_panel <- identical(config$label, "Final")
  top_line_state <- integer(0)
  if (is_final_panel) {
    top_line_state <- diagnostics$summary_state %>%
      arrange(desc(line_attractor_score), desc(frac_time), state) %>%
      slice(1) %>%
      pull(state)
  }

  line_attractor_df_2d <- if (is_final_panel && !is.null(line_attractor_df) && nrow(line_attractor_df) > 0) {
    line_attractor_df %>% filter(state %in% top_line_state)
  } else {
    NULL
  }

  line_attractor_curve_df_2d <- if (is_final_panel && !is.null(line_attractor_curve_df) && nrow(line_attractor_curve_df) > 0) {
    line_attractor_curve_df %>% filter(state %in% top_line_state)
  } else {
    NULL
  }

  plot_limits <- compute_plot_limits(list(ctx), config$data_quantile)

  plot_3d <- plot_event_3d(
    ctx,
    surface,
    event_nums = event_nums,
    line_width = 4,
    time_marker_every = 50,
    arrow_every = 10,
    cone_size_ref = 0.15,
    cone_vec_len = 0.10,
    show_time_markers = TRUE,
    show_trough_label = TRUE,
    show_endpoints = TRUE,
    show_direction_cones = TRUE,
    time_label_z_lift = 0.10,
    axis_limits = plot_limits
  )

  plot_2d <- plot_energy_2d(
    ctx,
    surface,
    event_nums = event_nums,
    n_grid_e = config$n_grid_e,
    arrow_stride = config$arrow_stride,
    arrow_scale = config$arrow_scale,
    min_arrow_len = config$min_arrow_len,
    show_energy = isTRUE(config$show_energy_background),
    data_quantile = config$data_quantile,
    mask_dist_q = config$mask_dist_q,
    line_attractor_df = line_attractor_df_2d,
    line_attractor_curve_df = line_attractor_curve_df_2d,
    plot_limits = plot_limits,
    arrow_mag_quantile = config$arrow_mag_quantile,
    show_time_labels = TRUE,
    time_label_every = 50,
    traj_offset_u = 0.004,
    traj_offset_v = -0.012,
    field_mode = config$field_mode
  )

  encoding_or_plot <- plot_state_encoding_oddsratio(
    phase_state_fisher$pairwise_results,
    phase_state_fisher$contingency_table,
    ctx$state_palette,
    config$label
  )

  save_plotly_widget(plot_3d, loaded$paths$html_3d)
  ggsave(loaded$paths$pdf_2d, plot_2d, width = 6.5, height = 6.5)
  ggsave(loaded$paths$svg_2d, plot_2d, width = 6.5, height = 6.5)
  ggsave(loaded$paths$png_2d, plot_2d, width = 6.5, height = 6.5, dpi = 300)
  ggsave(loaded$paths$state_encoding_or_png, encoding_or_plot, width = 11, height = 3.8, dpi = 300)
  ggsave(loaded$paths$state_encoding_or_pdf, encoding_or_plot, width = 11, height = 3.8)
  ggsave(loaded$paths$state_encoding_or_svg, encoding_or_plot, width = 11, height = 3.8)

  data.table::fwrite(diagnostics$diag_df, loaded$paths$diagnostics_csv)
  data.table::fwrite(diagnostics$summary_state, loaded$paths$state_summary_csv)
  data.table::fwrite(line_attr_scores, loaded$paths$line_score_csv)
  data.table::fwrite(geometry_scores, loaded$paths$geometry_score_csv)

  overall_df <- data.frame(
    p_value = phase_state_fisher$overall_test$p.value,
    method = phase_state_fisher$overall_test$method,
    alternative = phase_state_fisher$overall_test$alternative,
    stringsAsFactors = FALSE
  )
  contingency_df <- as.data.frame.matrix(phase_state_fisher$contingency_table)
  contingency_df <- cbind(phase = rownames(contingency_df), contingency_df)
  rownames(contingency_df) <- NULL

  data.table::fwrite(overall_df, loaded$paths$phase_state_overall_csv)
  data.table::fwrite(phase_state_fisher$pairwise_results, loaded$paths$phase_state_pairwise_csv)
  data.table::fwrite(contingency_df, loaded$paths$phase_state_table_csv)

  invisible(list(
    context = ctx,
    surface = surface,
    diagnostics = diagnostics,
    line_attractor_scores = line_attr_scores,
    geometry_scores = geometry_scores,
    phase_state_fisher = phase_state_fisher,
    plot_3d = plot_3d,
    plot_2d = plot_2d,
    encoding_oddsratio_plot = encoding_or_plot,
    line_attractor_df = line_attractor_df,
    line_attractor_curve_df = line_attractor_curve_df
  ))
}

summarize_day_attractor <- function(label, summary_state) {
  # Squash one run down to a single row for the cross-day table.
  type_col <- if ("type_forced" %in% colnames(summary_state)) "type_forced" else "type"
  dominant <- summary_state[which.max(summary_state$frac_time), , drop = FALSE]
  get_n <- function(t) as.integer(sum(summary_state[[type_col]] == t))

  overall_verdict <- if (dominant$frac_time > 0.3) {
    sprintf("DOMINANT %s (state %d)", dominant[[type_col]], dominant$state)
  } else if (sum(summary_state$frac_time > 0.15) > 1) {
    sprintf("mixed (%d states >15%% time)", sum(summary_state$frac_time > 0.15))
  } else {
    sprintf("no clear dominant (top: state %d %.0f%%)", dominant$state, round(100 * dominant$frac_time))
  }

  data.frame(
    label = label,
    dominant_state = dominant$state,
    dominant_type = dominant[[type_col]],
    dominant_frac = round(dominant$frac_time, 3),
    max_abs_lam = round(dominant$max_abs_lam, 4),
    line_attractor_score = round(dominant$line_attractor_score, 4),
    geometry_score = round(dominant$geometry_score, 4),
    n_line_attractor = get_n("line attractor"),
    n_point_attractor = get_n("point attractor"),
    n_plane_original = as.integer(sum(summary_state[["type"]] == "plane attractor")),
    n_unstable = get_n("unstable / saddle"),
    overall_verdict = overall_verdict,
    stringsAsFactors = FALSE
  )
}

run_analysis_batch <- function(configs) {
  # Run every configured day and write the combined cross-day summary.
  loaded_runs <- lapply(configs, load_rslds_data)
  results <- mapply(
    function(loaded, config) run_single_analysis(loaded, config),
    loaded_runs,
    configs,
    SIMPLIFY = FALSE
  )

  day_summary <- lapply(seq_along(results), function(i) {
    summarize_day_attractor(configs[[i]]$label, results[[i]]$diagnostics$summary_state)
  }) %>% bind_rows()

  data.table::fwrite(day_summary, "all_days_attractor_summary_v23.csv")
  cat("\n===== Cross-day attractor summary =====\n")
  print(day_summary[, c("label", "dominant_type", "dominant_frac", "line_attractor_score", "geometry_score", "overall_verdict")], row.names = FALSE, right = FALSE)
  cat("Saved: all_days_attractor_summary_v23.csv\n")

  invisible(results)
}

# Run the batch analysis for all configured runs.
configs <- lapply(RUN_CONFIGS, function(run_cfg) utils::modifyList(DEFAULT_CONFIG, run_cfg))
results <- run_analysis_batch(configs)
