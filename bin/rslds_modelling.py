# This script is for fitting one rSLDS to every event, then write out the params, 
# posteriors, flow fields and a few diagnostic tables/plots that the R side picks up
# @author Huang Chenyuan, Xu Lingyu
# @date 2026-07-18

from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autograd.scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

# ssm still reaches for the old autograd.scipy.misc.logsumexp
_misc = types.ModuleType("autograd.scipy.misc")
_misc.logsumexp = logsumexp
sys.modules.setdefault("autograd.scipy.misc", _misc)

import ssm
from matplotlib.lines import Line2D
from ssm.preprocessing import interpolate_data

np.random.seed(123)

# 1. Constant set up
# Path setup
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "harmonized_data.csv"
DAY = "final"
OUT = ROOT / f"v2_new_result_44_masked_speed_test_123_{DAY}"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS_TXT = OUT / "trained_rslds_parameters.txt"
POSTERIOR_CSV = OUT / "trained_rslds_posterior_table.csv"
FLOW_PARAMS_CSV = OUT / "trained_rslds_flow_parameters.csv"
FLOW_VECTORS_CSV = OUT / "trained_rslds_flow_vectors.csv"
SUMMARY_PNG = OUT / "trained_rslds_summary.png"
FLOW_PNG = OUT / "trained_rslds_flow_field.png"
ATTRACTOR_PNG = OUT / "trained_rslds_attractors.png"
ELBO_PNG = OUT / "trained_rslds_elbo_curve.png"
SCALING_CSV = OUT / "feature_scaling_diagnostics.csv"
CA_STATS_CSV = OUT / "per_event_ca_signal_stats.csv"
MASK_CSV = OUT / "mask_diagnostics.csv"
ASINH_CSV = OUT / "per_event_asinh_scaling.csv"
STATE_COND_CSV = OUT / "state_conditioned_features.csv"
VARIANCE_CSV = OUT / "variance_decomposition.csv"

# Parameters setup
OBS_DIM = len(FEATURES)
N_STATES = 4
LATENT_DIM = 4
NUM_ITERS = 100
JITTER = 0.001
RIDGE = 1e-3

# Calcium features are normalized within events; behavioral/spatial features keep global scale.
ASINH_COLS = ["ca_signal", "ddff"]
STD_COLS = ["speed", "dist_to_water"]
RAW_COLS = ["cos_angle", "sin_angle"]
FEATURES = ASINH_COLS + STD_COLS + RAW_COLS
PRIMARY = "ca_signal"

# Mask when heading is undefined (cos==0 and sin==0)
MASKABLE = ["cos_angle", "sin_angle", "speed"]

# 2. Helpers for loading data, scaling, masking, and fitting the model
def _invert_with_mask(self, data, input=None, mask=None, tag=None):
    # Estimate initial latents from observations while respecting masked cells.
    if input is None:
        input = np.zeros((data.shape[0], self.M))
    if mask is None:
        mask = np.ones_like(data, dtype=bool)

    C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
    gram = C.T @ C + RIDGE * np.eye(C.shape[1])
    C_pinv = np.linalg.solve(gram, C.T).T
    bias = input.dot(F.T) + d

    if not mask.all():
        data = interpolate_data(data, mask)
        for _ in range(25):
            latents = (data - bias).dot(C_pinv)
            recon = latents.dot(C.T) + bias
            data = np.where(mask, data, recon)

    return (data - bias).dot(C_pinv)


def build_model():
    # Build the rSLDS and use the mask-aware inversion for latent initialization.
    model = ssm.SLDS(
        OBS_DIM, N_STATES, LATENT_DIM,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian",
    )
    model.emissions._invert = types.MethodType(_invert_with_mask, model.emissions)
    return model


def time_axis(timestamps):
    # Convert timestamps to seconds from event start, or fall back to frame index.
    t = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(t).all():
        return np.arange(len(timestamps), dtype=float)
    return t - t[0]


def fixed_point(A, b):
    # Solve the state's latent fixed point x = A x + b when it exists.
    try:
        return np.linalg.solve(np.eye(A.shape[0]) - A, b)
    except np.linalg.LinAlgError:
        return None


def state_blocks(states):
    # Yield contiguous time blocks with the same most likely discrete state.
    start = 0
    while start < len(states):
        end = start + 1
        while end < len(states) and states[end] == states[start]:
            end += 1
        yield start, end, states[start]
        start = end


def load_events():
    # Load the selected day's events, scale each feature group, 
    # and build masks for invalid heading frames.
    df = pd.read_csv(DATA_PATH)
    df = df[df["day_number"] == DAY].copy()
    print(f"Filtered day_number: {DAY}")

    has_moving = "is_moving" in df.columns
    if not has_moving:
        print("Note: no 'is_moving' column, mask diagnostics will skip it.")

    cos_i = FEATURES.index("cos_angle")
    sin_i = FEATURES.index("sin_angle")
    maskable_i = [FEATURES.index(c) for c in MASKABLE]
    asinh_i = [FEATURES.index(c) for c in ASINH_COLS]
    std_i = [FEATURES.index(c) for c in STD_COLS]

    # Drop anything with NaN/Inf
    raws, movings, ids, times, timestamps = [], [], [], [], []
    for eid, g in df.groupby("event"):
        g = g.sort_values("timestamp")
        raw = g[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            print(f"  -> skipping event {eid} (has NaN/Inf)")
            continue
        raws.append(raw)
        movings.append(g["is_moving"].to_numpy().astype(bool)
                       if has_moving else np.zeros(len(g), dtype=bool))
        ids.append(eid)
        times.append(time_axis(g["timestamp"]))
        timestamps.append(g["timestamp"].to_numpy())

    all_raw = np.concatenate(raws)

    # Speed / dist_to_water are always observed, so fit the scaler on everything
    std_scaler = StandardScaler().fit(all_raw[:, std_i])
    print(f"std_scaler fit on all {all_raw.shape[0]} rows.")

    sequences, masks, asinh_rows = [], [], []
    for raw in raws:
        scaled = raw.copy()

        # ca_signal, ddff: arcsinh then z-score within this event
        a = np.arcsinh(raw[:, asinh_i])
        mean, std = a.mean(0), a.std(0) + 1e-8
        scaled[:, asinh_i] = (a - mean) / std

        # speed, dist_to_water: global StandardScaler
        scaled[:, std_i] = std_scaler.transform(raw[:, std_i])

        # Jitter to keep things non-degenerate
        scaled += np.random.randn(*scaled.shape) * JITTER
        sequences.append(scaled)

        # hide cos/sin/speed on masked frames
        m = np.ones_like(scaled, dtype=bool)
        no_heading = (raw[:, cos_i] == 0) & (raw[:, sin_i] == 0)
        m[np.ix_(no_heading, maskable_i)] = False
        masks.append(m)

        asinh_rows.append({
            "event": ids[len(sequences) - 1],
            **{f"{n}_asinh_mean": float(mean[k]) for k, n in enumerate(ASINH_COLS)},
            **{f"{n}_asinh_std": float(std[k]) for k, n in enumerate(ASINH_COLS)},
        })

    write_diagnostics(sequences, masks, raws, movings, ids, all_raw,
                      cos_i, sin_i, asinh_rows)
    print(f"Total events loaded: {len(sequences)}")
    return sequences, ids, times, timestamps, masks


def write_diagnostics(sequences, masks, raws, movings, ids, all_raw,
                      cos_i, sin_i, asinh_rows):
    # Save pre-fit diagnostics for sanity check: scaling, masking, collinearity, and raw calcium outliers.
    scaled = np.concatenate(sequences)
    mask = np.concatenate(masks)

    # how each scaled feature looks, counting observed cells only
    print("\n=== Feature scaling diagnostics (observed cells only) ===")
    rows = []
    for i, name in enumerate(FEATURES):
        if name in ASINH_COLS:
            method = "asinh+zscore(event)"
        elif name in STD_COLS:
            method = "std(global)"
        else:
            method = "none"
        obs = scaled[mask[:, i], i]
        print(f"  {name:<16}{method:<22} n={obs.size:>7} "
              f"mean={obs.mean():+.3f} std={obs.std():.3f} "
              f"min={obs.min():+.3f} max={obs.max():+.3f}")
        rows.append({
            "feature": name, "method": method,
            "n_observed": int(obs.size), "n_masked": int((~mask[:, i]).sum()),
            "mean_observed": float(obs.mean()), "std_observed": float(obs.std()),
            "min_observed": float(obs.min()), "max_observed": float(obs.max()),
        })
    pd.DataFrame(rows).to_csv(SCALING_CSV, index=False)

    # Calculate how much and where are masked
    total = mask.size
    hidden = int((~mask).sum())
    print(f"\n=== Mask diagnostics ===\nMasked {hidden}/{total} cells "
          f"({100 * hidden / total:.2f}%), only on no-heading frames.")
    mask_rows = []
    for eid, mov, raw, m in zip(ids, movings, raws, masks):
        no_heading = (raw[:, cos_i] == 0) & (raw[:, sin_i] == 0)
        mask_rows.append({
            "event": eid, "len": len(raw),
            "n_invalid_angle": int(no_heading.sum()),
            "frac_invalid_angle": float(no_heading.mean()),
            "n_invalid_in_moving": int((no_heading & mov).sum()),
            "n_invalid_in_static": int((no_heading & ~mov).sum()),
            "n_moving": int(mov.sum()), "frac_moving": float(mov.mean()),
            "n_masked_cells": int((~m).sum()),
            "frac_masked_cells": float((~m).sum() / m.size),
        })
    pd.DataFrame(mask_rows).to_csv(MASK_CSV, index=False)

    # Check whether the model input feature matrix is too collinear.
    s = np.linalg.svd(scaled - scaled.mean(0), compute_uv=False)
    cond = s[0] / max(s[-1], 1e-12)
    print(f"Design matrix condition number: {cond:.2f} (>30 = too collinear)")

    pd.DataFrame(asinh_rows).to_csv(ASINH_CSV, index=False)

    # Raw ca_signal distribution to check outliers
    ca = all_raw[:, FEATURES.index("ca_signal")]
    print("\n=== Raw ca_signal distribution (before scaling) ===")
    for p in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
        print(f"  p{p:>5}: {np.percentile(ca, p):>10.3f}")
    print(f"  mean={ca.mean():.3f} std={ca.std():.3f}")

    # Per-event ca_signal stats, biggest swings first
    ca_i = FEATURES.index("ca_signal")
    ca_rows = [{
        "event": eid, "len": len(raw),
        "min": float(raw[:, ca_i].min()), "max": float(raw[:, ca_i].max()),
        "p99": float(np.percentile(raw[:, ca_i], 99)),
        "std": float(raw[:, ca_i].std()),
        "abs_max": float(np.abs(raw[:, ca_i]).max()),
    } for raw, eid in zip(raws, ids)]
    ca_df = pd.DataFrame(ca_rows).sort_values("abs_max", ascending=False)
    print("\n=== Per-event ca_signal stats (top 10 by |max|) ===")
    print(ca_df.head(10).to_string(index=False))
    ca_df.to_csv(CA_STATS_CSV, index=False)


def fit_model(model, sequences, masks):
    # Fit the rSLDS with masked observations and return ELBOs plus posterior states.
    print(f"\nFitting rSLDS: OBS_DIM={OBS_DIM}, N_STATES={N_STATES}, "
          f"LATENT_DIM={LATENT_DIM}")
    elbos, posterior = model.fit(
        sequences, masks=masks,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=NUM_ITERS, alpha=0.5, initialize=True,
    )
    elbos = np.asarray(elbos, dtype=float).reshape(-1)
    print(f"Done. Final ELBO: {elbos[-1]:.2f}")

    # which observations does the latent space actually explain?
    C = model.emissions.Cs[0]
    print("\n=== Emission matrix C row norms ===")
    for i, name in enumerate(FEATURES):
        print(f"  {name:<16} ||C[{i}]||={np.linalg.norm(C[i]):.3f}  "
              f"row={np.array2string(C[i], precision=3)}")

    return elbos, posterior


def collect_posteriors(posterior, sequences, masks, ids, times, timestamps):
    # Collect fitted latent trajectories and state probabilities for each event.
    q_x = posterior.mean_continuous_states
    q_z = posterior.mean_discrete_states
    summaries = []
    for eid, data, mask, t, ts, x, z in zip(
        ids, sequences, masks, times, timestamps, q_x, q_z
    ):
        z = np.asarray(z)
        summaries.append({
            "event_id": eid, "data": data, "mask": mask,
            "time": t, "timestamp": ts,
            "q_x": x, "q_z": z, "z": np.argmax(z, axis=1),
        })
    return summaries


def export_posterior_table(summaries):
    rows = []
    for s in summaries:
        for t in range(len(s["data"])):
            row = {
                "event_id": s["event_id"], "time_index": t,
                "relative_time": float(s["time"][t]),
                "original_timestamp": s["timestamp"][t],
                "state_most_likely": int(s["z"][t]),
            }
            for j, name in enumerate(FEATURES):
                row[f"observed_{name}"] = float(s["data"][t, j])
                row[f"observed_{name}_is_masked"] = bool(not s["mask"][t, j])
            for i in range(s["q_x"].shape[1]):
                row[f"latent_{i + 1}"] = float(s["q_x"][t, i])
            for k in range(s["q_z"].shape[1]):
                row[f"state_prob_{k}"] = float(s["q_z"][t, k])
            rows.append(row)
    pd.DataFrame(rows).to_csv(POSTERIOR_CSV, index=False)
    print(f"Posterior table saved to {POSTERIOR_CSV}")


def build_grid(summaries):
    # Build a 2D latent grid and state-specific latent clouds for flow visualization.
    latents = np.concatenate([s["q_x"] for s in summaries])
    times = np.concatenate([s["time"] for s in summaries])

    clouds = []
    for k in range(N_STATES):
        pts = [s["q_x"][s["z"] == k] for s in summaries if np.any(s["z"] == k)]
        clouds.append(np.concatenate(pts) if pts else np.empty((0, LATENT_DIM)))

    pad = 0.5
    gx = np.linspace(latents[:, 0].min() - pad, latents[:, 0].max() + pad, 40)
    gy = np.linspace(latents[:, 1].min() - pad, latents[:, 1].max() + pad, 40)
    X, Y = np.meshgrid(gx, gy)
    plane = np.column_stack([X.ravel(), Y.ravel()])

    return {
        "global_mean": latents.mean(0), "clouds": clouds,
        "gx": gx, "gy": gy, "X": X, "Y": Y, "plane": plane,
        "pad": pad, "z_base": float(times.min()), "stride": 4,
    }


def state_flow(model, grid, k):
    # Compute where state k pushes latent points on a 2D slice of the latent space.
    cloud = grid["clouds"][k]
    center = cloud.mean(0) if len(cloud) else grid["global_mean"].copy()

    full = np.repeat(center[None, :], grid["plane"].shape[0], axis=0)
    full[:, 0] = grid["plane"][:, 0]
    full[:, 1] = grid["plane"][:, 1]

    A, b = model.dynamics.As[k], model.dynamics.bs[k]
    nxt = full @ A.T + b
    return {
        "center": center, "full": full, "next": nxt, "vel": nxt - full,
        "fp": fixed_point(A, b),
        "radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "cloud": cloud, "A": A, "b": b,
    }


def export_flow_tables(model, summaries):
    grid = build_grid(summaries)
    param_rows, vec_rows = [], []

    for k in range(N_STATES):
        f = state_flow(model, grid, k)
        prow = {
            "state_index": k, "spectral_radius": f["radius"],
            "is_stable": bool(f["radius"] < 1.0),
            "grid_x_min": float(grid["gx"][0]), "grid_x_max": float(grid["gx"][-1]),
            "grid_y_min": float(grid["gy"][0]), "grid_y_max": float(grid["gy"][-1]),
            "grid_size_x": len(grid["gx"]), "grid_size_y": len(grid["gy"]),
            "padding": grid["pad"], "quiver_stride": grid["stride"],
            "z_base": grid["z_base"],
        }
        for i in range(LATENT_DIM):
            prow[f"reference_point_latent_{i + 1}"] = float(f["center"][i])
            prow[f"bias_latent_{i + 1}"] = float(f["b"][i])
            prow[f"fixed_point_latent_{i + 1}"] = (
                np.nan if f["fp"] is None else float(f["fp"][i]))
            for j in range(LATENT_DIM):
                prow[f"A_{i + 1}_{j + 1}"] = float(f["A"][i, j])
        param_rows.append(prow)

        for gi, p in enumerate(grid["plane"]):
            row = {
                "state_index": k, "grid_index": gi,
                "grid_x": float(p[0]), "grid_y": float(p[1]),
                "grid_z": grid["z_base"],
                "velocity_latent_1": float(f["vel"][gi, 0]),
                "velocity_latent_2": float(f["vel"][gi, 1]),
                "speed_2d": float(np.linalg.norm(f["vel"][gi, :2])),
            }
            for i in range(LATENT_DIM):
                row[f"reference_point_latent_{i + 1}"] = float(f["center"][i])
                row[f"grid_latent_{i + 1}"] = float(f["full"][gi, i])
                row[f"next_latent_{i + 1}"] = float(f["next"][gi, i])
                row[f"velocity_latent_{i + 1}_full"] = float(f["vel"][gi, i])
            vec_rows.append(row)

    pd.DataFrame(param_rows).to_csv(FLOW_PARAMS_CSV, index=False)
    pd.DataFrame(vec_rows).to_csv(FLOW_VECTORS_CSV, index=False)
    print(f"Flow parameters saved to {FLOW_PARAMS_CSV}")
    print(f"Flow vectors saved to {FLOW_VECTORS_CSV}")


def save_parameter_summary(model, final_elbo):
    C, d = model.emissions.Cs[0], model.emissions.ds[0]
    with PARAMS_TXT.open("w", encoding="utf-8") as f:
        f.write("Trained rSLDS parameters\n")
        f.write(f"Day filter: {DAY}\n")
        f.write(f"Feature columns: {FEATURES}\n")
        f.write(f"asinh + within-event z-score: {ASINH_COLS}\n")
        f.write(f"StandardScaler (static-only): {STD_COLS}\n")
        f.write(f"unscaled (cyclic): {RAW_COLS}\n")
        f.write(f"Maskable when (cos==0)&(sin==0): {MASKABLE}\n")
        f.write(f"OBS_DIM: {OBS_DIM}\n")
        f.write(f"N_STATES: {N_STATES}\n")
        f.write(f"LATENT_DIM: {LATENT_DIM}\n")
        f.write(f"Final ELBO (all events): {final_elbo:.2f}\n\n")
        f.write("Emission matrix C (rows = observations, cols = latents):\n")
        for i, name in enumerate(FEATURES):
            f.write(f"  {name:<18} C_row = {np.array2string(C[i], precision=4)}, "
                    f"d = {d[i]:.4f}, ||C_row|| = {np.linalg.norm(C[i]):.4f}\n")
        f.write("\n")

        for k in range(N_STATES):
            A, b = model.dynamics.As[k], model.dynamics.bs[k]
            radius = float(np.max(np.abs(np.linalg.eigvals(A))))
            fp = fixed_point(A, b)
            f.write(f"State {k}: spectral_radius={radius:.4f}, "
                    f"stable={radius < 1.0}\n")
            if fp is None:
                f.write("State fixed point: not solvable because (I - A) is singular.\n")
            else:
                f.write("State fixed point (all latent dims): "
                        f"[{', '.join(f'{v:.4f}' for v in fp)}]\n")
    print(f"Parameter summary saved to {PARAMS_TXT}")


def plot_elbo(elbos):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    it = np.arange(1, elbos.size + 1)
    ax.plot(it, elbos, color="#1f77b4", lw=2)
    ax.scatter(it, elbos, color="#1f77b4", s=18, alpha=0.75)
    ax.set_title("Training ELBO by Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    ax.grid(alpha=0.25)

    if elbos.size > 1:
        w = min(10, elbos.size)
        ax.annotate(
            f"Last {w - 1} iters delta = {elbos[-1] - elbos[-w]:.2f}",
            xy=(it[-1], elbos[-1]), xytext=(-140, 20), textcoords="offset points",
            fontsize=9, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.85},
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "0.35"},
        )
    plt.tight_layout()
    fig.savefig(ELBO_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"ELBO plot saved to {ELBO_PNG}")


def pick_events(summaries, n=8):
    if len(summaries) <= n:
        return summaries
    idx = np.linspace(0, len(summaries) - 1, n, dtype=int)
    return [summaries[i] for i in idx]


def plot_summary(summaries):
    sampled = pick_events(summaries)
    colors = plt.cm.tab10(np.linspace(0, 1, max(N_STATES, 3)))

    fig = plt.figure(figsize=(15, 2.0 * max(OBS_DIM, 3) + 1.0))
    gs = fig.add_gridspec(OBS_DIM, 2, width_ratios=[1.0, 1.4])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axes = [fig.add_subplot(gs[i, 1]) for i in range(OBS_DIM)]

    offset = 0
    for s in sampled:
        x, z = s["q_x"], s["z"]
        t = np.arange(len(x))
        ax3d.plot(x[:, 0], x[:, 1], t, color="0.55", lw=0.8, alpha=0.6)
        ax3d.scatter(x[:, 0], x[:, 1], t, c=z, cmap="jet", s=12, alpha=0.45)

        xs = np.arange(len(s["data"])) + offset
        for j, name in enumerate(FEATURES):
            ax = axes[j]
            col = s["data"][:, j]
            visible = np.where(s["mask"][:, j], col, np.nan)
            hidden = np.where(s["mask"][:, j], np.nan, col)
            ax.plot(xs, visible, color="black", lw=1.0, alpha=0.85)
            ax.plot(xs, hidden, color="0.7", lw=0.8, alpha=0.6, ls=":")
            if name == PRIMARY:
                ax.plot(xs, x[:, 0], color="#ff5a5a", ls="--", lw=1.0, alpha=0.95)
            for a, b, st in state_blocks(z):
                ax.axvspan(offset + a, offset + b, color=colors[st], alpha=0.13, lw=0)
            ax.axvline(offset + len(s["data"]), color="0.55", ls=":", lw=1.5)
        offset += len(s["data"])

    ax3d.set_xlabel("Latent Dim 1")
    ax3d.set_ylabel("Latent Dim 2")
    ax3d.set_zlabel("Time")
    ax3d.set_title("Latent Trajectories (Sampled Events)")

    for j, name in enumerate(FEATURES):
        axes[j].set_ylabel(name, fontsize=9)
        if j == 0:
            axes[j].set_title(f"Inferred States ({len(sampled)} Sampled Events)")
        if j == OBS_DIM - 1:
            axes[j].set_xlabel("Concatenated Time Steps")
        else:
            axes[j].set_xticklabels([])

    axes[0].legend(handles=[
        Line2D([0], [0], color="black", lw=1.5, label="Observed"),
        Line2D([0], [0], color="0.7", lw=1.0, ls=":", label="Masked"),
        Line2D([0], [0], color="#ff5a5a", lw=1.2, ls="--", label="Latent Dim 1"),
    ], loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary plot saved to {SUMMARY_PNG}")


def plot_flow_field(model, summaries):
    grid = build_grid(summaries)
    fig, axes = plt.subplots(1, N_STATES, figsize=(6 * N_STATES, 5), squeeze=False)
    axes = axes.ravel()

    contour = None
    for k, ax in enumerate(axes):
        f = state_flow(model, grid, k)
        u = f["vel"][:, 0].reshape(grid["X"].shape)
        v = f["vel"][:, 1].reshape(grid["Y"].shape)
        speed = np.sqrt(u**2 + v**2)

        contour = ax.contourf(grid["X"], grid["Y"], speed,
                              levels=18, cmap="magma", alpha=0.65)
        ax.streamplot(grid["gx"], grid["gy"], u, v,
                      color="white", density=1.0, linewidth=0.8, arrowsize=0.9)
        if len(f["cloud"]):
            ax.scatter(f["cloud"][:, 0], f["cloud"][:, 1],
                       s=10, color="black", alpha=0.18)
        if f["fp"] is not None:
            ax.scatter(f["fp"][0], f["fp"][1], marker="*", s=220, color="cyan",
                       edgecolor="black", linewidth=0.7, zorder=5)
        ax.set_title(f"State {k}\nspectral radius={f['radius']:.3f}")
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")

    fig.colorbar(contour, ax=axes.tolist(), shrink=0.8, label="Flow speed")
    fig.suptitle("State-wise Latent Flow Field", fontsize=14)
    plt.tight_layout()
    fig.savefig(FLOW_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Flow plot saved to {FLOW_PNG}")


def plot_attractors(model, summaries):
    grid = build_grid(summaries)
    colors = plt.cm.tab10(np.linspace(0, 1, max(N_STATES, 3)))
    fig, ax = plt.subplots(figsize=(8, 7))

    for k in range(N_STATES):
        f = state_flow(model, grid, k)
        if len(f["cloud"]):
            ax.scatter(f["cloud"][:, 0], f["cloud"][:, 1], s=14, alpha=0.18,
                       color=colors[k], label=f"State {k} cloud")
        ax.scatter(f["center"][0], f["center"][1], s=70, color=colors[k],
                   edgecolor="black", linewidth=0.5, zorder=4)
        if f["fp"] is not None:
            ax.scatter(f["fp"][0], f["fp"][1], marker="*", s=260, color=colors[k],
                       edgecolor="black", linewidth=0.8, zorder=5)
            ax.annotate(f"S{k}\n$r={f['radius']:.2f}$",
                        xy=(f["fp"][0], f["fp"][1]), xytext=(8, 8),
                        textcoords="offset points", fontsize=10, weight="bold")
            ax.annotate("", xy=(f["fp"][0], f["fp"][1]),
                        xytext=(f["center"][0], f["center"][1]),
                        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": colors[k]})

    ax.set_title("Latent Attractors / Fixed Points")
    ax.set_xlabel("Latent Dim 1")
    ax.set_ylabel("Latent Dim 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    fig.savefig(ATTRACTOR_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Attractor plot saved to {ATTRACTOR_PNG}")


def diagnose_state_conditioned(summaries):
    # Check whether discrete states separate observed feature values.
    z = np.concatenate([s["z"] for s in summaries])
    y = np.concatenate([s["data"] for s in summaries])
    mask = np.concatenate([s["mask"] for s in summaries])

    print("\n" + "=" * 96)
    print("=== State-conditioned feature statistics ===")
    print("(how much of a feature's variation the discrete state explains)")
    print("=" * 96)

    rows = []
    for j, name in enumerate(FEATURES):
        means = np.full(N_STATES, np.nan)
        stds = np.full(N_STATES, np.nan)
        ns = np.zeros(N_STATES, dtype=int)
        for k in range(N_STATES):
            sel = (z == k) & mask[:, j]
            ns[k] = sel.sum()
            if ns[k] > 0:
                means[k] = y[sel, j].mean()
                stds[k] = y[sel, j].std()

        rng = 0.0 if np.all(np.isnan(means)) else float(np.nanmax(means) - np.nanmin(means))
        # Thresholds are rough diagnostics on scaled features: ~0.2 weak, ~0.5 strong state separation.
        if rng > 0.5:
            verdict = "carried by state ***"
        elif rng > 0.2:
            verdict = "weakly state-related"
        else:
            verdict = "NOT state-related"
        print(f"  {name:<16} range={rng:6.3f}   {verdict}")

        row = {"feature": name, "range_of_means": rng, "verdict": verdict}
        for k in range(N_STATES):
            row[f"state_{k}_mean"] = None if np.isnan(means[k]) else float(means[k])
            row[f"state_{k}_std"] = None if np.isnan(stds[k]) else float(stds[k])
            row[f"state_{k}_n"] = int(ns[k])
        rows.append(row)

    print("\n--- State occupancy ---")
    for k in range(N_STATES):
        n = int((z == k).sum())
        print(f"  State {k}: n={n} ({100 * n / len(z):.2f}%)")

    print("\n--- Pairwise state-mean distance ---")
    centroids = np.zeros((N_STATES, len(FEATURES)))
    for k in range(N_STATES):
        for j in range(len(FEATURES)):
            sel = (z == k) & mask[:, j]
            if sel.sum() > 0:
                centroids[k, j] = y[sel, j].mean()
    for a in range(N_STATES):
        for b in range(a + 1, N_STATES):
            dist = float(np.linalg.norm(centroids[a] - centroids[b]))
            flag = "  <-- DEGENERATE" if dist < 0.5 else ""
            print(f"  ||mean(S{a}) - mean(S{b})|| = {dist:.3f}{flag}")

    pd.DataFrame(rows).to_csv(STATE_COND_CSV, index=False)
    print(f"\nState-conditioned stats saved to {STATE_COND_CSV}")


def diagnose_variance(model, summaries):
    # Post-hoc diagnostic, not a formal variance decomposition model.
    # 1. frac_lat is corr(y, y_hat)^2, so it is a reconstruction correlation, not strict R2.
    # 2. frac_state is between-state variance using the most likely state z = argmax(q_z).
    # 3. Latent and state fractions can overlap, so they should not be summed as 100% variance.
    # 4. Only unmasked observations are used, and uncertain state assignments make this rougher.
    C, d = model.emissions.Cs[0], model.emissions.ds[0]
    y = np.concatenate([s["data"] for s in summaries])
    x = np.concatenate([s["q_x"] for s in summaries])
    z = np.concatenate([s["z"] for s in summaries])
    mask = np.concatenate([s["mask"] for s in summaries])
    y_hat = x @ C.T + d   # what the latent alone predicts

    print("\n" + "=" * 96)
    print("=== Variance decomposition per feature ===")
    print("=" * 96)
    print(f"{'feature':<16}{'var(y)':>10}{'frac_latent':>13}"
          f"{'frac_state':>13}{'residual':>11}    diagnosis")
    print("-" * 96)

    rows = []
    for j, name in enumerate(FEATURES):
        sel = mask[:, j]
        if sel.sum() < 5:
            continue
        yj = y[sel, j]
        var_y = float(yj.var())

        yh = y_hat[sel, j]
        if yj.std() > 1e-9 and yh.std() > 1e-9:
            frac_lat = float(np.corrcoef(yj, yh)[0, 1]) ** 2
        else:
            frac_lat = 0.0

        # between-state variance / total variance
        between = 0.0
        for k in range(N_STATES):
            sk = sel & (z == k)
            if sk.sum() > 0:
                between += sk.sum() * (y[sk, j].mean() - yj.mean()) ** 2
        between /= max(sel.sum(), 1)
        frac_state = between / var_y if var_y > 1e-9 else 0.0
        residual = max(0.0, 1.0 - frac_lat - frac_state)

        if frac_lat > 0.3:
            diag = "latent encodes it"
        elif frac_state > 0.3:
            diag = "state encodes it"
        elif frac_lat + frac_state > 0.3:
            diag = "shared latent+state"
        else:
            diag = "ABSORBED INTO NOISE ***"

        print(f"{name:<16}{var_y:>10.3f}{frac_lat:>12.2%} "
              f"{frac_state:>12.2%} {residual:>10.2%}    {diag}")
        rows.append({
            "feature": name, "var_observed": var_y,
            "frac_explained_by_latent": frac_lat,
            "frac_explained_by_state": frac_state,
            "residual_frac": residual, "diagnosis": diag,
        })

    pd.DataFrame(rows).to_csv(VARIANCE_CSV, index=False)
    print(f"\nVariance decomposition saved to {VARIANCE_CSV}")


# 3. Main entry point: load data, fit model, save results, and plot diagnostics.
def main():
    sequences, ids, times, timestamps, masks = load_events()

    model = build_model()
    elbos, posterior = fit_model(model, sequences, masks)
    summaries = collect_posteriors(posterior, sequences, masks,
                                   ids, times, timestamps)

    save_parameter_summary(model, float(elbos[-1]))
    export_posterior_table(summaries)
    export_flow_tables(model, summaries)

    plot_elbo(elbos)
    plot_summary(summaries)
    plot_flow_field(model, summaries)
    plot_attractors(model, summaries)

    diagnose_state_conditioned(summaries)
    diagnose_variance(model, summaries)


if __name__ == "__main__":
    main()
