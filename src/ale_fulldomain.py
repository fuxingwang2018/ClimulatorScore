import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

FILL       = -9999.9
FIXED_BS   = 50          # must match model's fixed batch size
MRSOL_IDX  = 20         # index of mrsol in your 22 channels — verify this!

PREDICTOR_VARS = [
    "phi500","phi700","phi850","phi950",
    "hus500","hus700","hus850","hus950",
    "ta500", "ta700", "ta850", "ta950",
    "ua500", "ua700", "ua850", "ua950",
    "va500", "va700", "va850", "va950",
    "mrsol", "tas"                          # tas here = low-res input channel
]
assert PREDICTOR_VARS[MRSOL_IDX] == "mrsol", "Check MRSOL_IDX!"

# ── 1. Load data ───────────────────────────────────────────────────────────────
ds_pred  = xr.open_dataset("predictor_1.nc")
ds_ypred = xr.open_dataset("predictant_ypred_1.nc")

# ── 2. Build full low-res array (time, y, x, channels) ────────────────────────
print("Loading low-res predictor array...")
X_lowres = np.stack(
    [ds_pred[v].values.astype(np.float32) for v in PREDICTOR_VARS],
    axis=-1
)                                           # (1460, 88, 106, 22)

# Replace all fill values with NaN then interpolate to 0 for model safety
X_lowres = np.where(
    (X_lowres == FILL) | (np.abs(X_lowres) > 1e19),
    np.nan, X_lowres
)
# Fill remaining NaNs with channel means
for c in range(X_lowres.shape[-1]):
    ch = X_lowres[..., c]
    ch_mean = np.nanmean(ch)
    X_lowres[..., c] = np.where(np.isfinite(ch), ch, ch_mean)

print(f"Low-res array shape: {X_lowres.shape}")   # (1460, 88, 106, 22)

# ── 3. Build full high-res array (time, y, x, 1) ──────────────────────────────
print("Loading high-res target array...")
tas_hires = ds_ypred["tas"].values.astype(np.float32)   # (1460, 352, 424)
tas_hires = np.where(
    (tas_hires == FILL) | (np.abs(tas_hires) > 1e19),
    np.nanmean(tas_hires), tas_hires
)
X_hires = tas_hires[..., np.newaxis]                    # (1460, 352, 424, 1)
print(f"High-res array shape: {X_hires.shape}")

# ── 4. Load generator ──────────────────────────────────────────────────────────
generator = tf.keras.models.load_model("srgan_generator.h5", compile=False)
print("Generator loaded.")

# ── 5. Verify MRSOL_IDX ───────────────────────────────────────────────────────
mrsol_data = X_lowres[..., MRSOL_IDX]                   # (1460, 88, 106)
mrsol_min  = np.nanmin(mrsol_data)
mrsol_max  = np.nanmax(mrsol_data)
print(f"mrsol range: {mrsol_min:.3f} – {mrsol_max:.3f} kg/m²")

# ── 6. Define ALE bin edges from actual mrsol distribution ────────────────────
N_BINS   = 20                                           # adjust as needed
quantiles = np.percentile(mrsol_data[np.isfinite(mrsol_data)],
                          np.linspace(0, 100, N_BINS + 1))
quantiles = np.unique(quantiles)
N_BINS    = len(quantiles) - 1
print(f"ALE bins: {N_BINS}  |  edges: {quantiles[0]:.3f} ... {quantiles[-1]:.3f}")

# ── 7. Core ALE computation ────────────────────────────────────────────────────
def predict_mean_tas(X_lr, X_hr, mrsol_value):
    """
    Set mrsol channel to mrsol_value everywhere, run generator,
    return mean predicted tas across all pixels and time steps.
    """
    X_mod = X_lr.copy()
    X_mod[..., MRSOL_IDX] = mrsol_value                # perturb mrsol globally

    n_time = X_mod.shape[0]                             # 1460
    tas_preds = []

    # Process in batches of FIXED_BS=50 (required by model)
    for start in range(0, n_time, FIXED_BS):
        end   = min(start + FIXED_BS, n_time)
        batch_lr = X_mod[start:end]                     # (≤50, 88, 106, 22)
        batch_hr = X_hr[start:end]                      # (≤50, 352, 424, 1)

        # Pad to exactly FIXED_BS if last batch is smaller
        if len(batch_lr) < FIXED_BS:                    # ← handles remainder
            pad   = FIXED_BS - len(batch_lr)
            batch_lr = np.concatenate(
                [batch_lr, np.zeros((pad, *batch_lr.shape[1:]), dtype=np.float32)], axis=0)
            batch_hr = np.concatenate(
                [batch_hr, np.zeros((pad, *batch_hr.shape[1:]), dtype=np.float32)], axis=0)
            out = generator([batch_lr, batch_hr], training=False)
            out = out[:end-start]                       # remove padding
        else:
            out = generator([batch_lr, batch_hr], training=False)

        # out shape: (batch, 352, 424, 1)
        tas_preds.append(out[..., 0].numpy())           # (batch, 352, 424)

    return np.concatenate(tas_preds, axis=0)            # (1460, 352, 424)

# ── 8. Compute ALE effects per bin ────────────────────────────────────────────
print("\nComputing ALE effects across bins...")
ale_effects = np.zeros(N_BINS)
bin_centres = 0.5 * (quantiles[:-1] + quantiles[1:])

for i in range(N_BINS):
    z_lo, z_hi = quantiles[i], quantiles[i + 1]

    pred_hi = predict_mean_tas(X_lowres, X_hires, z_hi).mean()
    pred_lo = predict_mean_tas(X_lowres, X_hires, z_lo).mean()

    ale_effects[i] = pred_hi - pred_lo
    print(f"  Bin {i+1:2d}/{N_BINS}: mrsol [{z_lo:.3f}, {z_hi:.3f}]"
          f"  Δtas = {ale_effects[i]:.4f} K")

# Accumulate and centre
ale_accumulated = np.cumsum(ale_effects)
ale_accumulated -= np.mean(ale_accumulated)             # centre around 0

# ── 9. Plot ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(bin_centres, ale_accumulated, color="steelblue", lw=2.5,
         marker="o", ms=4, label="ALE")
ax1.fill_between(bin_centres, ale_accumulated, alpha=0.15, color="steelblue")
ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.set_xlabel("Surface Soil Moisture mrsol (kg m⁻²)", fontsize=12)
ax1.set_ylabel("ALE of tas (K)", fontsize=12)
ax1.set_title("ALE: Effect of mrsol on SRGAN-predicted Near-Surface Air Temperature",
              fontsize=12)
ax1.legend(fontsize=11)

# Sample density in each bin
bin_counts = np.array([
    np.sum((mrsol_data >= quantiles[i]) & (mrsol_data < quantiles[i+1]))
    for i in range(N_BINS)
])
ax2 = ax1.twinx()
ax2.bar(bin_centres, bin_counts,
        width=np.diff(bin_centres).mean() * 0.6,
        alpha=0.15, color="grey", label="Sample count")
ax2.set_ylabel("Sample count", color="grey", fontsize=11)
ax2.tick_params(axis="y", colors="grey")

plt.tight_layout()
plt.savefig("ale_mrsol_srgan_fullgrid.png", dpi=150)
plt.show()
print("Saved: ale_mrsol_srgan_fullgrid.png")
