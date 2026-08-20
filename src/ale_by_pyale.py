"""
Proper ALE analysis using trained SRGAN generator.
mrsol (surface soil moisture) → tas (near-surface air temperature)
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import zoom
from PyALE import ale

# ── 1. Load data ───────────────────────────────────────────────────────────────
basedir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmto_ERAI_2003_arrhenius'
ds_pred  = xr.open_dataset(f"{basedir}/predictor_1.nc")
ds_ypred = xr.open_dataset(f"{basedir}/predictant_ypred_1.nc")
ds_ytest = xr.open_dataset(f"{basedir}/predictant_ytest_1.nc")

FILL = -9999.9

# ── 2. Define all predictor variable names (matching SRGAN input channels) ─────
PREDICTOR_VARS = [
    "phi500","phi700","phi850","phi950",
    "hus500","hus700","hus850","hus950",
    "ta500", "ta700", "ta850", "ta950",
    "ua500", "ua700", "ua850", "ua950",
    "va500", "va700", "va850", "va950",
    "mrsol", "tas"       # tas here is low-res input (12km), not output
]
MRSOL_IDX = PREDICTOR_VARS.index("mrsol")  # index in the channel dimension

# ── 3. Flatten (time, y, x) → (N, n_features) at LOW-RES grid ────────────────
def extract_flat(ds, variables, fill=FILL):
    arrays = []
    for v in variables:
        arr = ds[v].values.astype(np.float32)        # (time, y, x)
        arr = np.where(arr == fill, np.nan, arr)
        arrays.append(arr.reshape(-1))               # flatten
    X = np.stack(arrays, axis=1)                     # (N, n_features)
    return X

X_all = extract_flat(ds_pred, PREDICTOR_VARS)        # (1460*88*106, n_features)

# ── 4. Remove rows with any NaN ────────────────────────────────────────────────
valid_mask = np.all(np.isfinite(X_all), axis=1)
X_valid = X_all[valid_mask]
print(f"Valid samples: {valid_mask.sum():,}")

# ── 5. Subsample for speed ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)
N = min(50_000, len(X_valid))
idx = rng.choice(len(X_valid), size=N, replace=False)
X_sample = X_valid[idx]                              # (N, n_features)

# ── 6. Load trained SRGAN generator ───────────────────────────────────────────
# Adjust path and loading method to match how you saved your model
basedir_generator = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmto_ERAI_2003_arrhenius'
file_generator = 'model_1_generator.h5'
generator = tf.keras.models.load_model(basedir_generator + '/' + file_generator, compile=False)
# Or if saved as SavedModel:
# generator = tf.saved_model.load("srgan_generator_savedmodel/")

# ── 7. Define predict function for PyALE ──────────────────────────────────────
# SRGAN takes spatial patches (batch, H, W, C); here we approximate by
# treating each flattened pixel as a 1×1 patch — adjust patch_size if needed.

PATCH_SIZE = 1   # set to your actual patch size, e.g. 8 or 16
BATCH_SIZE = 512

def predict_tas(X_df):
    """
    Takes a DataFrame with columns = PREDICTOR_VARS.
    Returns predicted tas as 1D numpy array.
    SRGAN input shape: (batch, patch_H, patch_W, n_channels)
    """
    X_np = X_df[PREDICTOR_VARS].values.astype(np.float32)  # (N, n_features)
    n = len(X_np)
    preds = []

    for start in range(0, n, BATCH_SIZE):
        batch = X_np[start:start + BATCH_SIZE]
        # Reshape to (batch, 1, 1, n_channels) — single-pixel patch
        batch_4d = batch[:, np.newaxis, np.newaxis, :]      # (B, 1, 1, C)
        out = generator(batch_4d, training=False)            # (B, scale, scale, 1)
        # Take centre pixel of the upscaled output
        cy = out.shape[1] // 2
        cx = out.shape[2] // 2
        tas_out = out[:, cy, cx, 0].numpy()                  # (B,)
        preds.append(tas_out)

    return np.concatenate(preds)

# ── NEW: wrap function in a class so PyALE can find .predict() ────────────────
class SRGANWrapper:                          # ← NEW
    def predict(self, X_df):                 # ← NEW
        return predict_tas(X_df)             # ← NEW

srgan_model = SRGANWrapper()                 # ← NEW


# ── 8. Build DataFrame for PyALE ──────────────────────────────────────────────
df_sample = pd.DataFrame(X_sample, columns=PREDICTOR_VARS)

# Run this to see both input shapes and names
print("Number of inputs:", len(generator.inputs))
for i, inp in enumerate(generator.inputs):
    print(f"  Input {i}: name='{inp.name}', shape={inp.shape}, dtype={inp.dtype}")

# ── DEBUG: test wrapper before passing to PyALE ───────────────────────────────
print("Testing SRGANWrapper.predict() independently...")

test_input = df_sample.head(10)              # take just 10 rows
print("Input shape to predict:", test_input.shape)
print("Input columns:", test_input.columns.tolist())

try:
    test_output = srgan_model.predict(test_input)
    print("Output shape:", test_output.shape)  # should be (10,)
    print("Output sample:", test_output[:3])
    print(" Wrapper works fine")
except Exception as e:
    print(f" Wrapper failed: {e}")

# ── 9. Run 1D ALE for mrsol ───────────────────────────────────────────────────
print("Computing 1D ALE for mrsol...")
ale_eff = ale(
    X=df_sample,
    #model=predict_tas,
    model=srgan_model,
    feature=["mrsol"],
    grid_size=50,
    plot=True,
    #plot_params={
    #    "line_kw": {"color": "steelblue", "lw": 2},
    #    "fill_between_kw": {"alpha": 0.2},
    #}
)

fig, ax1 = plt.subplots(figsize=(8, 5))

# ALE curve
ax1.plot(ale_eff.index, ale_eff["eff"], color="steelblue", lw=2, label="ALE")
ax1.fill_between(ale_eff.index, ale_eff["eff"], alpha=0.15, color="steelblue")
ax1.axhline(0, color="black", lw=0.8, ls="--")
ax1.set_xlabel("Surface Soil Moisture mrsol (kg m⁻²)")
ax1.set_ylabel("ALE of tas (K)")
ax1.set_title("ALE: Effect of mrsol on SRGAN-predicted tas")
ax1.legend(loc="upper right")

# Sample count as background bars
ax2 = ax1.twinx()
ax2.bar(ale_eff.index, ale_eff["size"],
        width=np.diff(ale_eff.index).mean(),
        alpha=0.15, color="grey", label="Sample count")
ax2.set_ylabel("Sample count", color="grey")
ax2.tick_params(axis="y", colors="grey")

plt.xlabel("Surface Soil Moisture mrsol (kg m⁻²)")
plt.ylabel("ALE of tas (K)")
plt.title("ALE: Effect of mrsol on SRGAN-predicted tas")
plt.tight_layout()
plt.savefig("ale_mrsol_srgan.png", dpi=150)
plt.show()


# ── 10. Optional: 2D ALE (mrsol × ta950 interaction) ─────────────────────────
print("Computing 2D ALE for mrsol × ta950...")
ale_2d = ale(
    X=df_sample,
    model=predict_tas,
    feature=["mrsol", "ta950"],
    grid_size=20,
    plot=True,
)
plt.title("2D ALE: mrsol × ta950 interaction on SRGAN tas")
plt.tight_layout()
plt.savefig("ale_mrsol_ta950_2d.png", dpi=150)
plt.show()

# ── 11. Comparison: ALE on SRGAN output vs ground truth (no model needed) ─────
# This is the model-free approach from before — useful as a cross-check
print("Computing observational ALE on saved outputs...")

tas_pred_flat = ds_ypred["tas"].values.ravel().astype(np.float32)
tas_test_flat = ds_ytest["tas"].values.ravel().astype(np.float32)

# Upsample mrsol to high-res grid for alignment
mrsol_lr = ds_pred["mrsol"].values.astype(np.float32)
mrsol_lr = np.where(mrsol_lr == FILL, np.nan, mrsol_lr)
scale_y = ds_ypred.dims["y"] / ds_pred.dims["y"]   # 352/88 = 4
scale_x = ds_ypred.dims["x"] / ds_pred.dims["x"]   # 424/106 = 4
mrsol_hr = zoom(mrsol_lr, (1, scale_y, scale_x), order=1).ravel()

valid2 = (
    np.isfinite(mrsol_hr) &
    (tas_pred_flat != FILL) & np.isfinite(tas_pred_flat) &
    (tas_test_flat != FILL) & np.isfinite(tas_test_flat)
)
idx2 = rng.choice(valid2.sum(), size=min(200_000, valid2.sum()), replace=False)

df_obs = pd.DataFrame({
    "mrsol":    mrsol_hr[valid2][idx2],
    "tas_pred": tas_pred_flat[valid2][idx2],
    "tas_test": tas_test_flat[valid2][idx2],
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, target_col, label, color in zip(
    axes,
    ["tas_pred", "tas_test"],
    ["SRGAN output", "Ground truth (dynamical model)"],
    ["steelblue",   "darkorange"],
):
    df_tmp = df_obs[["mrsol", target_col]].rename(columns={target_col: "tas"})

    class _Wrapper:
        def predict(self, X):
            return X["tas"].values

    ale_obs = ale(
        X=df_tmp, model=_Wrapper(),
        feature=["mrsol"], grid_size=50, plot=False
    )
    ax.plot(ale_obs.index, ale_obs["eff"], color=color, lw=2)
    ax.fill_between(ale_obs.index, ale_obs["eff"], alpha=0.15, color=color)
    ax.axhline(0, ls="--", lw=0.8, color="k")
    ax.set_xlabel("mrsol (kg m⁻²)")
    ax.set_ylabel("ALE of tas (K)")
    ax.set_title(f"Observational ALE\n{label}")

plt.tight_layout()
plt.savefig("ale_mrsol_observational_comparison.png", dpi=150)
plt.show()
