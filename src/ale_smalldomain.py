import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pandas as pd

# ── masking approach for fixed-size model ────────────────────────────────
def predict_mean_tas_smalldomain(X_lr_full, X_hr_full, mrsol_value, MRSOL_IDX, FIXED_BS, generator, \
        hr_r0, hr_r1, hr_c0, hr_c1, \
        tas_mean = 0, tas_std = 1):
    """
    X_lr_full: full low-res array  (1460, 88, 106, 22)
    X_hr_full: full high-res array (1460, 352, 424, 1)
    Perturbs mrsol, runs full-domain model, returns mean tas
    over the subdomain only.
    """
    X_mod = X_lr_full.copy()
    X_mod[..., MRSOL_IDX] = mrsol_value               # perturb full domain

    n_time = X_mod.shape[0]
    tas_preds = []

    for start in range(0, n_time, FIXED_BS):
        end      = min(start + FIXED_BS, n_time)
        batch_lr = X_mod[start:end]
        batch_hr = X_hr_full[start:end]

        if len(batch_lr) < FIXED_BS:
            pad      = FIXED_BS - len(batch_lr)
            batch_lr = np.concatenate([batch_lr,
                np.zeros((pad, *batch_lr.shape[1:]), dtype=np.float32)], axis=0)
            batch_hr = np.concatenate([batch_hr,
                np.zeros((pad, *batch_hr.shape[1:]), dtype=np.float32)], axis=0)
            out = generator([batch_lr, batch_hr], training=False)
            out = out[:end-start]
        else:
            out = generator([batch_lr, batch_hr], training=False)

        # ── CHANGED: de-normalise output ──────────────────────────────────────
        out = out * tas_std + tas_mean                         # ← NEW: now in Kelvin

        # ── CHANGED: extract subdomain from full output before averaging ───────
        out_sub = out[:, hr_r0:hr_r1, hr_c0:hr_c1, 0]  # ← NEW crop output
        tas_preds.append(out_sub.numpy())                # (batch, hr_rows, hr_cols)

    return np.concatenate(tas_preds, axis=0)             # (1460, hr_rows, hr_cols)


def predict_mean_tas_cnn_smalldomain_nonormalize(X_lr_full, mrsol_value, # ← CHANGED: no X_hr_full
                                  mrsol_idx, fixed_bs,
                                  generator,
                                  hr_r0, hr_r1, hr_c0, hr_c1):
    """
    CNN version:
    - Single input:  (fixed_bs, 88, 106, 21)
    - Single output: (fixed_bs, 149248) → reshape to (fixed_bs, 352, 424)
    - Returns mean tas over subdomain only: (n_time, hr_H, hr_W)
    """
    #assert X_lr_full.shape[1:] == (88, 106, 21), \       # ← CHANGED: 21 not 22
    #    f"Expected (88,106,21), got {X_lr_full.shape[1:]}"

    # ── CNN output is (None, 149248) = flat 352×424 ───────────────────────────────
    HR_H = 352                                               # ← NEW: high-res spatial dims
    HR_W = 424                                               # ← NEW
    assert HR_H * HR_W == 149248, "Check HR_H and HR_W"     # ← NEW safety check

    X_mod = X_lr_full.copy()
    X_mod[..., mrsol_idx] = mrsol_value                  # perturb mrsol globally

    n_time    = X_mod.shape[0]
    tas_preds = []

    for start in range(0, n_time, fixed_bs):
        end      = min(start + fixed_bs, n_time)
        batch_lr = X_mod[start:end].astype(np.float32)   # (≤BS, 88, 106, 21)
        actual_n = len(batch_lr)

        # Pad to fixed_bs if last batch is smaller
        if actual_n < fixed_bs:
            pad      = fixed_bs - actual_n
            batch_lr = np.concatenate([
                batch_lr,
                np.zeros((pad, 88, 106, 21), dtype=np.float32)  # ← CHANGED: 21
            ], axis=0)                                    # (fixed_bs, 88, 106, 21)

        out = generator(batch_lr, training=False)         # ← CHANGED: single input
                                                          # out shape: (fixed_bs, 149248)

        # Remove padding
        out = out[:actual_n]                              # (actual_n, 149248)

        # ── CHANGED: reshape flat output to spatial ────────────────────────────
        out_spatial = out.numpy().reshape(actual_n, HR_H, HR_W)   # (actual_n, 352, 424)

        # ── Crop to subdomain ──────────────────────────────────────────────────
        out_sub = out_spatial[:, hr_r0:hr_r1, hr_c0:hr_c1]       # (actual_n, hr_H, hr_W)

        tas_preds.append(out_sub)
    print(f"X_mod shape:          {X_mod.shape}")            # must be (365, 88, 106, 21)

    return np.concatenate(tas_preds, axis=0)              # (n_time, hr_H, hr_W)


def predict_mean_tas_cnn_smalldomain(X_lr_full, mrsol_value,
                                      mrsol_idx, fixed_bs,
                                      generator,
                                      hr_r0, hr_r1, hr_c0, hr_c1,
                                      tas_mean = 0, tas_std = 1):        # ← NEW: required args
    HR_H, HR_W = 352, 424
    assert HR_H * HR_W == 149248

    X_mod = X_lr_full.copy()
    X_mod[..., mrsol_idx] = mrsol_value

    n_time    = X_mod.shape[0]
    tas_preds = []

    for start in range(0, n_time, fixed_bs):
        end      = min(start + fixed_bs, n_time)
        batch_lr = X_mod[start:end].astype(np.float32)
        actual_n = len(batch_lr)                               # true batch size

        if actual_n < fixed_bs:
            pad      = fixed_bs - actual_n
            batch_lr = np.concatenate([
                batch_lr,
                np.zeros((pad, 88, 106, 21), dtype=np.float32)
            ], axis=0)                                         # (fixed_bs, 88, 106, 21)

        out = generator(batch_lr, training=False).numpy()     # (fixed_bs, 149248)

        # ── CHANGED: trim padding BEFORE reshape ──────────────────────────────
        out = out[:actual_n]                                   # (actual_n, 149248) ← trim first

        # ── CHANGED: de-normalise output ──────────────────────────────────────
        out = out * tas_std + tas_mean                         # ← NEW: now in Kelvin

        # Reshape and crop to subdomain
        out_spatial = out.reshape(actual_n, HR_H, HR_W)       # (actual_n, 352, 424)
        out_sub     = out_spatial[:, hr_r0:hr_r1, hr_c0:hr_c1]# (actual_n, hr_H, hr_W)
        tas_preds.append(out_sub)

    return np.concatenate(tas_preds, axis=0)                   # (n_time, hr_H, hr_W)


# ── 7. Core ALE computation ────────────────────────────────────────────────────
def predict_mean_tas_fulldomain(X_lr, X_hr, mrsol_value, MRSOL_IDX, FIXED_BS, generator):
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


def cut_domain(ds_ypred, ds_pred):

    # ── NEW: define target region ──────────────────────────────────────────────────
    lat_min, lat_max = 44.0, 45.5                          # ← NEW
    lon_min, lon_max = 7.0,  12.0                          # ← NEW

    # ── NEW: high-res mask from predictant_ypred_1.nc (3km, 352×424) ──────────────
    lat_hr = ds_ypred["lat"].values                        # (352, 424) ← NEW
    lon_hr = ds_ypred["lon"].values                        # (352, 424) ← NEW

    mask_hr = (                                            # ← NEW
        (lat_hr >= lat_min) & (lat_hr <= lat_max) &        # ← NEW
        (lon_hr >= lon_min) & (lon_hr <= lon_max)          # ← NEW
    )                                                      # (352, 424) bool ← NEW
    print(f"High-res mask: {mask_hr.sum()} pixels selected out of {mask_hr.size}")  # ← NEW

    # ── NEW: low-res mask from predictor_1.nc (12km, 88×106) ──────────────────────
    lat_lr = ds_pred["lat"].values                         # (88, 106)  ← NEW
    lon_lr = ds_pred["lon"].values                         # (88, 106)  ← NEW

    mask_lr = (                                            # ← NEW
        (lat_lr >= lat_min) & (lat_lr <= lat_max) &        # ← NEW
        (lon_lr >= lon_min) & (lon_lr <= lon_max)          # ← NEW
    )                                                      # (88, 106) bool  ← NEW
    print(f"Low-res mask:  {mask_lr.sum()} pixels selected out of {mask_lr.size}")  # ← NEW

    # ── NEW: get bounding box row/col indices to crop rectangular arrays ───────────
    lr_rows = np.where(mask_lr.any(axis=1))[0]            # ← NEW
    lr_cols = np.where(mask_lr.any(axis=0))[0]            # ← NEW
    lr_r0, lr_r1 = lr_rows[0], lr_rows[-1] + 1            # ← NEW
    lr_c0, lr_c1 = lr_cols[0], lr_cols[-1] + 1            # ← NEW
    print(f"Low-res crop:  rows {lr_r0}:{lr_r1}, cols {lr_c0}:{lr_c1}"
          f"  → shape ({lr_r1-lr_r0}, {lr_c1-lr_c0})")    # ← NEW

    hr_rows = np.where(mask_hr.any(axis=1))[0]            # ← NEW
    hr_cols = np.where(mask_hr.any(axis=0))[0]            # ← NEW
    hr_r0, hr_r1 = hr_rows[0], hr_rows[-1] + 1            # ← NEW
    hr_c0, hr_c1 = hr_cols[0], hr_cols[-1] + 1            # ← NEW
    print(f"High-res crop: rows {hr_r0}:{hr_r1}, cols {hr_c0}:{hr_c1}"
          f"  → shape ({hr_r1-hr_r0}, {hr_c1-hr_c0})")    # ← NEW

    # ── ADD THIS BEFORE the quantiles line ────────────────────────────────────────
    print("=== Subdomain crop diagnostics ===")
    print(f"lat_lr range in full domain: {lat_lr.min():.2f} – {lat_lr.max():.2f}")
    print(f"lon_lr range in full domain: {lon_lr.min():.2f} – {lon_lr.max():.2f}")
    print(f"lat_hr range in full domain: {lat_hr.min():.2f} – {lat_hr.max():.2f}")
    print(f"lon_hr range in full domain: {lon_hr.min():.2f} – {lon_hr.max():.2f}")
    print()
    print(f"Your target region: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
    print()
    print(f"Low-res  mask pixels selected: {mask_lr.sum()}")
    print(f"High-res mask pixels selected: {mask_hr.sum()}")
    print()
    print(f"lr_rows found: {lr_rows}")
    print(f"lr_cols found: {lr_cols}")
    print(f"hr_rows found: {hr_rows}")
    print(f"hr_cols found: {hr_cols}")

    return lr_r0, lr_r1, lr_c0, lr_c1, hr_r0, hr_r1, hr_c0, hr_c1


def main():

    MLMODEL = "SRGAN" #'SRGAN' #CNN
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

    # ── NEW: define time period ────────────────────────────────────────────────────
    TIME_START = "2003-06-01"                               # ← NEW
    TIME_END   = "2003-08-31"                               # ← NEW

    basedir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmto_ERAI_2003_arrhenius'

    fig_outdir = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/ALE/'

    # ── 1. Load data ───────────────────────────────────────────────────────────────
    ds_pred  = xr.open_dataset(f"{basedir}/predictor_1.nc")
    if MLMODEL == 'CNN':
       ds_pred = ds_pred.drop_vars("tas")                     # ← removes 'tas' from ds_pred
       PREDICTOR_VARS.remove("tas")
    print(list(ds_pred.data_vars))
    print(f"Variable count: {len(ds_pred.data_vars)}")
    ds_ypred = xr.open_dataset(f"{basedir}/predictant_ypred_1.nc")
    ds_ytest = xr.open_dataset(f"{basedir}/predictant_ytest_1.nc")

    ds_pred_decoded  = xr.open_dataset(f"{basedir}/predictor_1.nc",  # ← CHANGED
                                       decode_times=True)  # ← CHANGED
    ds_ypred_decoded = xr.open_dataset(f"{basedir}/predictant_ypred_1.nc",  # ← CHANGED
                                        decode_times=True)         # ← CHANGED

    times_pred  = pd.DatetimeIndex(ds_pred_decoded["time"].values)                # ← CHANGED
    times_ypred = pd.DatetimeIndex(ds_ypred_decoded["time"].values)               # ← CHANGED
    print(f"Predictor  time axis: {times_pred[0].date()} – {times_pred[-1].date()}")  # ← NEW
    print(f"Predictant time axis: {times_ypred[0].date()} – {times_ypred[-1].date()}")  # ← NEW

    # ── NEW: decode times and find matching indices for predictor (12km) ───────────
    #times_pred = xr.coding.times.decode_cf_datetime(       # ← NEW
    #    ds_pred["time"].values,                            # ← NEW
    #    units=ds_pred["time"].attrs["units"],              # ← NEW
    #    calendar=ds_pred["time"].attrs["calendar"]         # ← NEW
    #)                                                      # ← NEW  array of datetime64
    #times_pred = pd.DatetimeIndex(times_pred)              # ← NEW

    mask_time_pred = (                                     # ← NEW
        (times_pred >= TIME_START) &                       # ← NEW
        (times_pred <= TIME_END)                           # ← NEW
    )                                                      # ← NEW  boolean array (1460,)
    idx_pred = np.where(mask_time_pred)[0]                 # ← NEW  integer indices
    print(f"Predictor time steps selected: {len(idx_pred)}"   # ← NEW
          f"  ({times_pred[idx_pred[0]].date()} – "           # ← NEW
          f"{times_pred[idx_pred[-1]].date()})")               # ← NEW

    # ── NEW: decode times and find matching indices for predictant (3km) ───────────
    #times_ypred = xr.coding.times.decode_cf_datetime(      # ← NEW
    #    ds_ypred["time"].values,                           # ← NEW
    #    units=ds_ypred["time"].attrs["units"],             # ← NEW
    #    calendar=ds_ypred["time"].attrs["calendar"]        # ← NEW
    #)                                                      # ← NEW
    #times_ypred = pd.DatetimeIndex(times_ypred)            # ← NEW

    mask_time_ypred = (                                    # ← NEW
        (times_ypred >= TIME_START) &                      # ← NEW
        (times_ypred <= TIME_END)                          # ← NEW
    )                                                      # ← NEW
    idx_ypred = np.where(mask_time_ypred)[0]               # ← NEW
    print(f"Predictant time steps selected: {len(idx_ypred)}" # ← NEW
          f"  ({times_ypred[idx_ypred[0]].date()} – "         # ← NEW
          f"{times_ypred[idx_ypred[-1]].date()})")             # ← NEW

    # ── NEW: guard — both files must have same number of selected time steps ────────
    assert len(idx_pred) == len(idx_ypred), (              # ← NEW
        f"Time step mismatch: predictor={len(idx_pred)}, "
        f"predictant={len(idx_ypred)}. Check time axes.")  # ← NEW

    # ── NEW: guard — period must be divisible by FIXED_BS (pad if not) ─────────────
    print(f"Note: FIXED_BS={FIXED_BS}, selected steps={len(idx_pred)}, "  # ← NEW
          f"remainder={len(idx_pred) % FIXED_BS}")                         # ← NEW


    lr_r0, lr_r1, lr_c0, lr_c1, hr_r0, hr_r1, hr_c0, hr_c1 = cut_domain(ds_ypred, ds_pred)


    # ── 2. Build full low-res array (time, y, x, channels) ────────────────────────
    print("Loading low-res predictor array...")
    X_lowres_full = np.stack(
        [ds_pred[v].values[idx_pred].astype(np.float32) for v in PREDICTOR_VARS],
        axis=-1
    )                                           # (1460, 88, 106, 22)
    print(f"Low-res-full array shape: {X_lowres_full.shape}")   # (1460, 88, 106, 22)

    if MLMODEL == 'CNN':
        X_lowres_full_clim = np.stack(
            [ds_pred[v].values.astype(np.float32) for v in PREDICTOR_VARS],
            axis=-1
        )                                           # (1460, 88, 106, 22)
        # ── CHANGED: compute normalisation from X_lowres_full ─────────────────────────
        print("Computing normalisation stats...")
        chan_means = np.array([np.nanmean(X_lowres_full_clim[..., c])
                           for c in range(X_lowres_full_clim.shape[-1])])   # (21,)
        chan_stds  = np.array([np.nanstd(X_lowres_full_clim[..., c])
                           for c in range(X_lowres_full_clim.shape[-1])])   # (21,)

        print("Per-channel normalisation stats:")
        for i, v in enumerate(PREDICTOR_VARS):
            print(f"  {v:10s}: mean={chan_means[i]:10.4f}, std={chan_stds[i]:8.4f}")


        # ── CHANGED: normalise X_lowres_full after NaN filling ────────────────────────
        print("Normalising X_lowres_full...")
        X_lowres_full_norm = (X_lowres_full - chan_means[np.newaxis, np.newaxis, np.newaxis, :]) \
                       / chan_stds [np.newaxis, np.newaxis, np.newaxis, :]  # ← NEW

        print(f"After normalisation — mrsol channel: "
          f"mean={X_lowres_full_norm[..., MRSOL_IDX].mean():.4f}, "
          f"std={X_lowres_full_norm[..., MRSOL_IDX].std():.4f}")
        # Should print mean≈0, std≈1




    # ── CHANGED: crop to subdomain ────────────────────────────────────────────────
    X_lowres = X_lowres_full[:, lr_r0:lr_r1, lr_c0:lr_c1, :]  # ← CHANGED (was full 88×106)
    print(f"Low-res array shape after crop: {X_lowres.shape}")  # e.g. (1460, 12, 42, 22)

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

    # ── 3. Build full high-res array (time, y, x, 1) ──────────────────────────────
    print("Loading high-res target array...")
    tas_hires_full = ds_ypred["tas"].values[idx_ypred].astype(np.float32)   # (1460, 352, 424)

    # ── CHANGED: crop to subdomain ────────────────────────────────────────────────
    tas_hires = tas_hires_full[:, hr_r0:hr_r1, hr_c0:hr_c1]   # ← CHANGED (was full 352×424)

    print(f"High-res array shape after crop: {tas_hires.shape}")  # e.g. (1460, 48, 168, 1)
    tas_hires = np.where(
        (tas_hires == FILL) | (np.abs(tas_hires) > 1e19),
        np.nanmean(tas_hires), tas_hires
    )
    X_hires = tas_hires[..., np.newaxis]                    # (1460, 352, 424, 1)
    print(f"High-res array shape: {X_hires.shape}")

    # ── 4. Load generator ──────────────────────────────────────────────────────────
    if MLMODEL == 'SRGAN':
        basedir_generator = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/EPOCH100_tas_wsmto_ERAI_2003_arrhenius'
        file_generator = 'model_1_generator.h5'
    elif MLMODEL == 'CNN':
        basedir_generator = '/nobackup/rossby27/users/sm_yicwa/DATA_shared/AIES_revision_aug2026/CNN_models_ERAI/'
        file_generator = 'cnn_mse_model_with_new_training_period_20032009.h5'

    generator = tf.keras.models.load_model(basedir_generator + '/' + file_generator, compile=False)
    print("Generator loaded.")

    # Run this to see both input shapes and names
    print("Number of inputs:", len(generator.inputs))
    for i, inp in enumerate(generator.inputs):
        print(f"  Input {i}: name='{inp.name}', shape={inp.shape}, dtype={inp.dtype}")

    # check if model accepts variable spatial size ─────────────────────────
    print("Model input 0 shape:", generator.inputs[0].shape)   # (50, 88, 106, 22)
    print("Cropped LR shape:   ", X_lowres.shape[1:])          # e.g. (12, 42, 22)

    # ── 5. Verify MRSOL_IDX ───────────────────────────────────────────────────────
    #mrsol_data = X_lowres[..., MRSOL_IDX]                   # (1460, 88, 106)
    #mrsol_min  = np.nanmin(mrsol_data)
    #mrsol_max  = np.nanmax(mrsol_data)
    #print(f"mrsol range: {mrsol_min:.3f} – {mrsol_max:.3f} kg/m²")

    # ── ADD THIS to diagnose empty mrsol_sub_valid ────────────────────────────────
    mrsol_data = X_lowres[..., MRSOL_IDX]              # (1460, 88, 106)
    #mrsol_sub  = mrsol_data[:, lr_r0:lr_r1, lr_c0:lr_c1]   # (1460, 15, 33)

    """
    print(f"X_lowres_full shape before mrsol extraction: {X_lowres.shape}")
    # Must print (1460, 88, 106, 22) — if it shows (1460, 15, 33, 22) that's the bug
    print("=== mrsol subdomain diagnostics ===")
    print(f"mrsol_sub shape:         {mrsol_sub.shape}")
    print(f"mrsol_sub total pixels:  {mrsol_sub.size}")
    print(f"NaN count:               {np.isnan(mrsol_sub).sum()}")
    print(f"Finite count:            {np.isfinite(mrsol_sub).sum()}")
    print(f"Raw min (with NaN):      {np.nanmin(mrsol_sub) if np.isfinite(mrsol_sub).any() else 'ALL NaN'}")
    print(f"Raw max (with NaN):      {np.nanmax(mrsol_sub) if np.isfinite(mrsol_sub).any() else 'ALL NaN'}")
    print()

    # Check raw values BEFORE fill masking
    mrsol_raw = ds_pred["mrsol"].values                     # (1460, 88, 106) original
    mrsol_raw_sub = mrsol_raw[:, lr_r0:lr_r1, lr_c0:lr_c1] # (1460, 15, 33)
    print("=== Raw mrsol (before any masking) ===")
    print(f"Raw unique values sample: {np.unique(mrsol_raw_sub.ravel()[:20])}")
    print(f"Raw min: {np.nanmin(mrsol_raw_sub):.4f}")
    print(f"Raw max: {np.nanmax(mrsol_raw_sub):.4f}")
    print(f"Count of -9999.9:        {(mrsol_raw_sub == -9999.9).sum()}")
    print(f"Count of values > 1e19:  {(np.abs(mrsol_raw_sub) > 1e19).sum()}")
    print(f"Count of NaN:            {np.isnan(mrsol_raw_sub).sum()}")
    print(f"Count of valid:          {np.isfinite(mrsol_raw_sub).sum() - (mrsol_raw_sub == -9999.9).sum()}")
    print()

    # Check the fill value used in extract_flat
    print("=== Checking extract_flat fill masking ===")
    print(f"FILL value used: {FILL}")
    sample_vals = mrsol_raw_sub[0, :3, :3]
    print(f"Sample 3x3 raw values at t=0:\n{sample_vals}")


    # ── CHANGED: mrsol stats from subdomain only ──────────────────────────────────
    mrsol_data = X_lowres[..., MRSOL_IDX]              # full domain for model input
    mrsol_sub  = mrsol_data[:, lr_r0:lr_r1, lr_c0:lr_c1]   # ← NEW subdomain for ALE bins
    mrsol_sub_valid = mrsol_sub[np.isfinite(mrsol_sub)]
    #mrsol_min  = np.nanmin(mrsol_sub)
    #mrsol_max  = np.nanmax(mrsol_sub)
    #print(f"mrsol range: {mrsol_min:.3f} – {mrsol_max:.3f} kg/m²")
    print(f"mrsol subdomain range: {mrsol_sub_valid.min():.3f} – {mrsol_sub_valid.max():.3f} kg/m²")
    """

    # ── 6. Define ALE bin edges from actual mrsol distribution ────────────────────
    N_BINS   = 20                                           # adjust as needed
    quantiles = np.percentile(mrsol_data[np.isfinite(mrsol_data)],
                              np.linspace(0, 100, N_BINS + 1))
    #quantiles  = np.percentile(
    #    mrsol_sub[np.isfinite(mrsol_sub)],                  # ← CHANGED: use subdomain
    #    np.linspace(0, 100, N_BINS + 1)
    #)
    quantiles = np.unique(quantiles)
    N_BINS    = len(quantiles) - 1
    print(f"ALE bins: {N_BINS}  |  edges: {quantiles[0]:.3f} ... {quantiles[-1]:.3f}")

    # ── ADD before loop: estimate normalisation stats ──────────────────────────────
    tas_raw  = ds_ytest["tas"].values.astype(np.float32)
    tas_raw  = np.where((tas_raw == FILL) | (np.abs(tas_raw) > 1e19), np.nan, tas_raw)
    TAS_MEAN = float(np.nanmean(tas_raw))
    TAS_STD  = float(np.nanstd(tas_raw))
    print(f"TAS_MEAN = {TAS_MEAN:.4f} K,  TAS_STD = {TAS_STD:.4f} K")

    # ── 8. Compute ALE effects per bin ────────────────────────────────────────────
    print("\nComputing ALE effects across bins...")
    ale_effects = np.zeros(N_BINS)
    bin_centres = 0.5 * (quantiles[:-1] + quantiles[1:])

    for i in range(N_BINS):
        z_lo, z_hi = quantiles[i], quantiles[i + 1]
        if MLMODEL == "SRGAN":
            pred_hi = predict_mean_tas_smalldomain(X_lowres_full, tas_hires_full, z_hi, MRSOL_IDX, FIXED_BS, generator, \
                hr_r0, hr_r1, hr_c0, hr_c1, tas_mean = TAS_MEAN, tas_std = TAS_STD).mean()
            pred_lo = predict_mean_tas_smalldomain(X_lowres_full, tas_hires_full, z_lo, MRSOL_IDX, FIXED_BS, generator, \
                hr_r0, hr_r1, hr_c0, hr_c1, tas_mean = TAS_MEAN, tas_std = TAS_STD).mean()
        elif MLMODEL == "CNN":
            pred_hi = predict_mean_tas_cnn_smalldomain(X_lowres_full_norm, z_hi, MRSOL_IDX, FIXED_BS, generator, hr_r0, hr_r1, hr_c0, hr_c1, \
                tas_mean=TAS_MEAN, tas_std=TAS_STD \
                ).mean()
            pred_lo = predict_mean_tas_cnn_smalldomain(X_lowres_full_norm, z_lo, MRSOL_IDX, FIXED_BS, generator, hr_r0, hr_r1, hr_c0, hr_c1, \
                tas_mean=TAS_MEAN, tas_std=TAS_STD \
                ).mean()

        ale_effects[i] = pred_hi - pred_lo
        print(f"  Bin {i+1:2d}/{N_BINS}: mrsol [{z_lo:.3f}, {z_hi:.3f}]"
              f"  pred_hi={pred_hi:.4f}  pred_lo={pred_lo:.4f}"  # ← NEW: show raw values
              f"  Δtas = {ale_effects[i]:.4f} K")

    # Accumulate and centre
    ale_accumulated = np.cumsum(ale_effects)
    ale_accumulated -= np.mean(ale_accumulated)             # centre around 0

    title_number = {'CNN':'(a)', 'SRGAN':'(b)'}

    # ── 9. Plot ────────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(bin_centres, ale_accumulated, color="steelblue", lw=2.5,
             marker="o", ms=4, label="ALE")
    ax1.fill_between(bin_centres, ale_accumulated, alpha=0.15, color="steelblue")
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xlabel("Surface Soil Moisture (kg m⁻²)", fontsize=12)
    ax1.set_ylabel("ALE of 2-m air temperature (K)", fontsize=12)
    #ax1.set_title("ALE: Effect of mrsol on SRGAN-predicted Near-Surface Air Temperature",
    #              fontsize=12)
    ax1.set_title(                                         # ← CHANGED: add period to title
        #f"ALE: mrsol → tas  ({TIME_START} to {TIME_END})", # ← CHANGED
        f"{title_number[MLMODEL]} ALE for ${MLMODEL} ({TIME_START} to {TIME_END})", # ← CHANGED
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
    #plt.savefig(f"{fig_outdir}/ale_mrsol_srgan_fullgrid.png", dpi=150)
    plt.savefig(f"{fig_outdir}/ale_mrsol_{MLMODEL}_{TIME_START}_{TIME_END}.png", dpi=150)  # ← CHANGED
    plt.show()
    #print("Saved: ale_mrsol_srgan_fullgrid.png")
    print(f"Saved: ale_mrsol_{MLMODEL}_{TIME_START}_{TIME_END}.png")          # ← CHANGED

if __name__ == "__main__":
    main()
