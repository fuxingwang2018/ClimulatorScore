"""
plot_heatwave_winds_tas_aug15.py

Converted from plot_heatwave_winds_tas_aug15.ipynb

Plots, for the heatwave event of 2003-08-15 12:00 UTC:
  1. Z500 geopotential height contours + soil moisture shading
  2. Z500 contours + soil moisture shading + 950 hPa wind vectors
  3. 3 km "truth" surface temperature (TAS) map for Emilia-Romagna
  4. Simple CNN emulator TAS prediction map
  5. SRGAN emulator TAS prediction map

Each figure is saved to disk as a PNG. Interactive-inspection cells from the
original notebook (bare variable reprs, `print(tas_c)`, `.head`, and the
trailing standalone `plt.show()`) have been removed since they have no
effect in a script.

Plots 1 and 2 (Z500/soil-moisture, with and without wind vectors) are now a
single `plot_z500_soil_moisture()` function toggled by `include_wind`, since
the two notebook cells were near-duplicates of the same plot.
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from datetime import datetime


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_TIME = "2003-08-15T12:00:00"
TARGET_TIME_PNG = "20030815T12"

# All PNG output goes here (created automatically if it doesn't exist)
OUTPUT_DIR = "/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/synoptic_conditions/"

COMBINED_12KM_FILE = (
    "/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/"
    "01-rampal2021-unet/Emulator_HCLIM_CRM_T_withSM_whus/training_data_fuxing/"
    "combined_12km_6hr_20000101-20091231.nc"
)

TAS_MRSOL_3KM_FILE = (
    "/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/"
    "01-rampal2021-unet/Emulator_HCLIM_CRM_T_withSM_whus/training_data_fuxing/"
    "tas_mrsol_3km_6hr_200001010000-200912311800.nc"
)

CNN_PREDICTION_FILE = (
    "/nobackup/rossby27/users/sm_yicwa/PROJECTS/01-PROJ_emulator/"
    "01-rampal2021-unet/Emulator_HCLIM_CRM_T_withSM_whus/03-inference_comp/"
    "simple_cnn_prediction_normalized_20030815T1200.nc"
)

SRGAN_PREDICTION_FILE = (
    #"/nobackup/rossby27/users/sm_fuxwa/data_share/GAN_out/predictant_ypred_1_PDA.nc"
    "/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/SG/SRGAN_OUT/ARRHENIUS/EPOCH100_tas_wsmt_ERAI_2003_2009_arrhenius/predictant_ypred_1.nc"
)


# ---------------------------------------------------------------------------
# 1/2. Z500 height + soil moisture map, optionally with 950 hPa wind vectors
# ---------------------------------------------------------------------------
def plot_z500_soil_moisture(target_time, include_wind=False):
    """Plot Z500 (500 hPa) geopotential height contours over soil-moisture
    shading, and save the figure to disk.

    If include_wind is False (default), Z950 height contours (in blue) are
    drawn as well. If include_wind is True, 950 hPa wind vectors are drawn
    instead of the Z950 contours.
    """
    ds = xr.open_dataset(COMBINED_12KM_FILE)
    data_slice = ds.sel(time=target_time)

    sm = data_slice["mrsol"]
    z500 = data_slice["phi500"] / 98.0665  # geopotential -> geopotential height (dam)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Soil moisture shading
    sm_alpha = 0.8 if include_wind else 1.0
    sm_plot = ax.pcolormesh(sm.lon, sm.lat, sm, cmap="YlGn", shading="auto", alpha=sm_alpha)

    # Z500 contours
    levels = np.arange(540, 604, 2)
    contours = ax.contour(z500.lon, z500.lat, z500, levels=levels, colors="black", linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=12, fmt="%1.0f")

    if include_wind:
        # 950 hPa wind vectors (quiver) instead of Z950 contours
        u950 = data_slice["ua950"]
        v950 = data_slice["va950"]

        skip = 5  # grid-point subsampling; increase if the plot is too crowded
        q = ax.quiver(
            u950.lon[::skip, ::skip],
            u950.lat[::skip, ::skip],
            u950[::skip, ::skip],
            v950[::skip, ::skip],
            color="darkred",
            scale=150,      # larger scale = smaller arrows
            width=0.0015,   # arrow shaft thickness
            headwidth=10,   # arrow head size
        )
        ax.quiverkey(q, X=0.9, Y=1.05, U=10, label="10 m/s", labelpos="E", coordinates="axes")

        #title = f"Z500 (contour), mrsol (shading) & 950hPa Wind\n{target_time}"
        title = f"(a) HCLIM 12 km synoptic conditions"
        title_pad = 30
    else:
        # Z950 contours
        z950 = data_slice["phi950"] / 98.0665
        levels_z950 = np.arange(50, 60, 2)
        contours_z950 = ax.contour(
            z950.lon, z950.lat, z950, levels=levels_z950, colors="blue", linewidths=1.5
        )
        ax.clabel(contours_z950, inline=True, fontsize=12, fmt="%1.0f")

        #title = f"Z500 Height (dam) & Soil Moisture (mrsol)\nValid: {target_time}"
        title = f"(a) HCLIM 12 km synoptic conditions"
        title_pad = 20

    #ax.set_xlabel("Longitude [\u00b0E]", fontsize=18, fontweight="bold", labelpad=15)
    #ax.set_ylabel("Latitude [\u00b0N]", fontsize=18, fontweight="bold", labelpad=15)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=title_pad)

    cbar = plt.colorbar(sm_plot, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
    #cbar.set_label("Soil Moisture (mrsol) [kg/m\u00b2]", fontsize=16, labelpad=10)
    cbar.set_label("Soil moisture [m\u00b3/m\u00b3]", fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    suffix = "wind" if include_wind else "z950"
    target_time_plot = datetime.fromisoformat(target_time).strftime("%Y%m%dT%H")
    output_name = os.path.join(OUTPUT_DIR, f"synoptic_contidion_{target_time_plot}_{suffix}.png")
    plt.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_name}")


# ---------------------------------------------------------------------------
# Shared TAS plotting helper (truth / CNN prediction / SRGAN prediction)
# ---------------------------------------------------------------------------
def _plot_tas_map(data_slice, tas_c, target_time, title, output_name):
    """Plot a shaded + contoured surface-temperature (TAS, in \u00b0C) map."""
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.pcolormesh(
        data_slice.lon, data_slice.lat, tas_c,
        cmap="YlOrRd", shading="auto", vmin=15, vmax=40,
    )

    temp_levels = np.arange(35, 40, 2.5)
    # Smooth the data slightly, only for the contour lines, to make them look cleaner
    tas_smoothed = gaussian_filter(tas_c.values, sigma=1.0)
    contours = ax.contour(
        data_slice.lon, data_slice.lat, tas_smoothed,
        levels=temp_levels, colors="black", linewidths=0.8, alpha=0.7,
    )
    ax.clabel(contours, inline=True, fontsize=10, fmt="%1.0f\u00b0C")

    #ax.set_xlabel("Longitude [\u00b0E]", fontsize=18, fontweight="bold", labelpad=10)
    #ax.set_ylabel("Latitude [\u00b0N]", fontsize=18, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title(f"{title}", fontsize=20, fontweight="bold", pad=20)
    #ax.set_title(f"{title}\n {target_time}", fontsize=20, fontweight="bold", pad=20)

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
    cbar.set_label("Temperature [\u00b0C]", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_name}")


# ---------------------------------------------------------------------------
# 3. Truth (3 km HCLIM) TAS map
# ---------------------------------------------------------------------------
def plot_truth_tas(target_time):
    ds = xr.open_dataset(TAS_MRSOL_3KM_FILE)
    data_slice = ds.sel(time=target_time)
    tas_c = data_slice["tas"] - 273.15  # Kelvin -> Celsius
    target_time_plot = datetime.fromisoformat(target_time).strftime("%Y%m%dT%H")

    _plot_tas_map(
        data_slice, tas_c, target_time,
        title="(b) HCLIM 3km air temperature",
        output_name=os.path.join(OUTPUT_DIR, f"HCLIM3_tas_map_{target_time_plot}.png"),
    )


# ---------------------------------------------------------------------------
# 4. Simple CNN emulator TAS prediction map
# ---------------------------------------------------------------------------
def plot_cnn_prediction_tas(target_time):
    ds = xr.open_dataset(CNN_PREDICTION_FILE)
    data_slice = ds.sel(time=target_time)
    pred_data = data_slice["test"]
    tas_c = pred_data - 273.15
    target_time_plot = datetime.fromisoformat(target_time).strftime("%Y%m%dT%H")

    _plot_tas_map(
        data_slice, tas_c, target_time,
        title="(c) CNN air temperature",
        output_name=os.path.join(OUTPUT_DIR, f"cnn_fixed_prediction_map_{target_time_plot}.png"),
    )


# ---------------------------------------------------------------------------
# 5. SRGAN emulator TAS prediction map
# ---------------------------------------------------------------------------
def plot_srgan_prediction_tas(target_time):
    ds = xr.open_dataset(SRGAN_PREDICTION_FILE)
    data_slice = ds.sel(time=target_time)
    predata = data_slice["tas"]
    tas_c = predata - 273.15
    target_time_plot = datetime.fromisoformat(target_time).strftime("%Y%m%dT%H")

    _plot_tas_map(
        data_slice, tas_c, target_time,
        title="(d) SRGAN air temperature",
        output_name=os.path.join(OUTPUT_DIR, f"srgan_fixed_prediction_map_{target_time_plot}.png"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #plot_z500_soil_moisture(TARGET_TIME, include_wind=False)
    plot_z500_soil_moisture(TARGET_TIME, include_wind=True)
    plot_truth_tas(TARGET_TIME)
    plot_cnn_prediction_tas(TARGET_TIME)
    plot_srgan_prediction_tas(TARGET_TIME)


if __name__ == "__main__":
    main()
