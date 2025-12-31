import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Corrected sample data
# -----------------------------
samples = {
    'ECEHist': {
        'CNN':   {'MBE': 1.2,  'RMSE': 1.56, 'CORR': 0.97},
        'SRGAN': {'MBE': -0.21,'RMSE': 2.22, 'CORR': 0.97}
    },
    'ECEFutMC': {
        'CNN':   {'MBE': 1.2,  'RMSE': 1.56, 'CORR': 0.97},
        'SRGAN': {'MBE': -0.3, 'RMSE': 2.45, 'CORR': 0.97}
    },
    'ECEHistFutMC': {
        'CNN':   {'MBE': 1.22, 'RMSE': 1.57, 'CORR': 0.97},
        'SRGAN': {'MBE': -0.04,'RMSE': 2.40, 'CORR': 0.97}
    }
}

# Metrics to include
metrics = ["MBE", "RMSE", "CORR"]

# -----------------------------
# Normalize metrics so “higher = better”
# Negative MBE is better than positive MBE, so invert sign
# RMSE: lower is better → invert
# CORR: higher is better → keep
# -----------------------------

def normalize(values, invert=False):
    values = np.array(values)
    if invert:
        values = -values  # invert if error metric
    min_v = values.min()
    max_v = values.max()
    if max_v - min_v == 0:
        return np.ones_like(values)  # avoid division by zero
    return (values - min_v) / (max_v - min_v)

# -----------------------------
# Create radar plot for each dataset
# -----------------------------

for dataset, methods in samples.items():

    labels = metrics
    num_vars = len(labels)

    # Compute angles for radar axes
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Prepare figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    method_colors = {"CNN": "tab:blue", "SRGAN": "tab:red"}

    # Collect raw metric values for normalization
    mbe_vals = [methods[m]["MBE"] for m in methods]
    rmse_vals = [methods[m]["RMSE"] for m in methods]
    corr_vals = [methods[m]["CORR"] for m in methods]

    mbe_norm = normalize(mbe_vals, invert=True)   # lower abs(MBE) is better
    rmse_norm = normalize(rmse_vals, invert=True) # lower RMSE is better
    corr_norm = normalize(corr_vals)              # higher CORR is better

    for i, (method, metrics_dict) in enumerate(methods.items()):
        stats = [
            mbe_norm[i],
            rmse_norm[i],
            corr_norm[i]
        ]
        stats += stats[:1]

        ax.plot(angles, stats, linewidth=2, label=method, color=method_colors[method])
        ax.fill(angles, stats, alpha=0.15, color=method_colors[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_title(f"Radar Plot – {dataset}", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(f"radar_plot_{dataset}.png", dpi=300)
    plt.close()

print("Radar plots saved as radar_plot_<dataset>.png")

