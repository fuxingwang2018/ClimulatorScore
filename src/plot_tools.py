
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Define function to plot and save maps
def plot_and_save_maps(statistics, titles, output_file, vmin=None, vmax=None, cmap='coolwarm'):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    axes = axes.flatten()

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        im = axes[i].imshow(stat, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(title, fontsize=10)
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # Dynamically adjust colorbar size to match the axis height
        #cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        #cbar_height = axes[i].get_position().height  # Get the height of the axis
        #cbar.ax.set_aspect(cbar_height / cbar.ax.get_position().height)

        stat_domain_ave = np.mean(stat)
        # Add the statistics value to the lower right
        text_x = stat.shape[1] - 2  # Right-most position
        text_y = stat.shape[0] - 1  # Bottom position (because origin='lower')
        #text_y = 0  # Bottom position
        axes[i].text(text_x, text_y, f"{stat_domain_ave:.2f}",
            color='white', fontsize=12, ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    # Hide unused subplots if there are any
    for i in range(len(statistics), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
