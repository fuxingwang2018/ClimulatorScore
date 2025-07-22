
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

def plot_and_save_maps_latlon(statistics, lat2d,lon2d, titles, output_file, vmin=None, vmax=None, cmap='coolwarm'):
    # Get global vmin/vmax across all stat arrays
    if vmin is None:
        vmin = min([np.nanmin(stat) for stat in statistics])
    if vmax is None:
        vmax = max([np.nanmax(stat) for stat in statistics])

    levels = np.linspace(vmin, vmax, 7)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20),
             subplot_kw={'projection': ccrs.PlateCarree()},
             constrained_layout=True)
    axes = axes.flatten()

    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    for i, (stat, title) in enumerate(zip(statistics, titles)):
        #im = axes[i].imshow(stat, cmap=cmap, vmin=vmin, vmax=vmax)
        stat_flat= stat.flatten()  # Flattened to match the irregular structure
        contour = axes[i].tricontourf(lon_flat, lat_flat, stat_flat,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, levels=levels,
                    vmin=vmin, vmax=vmax)


        axes[i].set_title(title, fontsize=20)
        # Add coastlines and other features
        axes[i].coastlines(resolution='10m',linewidth=1.2, color='black')
        #axes[i].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        axes[i].gridlines(draw_labels=False)
        axes[i].add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
        #axes[i].add_feature(cfeature.RIVERS)

        # Not for CARTOPY
        #divider = make_axes_locatable(axes[i])
        #cax = divider.append_axes("bottom", size="5%", pad=0.4)
        #cbar = plt.colorbar(contour, cax=cax)
        #fig.colorbar(contour, ax=axes[i], orientation="horizontal", shrink=0.7, pad=0.1)

        stat_domain_ave = np.mean(stat)
        # Add the statistics value to the lower right
        #text_x = stat.shape[1] - 2  # Right-most position
        #text_y = stat.shape[0] - 1  # Bottom position (because origin='lower')
        text_x = lon2d[-1, -2]  # near bottom-right
        text_y = lat2d[-1, -1]
        #text_y = 0  # Bottom position
        axes[i].text(text_x, text_y, f"{stat_domain_ave:.2f}",
            color='white', fontsize=24, ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    cbar = fig.colorbar(contour, ax=axes, orientation="horizontal", shrink=0.7, pad=0.02)
    #cbar.set_label("Metric Name", fontsize=14)
    cbar.ax.tick_params(labelsize=20)

    # Hide unused subplots if there are any
    for i in range(len(statistics), len(axes)):
        axes[i].axis('off')

    #plt.tight_layout()
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file, dpi=300)
    plt.close()

