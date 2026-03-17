
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math

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

        stat_masked = stat[(stat >= -1e10) & (stat <= 1e10)]
        stat_domain_ave = np.nanmean(stat_masked)
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


def plot_and_save_maps_latlon(statistics, lat2d, lon2d, titles, output_file, \
    vmin=None, vmax=None, cmap='coolwarm', fig_parameters=None):

    if np.isfinite(statistics).any():
        statistics = np.nan_to_num(statistics, nan=0.0, posinf=0.0, neginf=0.0)
    nrows_def = fig_parameters['nrows_def']
    ncols_def = fig_parameters['ncols_def']
    figsize_def  = fig_parameters['figsize_def']
    fontsize_def = fig_parameters['fontsize_def']
    nlevels_def = fig_parameters['nlevels_def']
    extend_def = fig_parameters['extend_def']
    # Get global vmin/vmax across all stat arrays
    print('0 vmin, vmax, titles', vmin, vmax, titles)
    if vmin is None:
        vmin = min([np.nanmin(stat) for stat in statistics])
    if vmax is None:
        vmax = max([np.nanmax(stat) for stat in statistics])
    if any('correlation' in s for s in titles):
        vmin = math.floor(vmin)
        vmax = math.ceil(vmax)

    print('1 vmin, vmax', vmin, vmax)
    levels = np.linspace(vmin, vmax, nlevels_def)

    fig, axes = plt.subplots(nrows=nrows_def, ncols=ncols_def, 
             figsize=(figsize_def[0], figsize_def[1]),
             subplot_kw={'projection': ccrs.PlateCarree()},
             constrained_layout=True)
    axes = axes.flatten()

    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()


    plot_counter = 0 
    for i, (stat, title) in enumerate(zip(statistics, titles)):
        #im = axes[i].imshow(stat, cmap=cmap, vmin=vmin, vmax=vmax)
        stat_flat= stat.flatten()  # Flattened to match the irregular structure
        print("lat_flat:", len(lat_flat))
        print("lon_flat:", len(lon_flat))
        print("stat_flat:", len(stat_flat))
        print("stat shape:", stat.shape)
        print("lat2d shape:", lat2d.shape)
        print("lon2d shape:", lon2d.shape)

        contour = axes[i].tricontourf(lon_flat, lat_flat, stat_flat,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, levels=levels,
                    vmin=vmin, vmax=vmax,
                    extend=extend_def)


        letter = chr(97 + plot_counter + 4 + 3) 
        new_title = f"({letter}) {title}"
        axes[i].set_title(new_title, fontsize=fontsize_def)
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

        #stat_domain_ave = np.mean(stat)
        stat = np.where((stat > 1e10) | (stat < -1e10), np.nan, stat)
        stat_domain_ave = np.nanmean(stat)
        # Add the statistics value to the lower right
        #text_x = stat.shape[1] - 2  # Right-most position
        #text_y = stat.shape[0] - 1  # Bottom position (because origin='lower')
        text_x = lon2d[10, -2]  # near bottom-right
        text_y = lat2d[5, -2]
        #text_x = lon2d[-1, -2]  # near bottom-right
        #text_y = lat2d[-1, -1]
        #text_y = 0  # Bottom position
        # no texts for correlation coefficient
        if any('correlation' in s for s in titles):
            axes[i].text(text_x, text_y, f"{stat_domain_ave:.2f}",
                color='white', fontsize=fontsize_def, ha='right', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        plot_counter += 1

    #cbar = fig.colorbar(contour, ax=axes, orientation="horizontal", shrink=0.7, aspect=40, pad=0.02)
    cbar = fig.colorbar(
        contour, 
        ax=axes, 
        orientation="horizontal", 
        shrink=0.7,   # Controls the length (left-to-right)
        aspect=40,    # Controls the thickness (higher number = thinner bar)
        pad=0.02      # Distance from the plot
        )
    #cbar.set_label("Metric Name", fontsize=14)
    cbar.ax.tick_params(labelsize=fontsize_def)

    # Remove all spacing between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    # Make axes fill the figure more tightly
    #plt.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Hide unused subplots if there are any
    for i in range(len(statistics), len(axes)):
        axes[i].axis('off')

    #plt.tight_layout()
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_and_save_boxplot(statistics, titles, GCM, output_file, \
    fig_parameters=None):

    if GCM == 'ECMWF-ERAINT':
        title_def = '(a) ERAI-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_HIST":
        title_def = '(b) ECE-HI2HI'
    elif GCM == "ICHEC-EC-EARTH_RCP85_MC":
        title_def = '(c) ECE-MC2MC'
    elif GCM == "ICHEC-EC-EARTH_RCP85_LC":
        FIRST_YEAR_12km, LAST_YEAR_12km, FIRST_YEAR_3km, LAST_YEAR_3km = 2090, 2099, 2089, 2099
        title_def = '(d) ECE-LC2LC'

    all_stat_flat = []
    experiment_names = []
    for i, (stat, title) in enumerate(zip(statistics, titles)):
        stat = stat[np.isfinite(stat)]
        stat_flat = stat.flatten()  # Flattened to match the irregular structure
        #stat_flat = stat_flat[~np.isnan(stat_flat)]
        all_stat_flat.append(stat_flat)
        experiment_names.append(title)

    plt.figure(figsize=(10, 6))
    # patch_artist=True allows us to fill the boxes with color
    bp = plt.boxplot(all_stat_flat, labels=experiment_names, patch_artist=True,
                     showmeans=True, meanline=True, 
                     medianprops={'color': 'black', 'linewidth': 2},
                     flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})

    # Customize box colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Final Polish
    plt.ylim(-1, 1)  # As requested, y-axis ranging from 0 to 1
    plt.ylabel("Correlation ($r$)", fontsize=12)
    #plt.xlabel("Experiment", fontsize=12)
    plt.title(title_def, fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_file}", dpi=300, bbox_inches='tight')
    plt.close()

