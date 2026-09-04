
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

    print([type(x) for x in statistics])
    print([np.shape(x) for x in statistics])
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
    if not any('correlation' in s for s in titles) and vmin > 1 and vmax > 1:
        vmin = math.floor(vmin)
        vmax = math.ceil(vmax)
    #if any('Abs Value' in s for s in titles):
    #    vmin = 0.0

    print('1 vmin, vmax', vmin, vmax)
    levels = np.linspace(vmin, vmax, nlevels_def)

    fig, axes = plt.subplots(nrows=nrows_def, ncols=ncols_def, 
             figsize=(figsize_def[0], figsize_def[1]),
             subplot_kw={'projection': ccrs.PlateCarree()},
             constrained_layout=True)
    axes = axes.flatten()

    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()


    n_subplot = len(titles)
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

        letter = chr(97 + plot_counter) 
        if '99th Percentile' in title:
            letter = chr(97 + plot_counter + n_subplot) #4) 
        elif 'Difference' in title:
            letter = chr(97 + plot_counter + n_subplot) #4) 
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
        if not any('correlation' in s for s in titles):
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


# Plot Power Spectral Density (PSD) comparison 
def plot_psd_comparison(psd_list, exp_names, output_path, variables):
    """
    #psd_dict : dict of {label: (wavenumber, wavelength, psd_mean)}
    #           e.g. {'AROME (3km)': (k1, wl1, psd1),
    #                 'SRGAN (3km)': (k2, wl2, psd2),
    #                 'CNN (3km)':   (k3, wl3, psd3),
    #                 'ALADIN (12km)': (k4, wl4, psd4)}
    psd_list  : list of [wavenumber, wavelength, psd_mean] arrays, one per experiment
                e.g. [psd_arome, psd_aladin, psd_cnn, psd_srgan]
                where each psd_xxx = [wavenumber, wavelength, psd_mean]
    exp_names : list of experiment names, same order as psd_list
                e.g. ['AROME (3km)', 'ALADIN (12km)', 'CNN (3km)', 'SRGAN (3km)']
    output_path : path to save the PNG file
    """
    colors = {
        'HCLIM 3km': 'black',
        'HCLIM 12km': 'tab:red',
        'CNN': 'tab:orange',
        'SRGAN': 'tab:blue',
    }
    linestyles = {
        'HCLIM 3km': '-',
        'HCLIM 12km': ':',
        'CNN': '--',
        'SRGAN': '-.',
    }
    title_dict = { 'tas': '(a) Power spectral density for 2-m air temperature', \
        'pr': '(b) Power spectral density for precipitation' }
    fontsize_def = 18
    variable = ''.join(variables)

    fig, ax = plt.subplots(figsize=(7, 6))

    #for label, (wavenumber, wavelength, psd_mean) in psd_dict.items():
    for label, (wavenumber, wavelength, psd_mean) in zip(exp_names, psd_list):
        # pick color/linestyle by matching a keyword in the label, default otherwise
        color = next((c for key, c in colors.items() if key in label), None)
        ls = next((s for key, s in linestyles.items() if key in label), '-')

        # mask invalid values so log-log plot doesn't break
        valid = np.isfinite(wavelength) & np.isfinite(psd_mean) & (psd_mean > 0)

        ax.loglog(
            wavelength[valid],
            psd_mean[valid],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=2,
        )

    ax.set_xlabel('Wavelength (km)', fontsize = int(fontsize_def-4) )
    ax.set_ylabel('Power Spectral Density', fontsize = int(fontsize_def-4) )
    ax.set_title(f'{title_dict[variable]}', fontsize = int(fontsize_def-2) )

    # invert x-axis so large scales (long wavelength) are on the left, small scales on the right
    ax.invert_xaxis()

    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize= int(fontsize_def - 4), loc='best')
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved PSD comparison plot to {output_path}')



# Define function to plot normalized PSD ratio (model / reference) vs wavelength
def plot_psd_ratio(psd_list, exp_names, ref_name, output_path='psd_ratio.png'):
    """
    psd_list  : list of [wavenumber, wavelength, psd_mean], one per experiment (including reference)
    exp_names : list of experiment names, same order/length as psd_list
    ref_name  : name of the reference experiment (must match an entry in exp_names)
    output_path : path to save the PNG file
    """
    #colors = {'SRGAN': 'tab:blue', 'CNN': 'tab:orange', 'ALADIN': 'tab:red', '12km': 'tab:red'}
    #linestyles = {'SRGAN': '-.', 'CNN': '--', 'ALADIN': ':', '12km': ':'}
    colors = {
        'HCLIM 3km': 'black',
        'HCLIM 12km': 'tab:red',
        'CNN': 'tab:orange',
        'SRGAN': 'tab:blue',
    }
    linestyles = {
        'HCLIM 3km': '-',
        'HCLIM 12km': ':',
        'CNN': '--',
        'SRGAN': '-.',
    }

    # find reference PSD
    ref_idx = exp_names.index(ref_name)
    ref_wavelength = psd_list[ref_idx][1]
    ref_psd = psd_list[ref_idx][2]

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, (wavenumber, wavelength, psd_mean) in zip(exp_names, psd_list):
        if label == ref_name:
            continue  # skip plotting reference against itself

        # assumes wavelength bins are aligned/comparable across datasets;
        # if not (e.g. different grid spacing), interpolate onto ref_wavelength first
        #if not np.allclose(wavelength, ref_wavelength, equal_nan=True):
        #    ratio_psd = np.interp(ref_wavelength, wavelength[::-1], psd_mean[::-1])
        #else:
        #    ratio_psd = psd_mean

        same_shape = (wavelength.shape == ref_wavelength.shape)
        if same_shape and np.allclose(wavelength, ref_wavelength, equal_nan=True):
            ratio_psd = psd_mean
        else:
            ratio_psd = np.interp(ref_wavelength, wavelength[::-1], psd_mean[::-1])


        ratio = ratio_psd / ref_psd

        color = next((c for key, c in colors.items() if key in label), None)
        ls = next((s for key, s in linestyles.items() if key in label), '-')

        valid = np.isfinite(ref_wavelength) & np.isfinite(ratio)
        ax.semilogx(ref_wavelength[valid], ratio[valid], label=label, color=color, linestyle=ls, linewidth=2)

    ax.axhline(1.0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('Wavelength (km)', fontsize=12)
    ax.set_ylabel(f'PSD ratio (model / {ref_name})', fontsize=12)
    ax.set_title('Normalized Power Spectral Density Ratio', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved PSD ratio plot to {output_path}')
