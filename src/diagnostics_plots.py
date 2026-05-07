# =============================================================================
#                         diagnostics_plots
# =============================================================================
"""
Plotting routines for finite-time transition path diagnostics

The functions in this module visualize the geometry, source-to-target
transport diagnostics, whole-system transport distances, effective-current
fielsd, arrival-time maps, origin-likelihood maps, and source-density maps
computed by "diagnositcs_core.py"
"""

from config import (
    T, K_win, all_years, apolemia_years, targets,
    lon, lat, spatial_dis, week_step,
    latmin_g1, latmax_g1, lonpos_g1,
    latmin_g2, latmax_g2, lonpos_g2, late_week_range, clim_years)

from diagnostics_core import (
    leave_one_out_mean,
    gate_from_domain,
    add_source_target_diagnostics,
    climatology_flux,
    mean_flux,
    expected_arrival,
    clim_expected_arrival,
    get_source_A_og,
    build_target_indices,
    clim_source_pushforward,
    backward_traj_from_target,
    clim_origin_likelihood,
    clim_source_cum_occ)

# ---- Use non-interactive backend for multiprocessing safety ----
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable


import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
import cmocean

from pygtm_miron.physical import physical_space


# =============================================================================
#                           Plotting utilities
# =============================================================================

def overlay_source_target(ax, d0, A_og, ind_tar):
    """
    Overlay source and target bins on geographic axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy axis on which the regions are drawn.
    d0 : physical_space
        Physical-space domain.
    A_og : array-like
        Physical-bin indices of the source region.
    ind_tar : array-like
        Physical-bin indices of the target region.
    """
    # Source
    d0.bins_contour(
        ax,
        bin_id=A_og,
        edgecolor="darkorange",
        linewidth=1.2,
        projection=ccrs.PlateCarree())

    # Active target
    d0.bins_contour(
        ax,
        bin_id=ind_tar,
        edgecolor="dodgerblue",
        linewidth=1.2,
        projection=ccrs.PlateCarree())


def geo_map(ax):
    """
    Apply common geographic formatting to a Cartopy axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy axis to format.
    """
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20], crs=ccrs.PlateCarree())
    ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cfeature.LAND, facecolor='0.88', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=1)


def plot_yearly_timeseries(values, title, ylabel, filename, paths):
    """
    Plot a yearly diagnostic time series and highligh Apolemia years

    Parameters
    ----------
    values : array-like
        One valye per analysis year.
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    filename : str
        Output filename.
    paths : dict
        Path directory contatining "plot_outdir"
    """
    plot_outdir = paths["plot_outdir"]
    
    # Convert input to arrays
    years = np.asarray(all_years, dtype=int)
    values = np.asarray(values, dtype=float)
    ap_mask = np.isin(years, np.asarray(apolemia_years, dtype=int))
    
    # Create figure
    plt.figure(figsize=(10, 4.5))

    # Plot all years
    plt.plot(years, values, color="0.55", linewidth=1.2, zorder=1)
    plt.scatter(years, values, s=28, color="0.55", zorder=2)

    # Highlight Apolemia years
    plt.scatter(
        years[ap_mask],
        values[ap_mask],
        s=70,
        color="tab:orange",
        edgecolor="black",
        linewidth=0.5,
        zorder=3)

    # Labels and formatting
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    
    fn = plot_outdir / filename
    plt.savefig(fn, dpi=250, bbox_inches="tight")
    plt.close()
    return

def plot_rel_to_clim(summary_df, metric_key, late_season, title, paths):
    """
    Plot 2x2 panel target-wise relative deviations from climatology.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Dataframe contatining yearly summary diagnostics.
    metric_key : str
        Metric prefix, "Ktot" or "gate_share".
    late_season : bool
        If True, plot late-season deviations. Otherwise full-year.
    title : str
        Figure title.
    paths : dict
        Path directory containing "plot_outdir"
    """
    
    plot_outdir = paths["plot_outdir"]

    # Make a copy so the original dataframe is untouched
    df = summary_df.copy()
    
    # Sort rows by year so the x-axis is chronological
    df = df.sort_values("year").reset_index(drop=True)
    
    # Extract years as an integer NumPy array
    years = df["year"].to_numpy(dtype=int)
    
    # Extract years as an integer NumPy array
    ap_mask = np.isin(years, np.asarray(apolemia_years, dtype=int))
    
    
    if late_season:
        cols = {tar: f"{metric_key}_clim_late_{tar}" for tar in targets}
        filename = f"{metric_key}_clim_late.png"
    else:
        cols = {tar: f"{metric_key}_clim_{tar}" for tar in targets}
        filename = f"{metric_key}_clim.png"
    
    # Compute one shared y-limit for bergen, tromsø and skagerrak
    shared_targets = ["Bergen", "Tromsø", "Brønnøysund"]
    all_vals_shared = []

    for tar in shared_targets:
        vals_pct = 100.0 * df[cols[tar]].to_numpy(dtype=float)
        vals_pct = vals_pct[np.isfinite(vals_pct)]
        if vals_pct.size > 0:
            all_vals_shared.append(vals_pct)

    if len(all_vals_shared) > 0:
        all_vals_shared = np.concatenate(all_vals_shared)
        ymin_shared = np.nanmin(all_vals_shared)
        ymax_shared = np.nanmax(all_vals_shared)
    
        pad_shared = 0.1 * (ymax_shared - ymin_shared) if (
            ymax_shared > ymin_shared) else 1.0
    
        ymin_shared = min(ymin_shared - pad_shared, 0.0)
        ymax_shared = max(ymax_shared + pad_shared, 0.0)
    else:
        ymin_shared, ymax_shared = -10.0, 10.0

    # -------------------------------------------------
    # Compute Skagerrak y-limit separately
    # -------------------------------------------------
    vals_sk = 100.0 * df[cols["Skagerrak"]].to_numpy(dtype=float)
    vals_sk = vals_sk[np.isfinite(vals_sk)]
    
    if vals_sk.size > 0:
        ymin_sk = np.nanmin(vals_sk)
        ymax_sk = np.nanmax(vals_sk)
    
        pad_sk = 0.1 * (ymax_sk - ymin_sk) if ymax_sk > ymin_sk else 1.0
    
        ymin_sk = min(ymin_sk - pad_sk, 0.0)
        ymax_sk = max(ymax_sk + pad_sk, 0.0)
    else:
        ymin_sk, ymax_sk = -10.0, 10.0
    
    # 2x2 panel layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    
    # Flatten the 2x2 axes array to a 1D list for easier looping
    axes = axes.ravel()
    
    for ax, tar in zip(axes, targets):
        col = cols[tar]
        # Convert relative values to percent for plotting
        vals_pct = 100.0 * df[col].to_numpy(dtype=float)
    
        # Draw a horizontal zero line marking climatology
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, 
                   linestyle= "--")
    
        # All years: thin gray line + small points
        ax.plot(years, vals_pct, linewidth=1.0, color="0.7", zorder=1)
        ax.scatter(years, vals_pct, s=28, color="0.6", zorder=2)
    
        # Highlight Apolemia years
        ax.scatter(
            years[ap_mask],
            vals_pct[ap_mask],
            s=55,
            color="tab:orange",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label="Apolemia year")
        
        # Set the title of this panel to the target name
        ax.set_title(tar, fontsize=12)
        
        
        # First three on same y-axis, Skagerrak gets its own
        if tar == "Skagerrak":
            ax.set_ylim(ymin_sk, ymax_sk)
        else:
            ax.set_ylim(ymin_shared, ymax_shared)
        
        ax.grid(True, alpha=0.25)

    # Axis labels
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")
    axes[0].set_ylabel("Difference from climatology (%)")
    axes[2].set_ylabel("Difference from climatology (%)")


    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.87])


    fig.savefig(plot_outdir / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return

# =============================================================================
#                       Geometry overview plots
# =============================================================================
def plot_source_target(paths):
    """
    Plot source and target regions in a four-panel map

    Parameters
    ----------
    paths : dict
        Path directory contatining "plot_outdir"
    """
    plot_outdir = paths["plot_outdir"]

    # Create original domain
    d0 = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0)

    # Create 2x2 panel figure
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = axes.ravel()

    # Loop over panel axes and target names
    for i, (ax, tar) in enumerate(zip(axes, targets)):
        ind_b = build_target_indices(d0, tar)

        # Plot all bins
        d0.bins_contour(ax, projection=ccrs.PlateCarree(), 
                        edgecolor = "0.55")

        overlay_source_target(ax, d0, A_og, ind_b)

        # Set map ticks
        ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20], crs=ccrs.PlateCarree())
        ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        
        row = i//2
        col = i % 2
        
        if row == 0:
            ax.tick_params(labelbottom=False)
    
        if col == 1:
            ax.tick_params(labelleft=False)

        # Lock map extent
        ax.set_extent(
            [d0.lon[0], d0.lon[1], d0.lat[0], d0.lat[1]],
            crs=ccrs.PlateCarree())

        # Apply existing map styling
        geo_map(ax)

        # Panel title
        ax.set_title(tar, fontsize=12)

    # Main title
    fig.suptitle("Source and target regions", fontsize=18)

    # Tight layout with space for title
    fig.tight_layout(rect=[0, 0, 1, 0.87])

    # Save and close
    fn = plot_outdir / "source_target.png"
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return

def plot_gates(paths):
    """
    Plot one map showing source bins, all target bins, and both gates. 

    Parameters
    ----------
    paths : dict
        Path directory containing "plot_outdir.
    """
    plot_outdir = paths["plot_outdir"]
    
    # Build physical domain and source indices
    d0 = physical_space(lon, lat, spatial_dis)
    ind_a = get_source_A_og(d0)
    
    # Create chosen gate 1 on original domain
    west_gate1, east_gate1 = gate_from_domain(d0, latmin_g1,
                                              latmax_g1, lonpos_g1)
    
    # Create chosen gate 2 on original domain
    west_gate2, east_gate2 = gate_from_domain(d0, latmin_g2,
                                              latmax_g2, lonpos_g2)
   
    # Create single figure
    fig,ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300, 
                          subplot_kw={"projection": ccrs.PlateCarree()})
   
    # Plot all bins
    d0.bins_contour(ax, projection=ccrs.PlateCarree(), 
                    edgecolor = "0.55")

    
    # Plot chosen gate 1 bins
    d0.bins_contour(
        ax,
        bin_id=west_gate1,
        edgecolor="purple",
        linewidth=1.2,
        projection=ccrs.PlateCarree())
    
    d0.bins_contour(
        ax,
        bin_id=east_gate1,
        edgecolor="purple",
        linewidth=1.2,
        projection=ccrs.PlateCarree())
    
    # Plot chosen gate 2 bins
    d0.bins_contour(
        ax,
        bin_id=west_gate2,
        edgecolor="forestgreen",
        linewidth=1.2,
        projection=ccrs.PlateCarree())
    
    d0.bins_contour(
        ax,
        bin_id=east_gate2,
        edgecolor="forestgreen",
        linewidth=1.2,
        projection=ccrs.PlateCarree())
    
    # Plot source once
    d0.bins_contour(
        ax,
        bin_id=ind_a,
        edgecolor="darkorange",
        linewidth=1.2,
        projection=ccrs.PlateCarree())
    
    # Plot for all targets
    for tar in targets: 
        ind_tar = build_target_indices(d0, tar)
        d0.bins_contour(
            ax,
            bin_id=ind_tar,
            edgecolor="dodgerblue",
            linewidth=1.2,
            projection=ccrs.PlateCarree())

    # Set map ticks
    ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20], crs=ccrs.PlateCarree())
    ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Lock map extent
    ax.set_extent(
        [d0.lon[0], d0.lon[1], d0.lat[0], d0.lat[1]],
        crs=ccrs.PlateCarree())

    # Apply existing map styling
    geo_map(ax)

    # Main title
    ax.set_title("Source, targets and gates regions", fontsize=16)
    
    
    # Save and close
    fn = plot_outdir / "gates.png"
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return

# =============================================================================
#                 Cumulative transition rate plots
# =============================================================================
def plot_cum_k(years, paths):
    """
    Plot cumulative transition rate and normalized cumulative transition rate
    in 2x2 target panels, one figure per year.

    Parameters
    ----------
    years : int or list[int] or None
        Analysis year or years to plot
    paths : dict
        Path directory containing "metrics_dir" and "plot_outdir"
    """
    plot_outdir = paths["plot_outdir"]
    metrics_dir = paths["metrics_dir"]
    
    out_cum = plot_outdir / "cum_k"
    out_cum.mkdir(parents=True, exist_ok=True)

    for year in years:
        data_by_target = {}

        # --------------------------------------------------
        # Load data for all 4 targets
        # --------------------------------------------------
        for tar in targets:
            pkl = metrics_dir / f"metrics_{tar}{year}.pkl"
            if not pkl.exists():
                print(f"[plot_cumulative_transition_rate_2x2]" 
                      f"Missing file: {pkl}")
                data_by_target[tar] = None
                continue
            
            # Load week and k_cum values
            d = pd.read_pickle(pkl)
            weeks_end = np.asarray(d["weeks"], dtype=float) # end week numbers
            
            # list of (K,) arrays
            kcumnorm_list = [np.asarray(x, float) 
                             for x in d.get("k_cumnorm", [])]
            kcum_list     = [np.asarray(x, float)
                             for x in d.get("k_cum", [])]
            
            
            # Store filtered data for later plotting
            data_by_target[tar] = {
                "weeks_end": weeks_end,
                "k_cumnorm": kcumnorm_list,
                "k_cum": kcum_list}
            
    
        # Color by end-week number
        norm = matplotlib.colors.Normalize(vmin=float(np.nanmin(weeks_end)),
                                    vmax=float(np.nanmax(weeks_end)))
        cmap = matplotlib.cm.viridis
        
        # --------------------------------------------------
        # Plot normalized cumulative transition rate
        # --------------------------------------------------
        
        # Create a 2x2 panel figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200,
                                 sharex=True, sharey=True)
        axes = axes.ravel()
        
        # Loop over subplot axes and target names together
        for ax, tar in zip(axes, targets):
            dd = data_by_target[tar]
            
            # Extract week labels and cumulative curves
            weeks_end = dd["weeks_end"]
            kcumnorm_list = dd["k_cumnorm"]

            # Number of times steps in the cumulative curve
            K = len(kcumnorm_list[0])
            
            # Convert step number to days into the window
            t_days = T * (np.arange(K) + 1)

            # Plot one curve for each end-week, colored by week number
            for w, y in zip(weeks_end, kcumnorm_list):
                ax.plot(t_days, y, color=cmap(norm(float(w))),
                        linewidth=1.3, alpha=0.9)
            
            # Title and formatting
            ax.set_title(tar)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            
        # Label only left and bottom panels
        axes[0].set_ylabel("Normalized cumulative transition rate [-]")
        axes[2].set_ylabel("Normalized cumulative transition rate [-]")
        axes[2].set_xlabel(f"Days into {int(T * K_win)}-day window")
        axes[3].set_xlabel(f"Days into {int(T * K_win)}-day window")
        
        # Figure title
        fig.suptitle(f"Normalized cumulative transition rate, {year}", y=0.98)
        
        # Shared colorbar showing which color corresponds to which end-week
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        fig.subplots_adjust(top=0.90, wspace=0.16, hspace=0.15)
        cbar = fig.colorbar(sm, ax=axes, location="right",
                            fraction=0.035, pad=0.04)
        cbar.set_label("End week number")
        
        # Save figure
        fig.savefig(out_cum / f"kcum_norm{year}.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # --------------------------------------------------
        # Plot non-normalized cumulative transition rate
        # --------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                                 dpi=200, sharex=True, sharey=False)
        axes = axes.ravel()

        for ax, tar in zip(axes, targets):
            dd = data_by_target[tar]

            weeks_end = dd["weeks_end"]
            kcum_list = dd["k_cum"]

            # Time axis in days
            K = len(kcum_list[0])
            t_days = T * (np.arange(K) + 1)

            # Plot all end-week curves
            for w, y in zip(weeks_end, kcum_list):
                ax.plot(t_days, y, color=cmap(norm(float(w))),
                        linewidth=1.3, alpha=0.9)

            ax.set_title(tar)
            ax.grid(True, alpha=0.25)

        axes[0].set_ylabel("k_cum")
        axes[2].set_ylabel("k_cum")
        axes[2].set_xlabel(f"Days into {int(T * K_win)}-day window")
        axes[3].set_xlabel(f"Days into {int(T * K_win)}-day window")

        fig.suptitle(f"Cumulative transition rate, {year}", y=0.98)

        # Shared colorbar showing which color corresponds to which end-week
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        
        fig.subplots_adjust(right=0.88, top=0.90, wspace=0.16, hspace=0.15)
        cbar = fig.colorbar(
            sm,
            ax=axes.tolist(),
            orientation="vertical",
            fraction=0.025,
            pad=0.02)
        
        cbar.set_label("End week number")
        
        fig.savefig(out_cum / f"k_cum{year}.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    return


def plot_clim_cum_k(paths, ref_years=None):
    """
    Plot climatological normalized cumulative transition-rate curves.

    Parameters
    ----------
    paths : dict
        Path directory containing "metrics_dir" and "plot_outdir"
    ref_years : list[int] or None, optional
        Years used to form the climatology. If None, all analysis years
    """
    
    metrics_dir = paths["metrics_dir"]
    plot_outdir = paths["plot_outdir"]
    
    # Use all years as climatology if nothing else is specified
    if ref_years is None:
        ref_years = all_years
    
    # Dict to store data by target
    data_by_target = {}
    
    # Define week axis
    week_axis = np.arange(week_step, 53, week_step, dtype=int)
    
    # ------------ Load data -----------
    for tar in targets:
        # Dict to store all curves grouped by end week
        curves_by_week = {w: [] for w in week_axis}
        
        # Load data for each years
        for year in ref_years:
            pkl = metrics_dir / f"metrics_{tar}{year}.pkl"
            
            d = pd.read_pickle(pkl)
        
            # End weeks associated with the curves
            weeks_end  = np.asarray(d.get("weeks",[]), dtype=int)
            
            # List of normalized cum. transition rate curves
            kcumnorm_list = [np.asarray(x, dtype=float) 
                             for x in d.get("k_cumnorm", [])]
        
            # Loop through all end weekd and corresponding curves
            for w, curve in zip(weeks_end, kcumnorm_list):
                curve = np.asarray(curve, dtype=float)
                if int(w) in curves_by_week:
                    curves_by_week[int(w)].append(curve)

        
        # ----------- Average curves across years, week by week ----------
        
        weeks_sorted =[]
        mean_curves = []
        
        for w in week_axis:
            
            # Stack all curves for week w into a 2D array
            mats = np.stack(curves_by_week[w], axis=0)
            
            # Take the mean across the year dimension
            mean_curve = np.nanmean(mats, axis=0)
            weeks_sorted.append(w)
            mean_curves.append(mean_curve)
        
        weeks_sorted = np.asarray(weeks_sorted, dtype=int)
        
        # Store the climatological result for this target
        data_by_target[tar]= {
            "weeks_end": weeks_sorted,
            "k_cumnorm_mean": mean_curves}
    
    # -------- Build common color normalization ------
    all_weeks =[]
    
    # Color each curve by its end week. 
    for tar in targets:
        dd = data_by_target[tar]
        all_weeks.extend(dd["weeks_end"].tolist())
        
    all_weeks = np.asarray(all_weeks, dtype=float)
    
    # Normalize week numbers to the colormap range
    norm = matplotlib.colors.Normalize(vmin=float(np.nanmin(all_weeks)),
                                        vmax=float(np.nanmax(all_weeks)))
    
    cmap = matplotlib.cm.viridis
    
    # -------- Plotting ---------
    fig,axes = plt.subplots(2,2, figsize=(12,8), dpi=200,
                            sharex=True, sharey=True)
    axes = axes.ravel()
    
    for ax, tar in zip(axes,targets):
        dd = data_by_target[tar]
        
        weeks_end = dd["weeks_end"]
        mean_curves = dd["k_cumnorm_mean"]
        
        # Number of 14-day steps in each cumulative curve
        K = len(mean_curves[0])
        
        # Convert step to physical time in days
        t_days = T * (np.arange(K)+1)
        
        # Plot one climatological mean curve for each end week
        for w,y in zip(weeks_end, mean_curves):
            ax.plot(t_days, y, color=cmap(norm(float(w))),
                    linewidth = 1.6, alpha = 0.95)
        
        ax.set_title(tar)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
    
    # Axis labels
    axes[0].set_ylabel("Normalized cumulative transition rate [-]")
    axes[2].set_ylabel("Normalized cumulative transition rate [-]")
    axes[2].set_xlabel(f"Days into {int(T*K_win)}-day transport window")
    axes[3].set_xlabel(f"Days into {int(T*K_win)}-day transport window")
    
    # Figure title
    fig.suptitle("Climatological normalized cumulative transition rate",
                 y=0.95)
    
    # Colorbar
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(top=0.90, wspace=0.16, hspace=0.15)
    cbar = fig.colorbar(sm, ax=axes, location="right",
                        fraction=0.035, pad=0.04)
    cbar.set_label("End week number")
    
    # Save figure
    fig.savefig(plot_outdir / "clim_kcum_norm.png",
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    return

# =============================================================================
#                Whole-system and source-to-target anomaly plots
# =============================================================================
def plot_whole_system_distance(distance_result, paths):
    """
    Plot whole-system transport distance for full-year and late-seasons means.

    Parameters
    ----------
    distance_result : dict
        Output from "whole_system_distance" in "diagnostics_core.py"
    paths : dict
        Path directory containing "plot_outdir.
    """
    
    # Load precomputed anomaly results
    d_tot = distance_result["d_tot"]
    d_tot_late = distance_result["d_tot_late"]
    
    # Baseline distribution for the total anomaly plot
    years_arr = np.asarray(all_years, dtype=int)
    
    clim_mask = np.isin(years_arr, clim_years)
    ref_full = np.nanmean(d_tot[clim_mask])
    ref_late = np.nanmean(d_tot_late[clim_mask])
    
    #ref_full = leave_one_out_mean(values=d_tot, years= years_arr)
    #ref_late = leave_one_out_mean(values=d_tot_late, years= years_arr)
    
    # ---------- Annual anomaly score relative to baseline median ----------
    plot_yearly_timeseries(
        values=d_tot,
        title="Yearly whole-system transport distance",
        ylabel="Whole-system transport distance [-]", 
        filename = "anom.png",
        paths = paths)
    
    # ---------- Late-season anomaly score relative to baseline median -------
    plot_yearly_timeseries(
        values=d_tot_late,
        title="Late-season whole-system transport distance",
        ylabel="Whole-system transport distance [-]", 
        filename = "anom_late.png",
        paths = paths)
    return



def plot_source_target_rel_clim(distance_result, paths):
    """
    PLot source-to-target diagnostics relative to climatology.
    
    The plotted diagnostics are total transition rate and gate share, both for
    full-year summaries and late-season summaries
    
    Parameters
    ----------
    distance_result : dict
        Whole-system transport dictionary. 
    paths : path
        Path directory containing "metrics_dir" and "plot_outdir".
    """
    # Load earlier results
    years_arr = np.asarray(all_years)
    ny = len(all_years)
    
    step_tot = distance_result["step_tot"]
    late_mask = distance_result["late_mask"]
    d_tot = distance_result["d_tot"]
            
    # Late-season summaries from step_tot
    anom_tot_late = np.full(ny, np.nan, dtype=float)

    for i in range(ny):
        # Weeks late only
        vals_late = step_tot[i, late_mask]
        
        if vals_late.size > 0 and np.any(np.isfinite(vals_late)):
            anom_tot_late[i] = float(np.nansum(vals_late))
        
    # Start summary dataframe with the flow-anomaly quantitites
    summary_df = pd.DataFrame({
        "year": years_arr,
        "is_apolemia": np.isin(years_arr, apolemia_years).astype(int),
        "anom_total": d_tot,
        "anom_tot_late": anom_tot_late})
    
    # ------ Add summaries from stored transport metrics --------
    summary_df = add_source_target_diagnostics(summary_df=summary_df, 
                                               paths = paths)

    
    # -----Full-year total transition rate relative to climatology ----------
    plot_rel_to_clim(
        summary_df=summary_df,
        metric_key="Ktot",
        late_season=False,
        title="Yearly total transition rate relative to climatology",
        paths = paths)
    
    # Plot late-season Ktot relative to climatology for all targets
    plot_rel_to_clim(
        summary_df=summary_df,
        metric_key="Ktot",
        late_season=True,
        title="Late-season total transition rate relative to climatology",
        paths = paths)
    
    # ---------- Full-year gate share relative to climatology ----------
    plot_rel_to_clim(
        summary_df=summary_df,
        metric_key="gate_share",
        late_season=False,
        title="Yearly gate share relative to climatology",
        paths = paths)
    
    # Plot late-season gate share relative to climatology for all targets
    plot_rel_to_clim(
        summary_df=summary_df,
        metric_key="gate_share",
        late_season=True,
        title="Late-season gate share relative to climatology",
        paths = paths)
 
    return

# =============================================================================
#                     Effective-current maps
# =============================================================================
def plot_flux_map(year, tar, paths, week_range= None):
    """
    Plot mean effective-current field for one year, target, and week interval

    Parameters
    ----------
    year : int
        Analysis year. 
    tar : str
        Target name. 
    paths : dict
        Path directory containing "flux_field_dir" and "plot_outdir"
    week_range : tuple[int, int] or None, optional
        Inclusive end-week interval. If None, late-season interval is used
    """
    plot_outdir = paths["plot_outdir"]
    
    
    Fx, Fy, vx, vy = mean_flux(year, tar, paths, week_range = week_range)
    
    # Convert inputs
    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    # Bin centers
    x_c = 0.5 * (vx[:-1] + vx[1:])
    y_c = 0.5 * (vy[:-1] + vy[1:])
    Xc, Yc = np.meshgrid(x_c, y_c)
    
    # Create physical space
    d_tmin = physical_space(lon, lat, spatial_dis)

    # Shape check
    if Fx.shape != Xc.shape or Fy.shape != Xc.shape:
        raise ValueError(
            f"Shape mismatch: Fx.shape={Fx.shape}, Fy.shape={Fy.shape}, "
            f"grid shape={Xc.shape}"
        )

    # Magnitude
    mag = np.hypot(Fx, Fy)
    finite_mag = np.isfinite(mag)
    
    #vmax_used = np.nanpercentile(mag[finite_mag], 99)
    vmax_used = 0.013
    norm = matplotlib.colors.Normalize(vmin=0, vmax = vmax_used)

    # Normalize arrows so they show direction only
    eps = 1e-15
    Fx_norm = np.full_like(Fx, np.nan, dtype=float)
    Fy_norm = np.full_like(Fy, np.nan, dtype=float)

    valid = np.isfinite(mag) & (mag > eps)
    Fx_norm[valid] = Fx[valid] / mag[valid]
    Fy_norm[valid] = Fy[valid] / mag[valid]

    # Use magnitude as color field
    color_plot = mag.copy()

    # Mask weak arrows
    weak = color_plot < 0.02 * vmax_used
    Fx_norm[weak] = np.nan
    Fy_norm[weak] = np.nan
    color_plot[weak] = np.nan
    
    fig, ax = plt.subplots(
        1, 1,
        figsize=(7, 6),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    
    # Plot arrows
    Q = ax.quiver(
        Xc, Yc,
        Fx_norm, Fy_norm, color_plot,
        cmap=cmocean.cm.ice_r,
        norm = norm,
        scale=0.8,
        scale_units="xy",
        angles="xy",
        width=0.003,
        headwidth=3,
        headlength=4,
        pivot="middle",
        zorder=0,
        transform=ccrs.PlateCarree())
    
    Q.set_clim(vmin=0.0, vmax=vmax_used)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%",
                              pad=0.04, axes_class=plt.Axes)
    cbar = fig.colorbar(Q, cax=cax)
    cbar.set_label("Effective current magnitude [-]")
    
    # Plot target bins
    A_og = get_source_A_og(d_tmin)
    ind_tar = build_target_indices(d_tmin, tar)
    overlay_source_target(ax, d_tmin, A_og, ind_tar)

    # Map formatting
    geo_map(ax)
    title = f"Mean late-season effective current, {tar} {year}"
    ax.set_xticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
    ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(
        [d_tmin.lon[0], d_tmin.lon[1], d_tmin.lat[0], d_tmin.lat[1]],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(title)

    outname = plot_outdir / f"flux_{tar}{year}"

    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)    

def plot_clim_flux_map(tar, paths, week_range=None):
    """
    Plot climatological mean effective-current field for one target

    Parameters
    ----------
    tar : str
        Target name. 
    paths : dict
        Path directory containing "flux_field_dir" and "plot_outdir"
    week_range : tuple[int, int] or None, optional
        Inclusive end-week interval. If None, late-season interval is used
    """
    plot_outdir = paths["plot_outdir"]
    
    Fx, Fy, vx, vy = climatology_flux(tar, paths, week_range = week_range)
    
    # Convert inputs
    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)

    # Bin centers
    x_c = 0.5 * (vx[:-1] + vx[1:])
    y_c = 0.5 * (vy[:-1] + vy[1:])
    Xc, Yc = np.meshgrid(x_c, y_c)
    
    # Create physical space
    d_tmin = physical_space(lon, lat, spatial_dis)

    # Shape check
    if Fx.shape != Xc.shape or Fy.shape != Xc.shape:
        raise ValueError(
            f"Shape mismatch: Fx.shape={Fx.shape}, Fy.shape={Fy.shape}, "
            f"grid shape={Xc.shape}"
        )

    # Magnitude
    mag = np.hypot(Fx, Fy)
    finite_mag = np.isfinite(mag)
    
    #vmax_used = np.nanpercentile(mag[finite_mag], 99)
    #norm = matplotlib.colors.Normalize(vmin=0, vmax = vmax_used)
    vmax_used = 0.015
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_used)

    # Normalize arrows so they show direction only
    eps = 1e-15
    Fx_norm = np.full_like(Fx, np.nan, dtype=float)
    Fy_norm = np.full_like(Fy, np.nan, dtype=float)

    valid = np.isfinite(mag) & (mag > eps)
    Fx_norm[valid] = Fx[valid] / mag[valid]
    Fy_norm[valid] = Fy[valid] / mag[valid]

    # Use magnitude as color field
    color_plot = mag.copy()

    # Mask weak arrows
    weak = color_plot < 0.02 * vmax_used
    Fx_norm[weak] = np.nan
    Fy_norm[weak] = np.nan
    color_plot[weak] = np.nan
    
    fig, ax = plt.subplots(
        1, 1,
        figsize=(7, 6),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    
    # Plot arrows
    Q = ax.quiver(
        Xc, Yc,
        Fx_norm, Fy_norm, color_plot,
        cmap=cmocean.cm.ice_r,
        norm = norm,
        scale=0.8,
        scale_units="xy",
        angles="xy",
        width=0.003,
        headwidth=3,
        headlength=4,
        pivot="middle",
        zorder=2,
        transform=ccrs.PlateCarree())
    
    Q.set_clim(vmin=0.0, vmax=vmax_used)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04,
                              axes_class=plt.Axes)
    cbar = fig.colorbar(Q, cax=cax)
    cbar.set_label("Effective current [-]")
    
    # Plot target bins
    A_og = get_source_A_og(d_tmin)
    ind_tar = build_target_indices(d_tmin, tar)
    overlay_source_target(ax, d_tmin, A_og, ind_tar)

    # Map formatting
    geo_map(ax)
    title = f"Climatology late-season effective current, {tar}"
    ax.set_xticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
    ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(
        [d_tmin.lon[0], d_tmin.lon[1], d_tmin.lat[0], d_tmin.lat[1]],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title(title)

    outname = plot_outdir / f"clim_flux_{tar}"

    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_flux_map_anomaly(year, tar, paths, week_range=None):
    """
    Plot anomaly in effective-current magnitude relative to climatology.
    
    Arrows show the direction of the vector difference, while colors show the
    percent anomaly in effective-current magnitude relative to the 
    climatological field

    Parameters
    ----------
    year: int
        Analysis year
    tar : str
        Target name. 
    paths : dict
        Path directory containing "flux_field_dir" and "plot_outdir"
    week_range : tuple[int, int] or None, optional
        Inclusive end-week interval. If none, late-season interval is used
    """
    plot_outdir = paths["plot_outdir"]
    
    Fx_ref, Fy_ref, vx, vy = climatology_flux(tar, paths, week_range)
    Fx_cmp, Fy_cmp, _, _ = mean_flux(year, tar, paths, week_range)
    
    # Bin centers
    x_c = 0.5 * (vx[:-1] + vx[1:])
    y_c = 0.5 * (vy[:-1] + vy[1:])
    Xc, Yc = np.meshgrid(x_c, y_c)
    
    # Create physical space
    d_tmin = physical_space(lon, lat, spatial_dis)

    # Shape checks
    expected_shape = Xc.shape
    for name, arr in [
        ("Fx_ref", Fx_ref), ("Fy_ref", Fy_ref),
        ("Fx_cmp", Fx_cmp), ("Fy_cmp", Fy_cmp)]:
        
        if arr.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {name}.shape={arr.shape}",
                f"expected {expected_shape}")

    # Vector difference
    dFx = Fx_cmp - Fx_ref
    dFy = Fy_cmp - Fy_ref

    # Magnitudes
    mag_ref = np.hypot(Fx_ref, Fy_ref)
    mag_cmp = np.hypot(Fx_cmp, Fy_cmp)
    mag_diff = np.hypot(dFx, dFy)

    # Normalize anomaly arrows to show direction only
    eps = 1e-15
    dFx_norm = np.full_like(dFx, np.nan, dtype=float)
    dFy_norm = np.full_like(dFy, np.nan, dtype=float)

    valid = np.isfinite(mag_diff) & (mag_diff > eps)
    dFx_norm[valid] = dFx[valid] / mag_diff[valid]
    dFy_norm[valid] = dFy[valid] / mag_diff[valid]

    # Percent anomaly relative to reference magnitude
    pct_anom = 100.0 * (mag_cmp - mag_ref) / np.maximum(mag_ref, 1e-12)

    # Mask cells where ref. magnitude is too weak for a stable percent anomaly
    finite_ref = np.isfinite(mag_ref)

    thr_ref = 0.05 * np.nanpercentile(mag_ref[finite_ref], 95)
    weak_mask = mag_ref < thr_ref
    pct_anom[weak_mask] = np.nan
    dFx_norm[weak_mask] = np.nan
    dFy_norm[weak_mask] = np.nan

    # Also mask non-finite anomaly cells
    bad = ~np.isfinite(pct_anom)
    dFx_norm[bad] = np.nan
    dFy_norm[bad] = np.nan

    # Symmetric color scale
    finite_anom = np.isfinite(pct_anom)

    #vabs = np.nanpercentile(np.abs(pct_anom[finite_anom]), 95)
    #vabs = max(vabs, 10.0)
    vabs = 80
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots(
        1, 1,
        figsize=(7, 6),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Plot difference arrows
    Q = ax.quiver(
        Xc, Yc,
        dFx_norm, dFy_norm, pct_anom,
        cmap="RdBu_r",
        norm = norm,
        scale=0.8,
        scale_units="xy",
        angles="xy",
        width=0.003,
        headwidth=3,
        headlength=4,
        pivot="middle",
        zorder=0,
        transform=ccrs.PlateCarree())
    
    Q.set_clim(vmin=-vabs, vmax=vabs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.04,
                              axes_class=plt.Axes)
    cbar = fig.colorbar(Q, cax=cax)
    cbar.set_label("Effective current anomaly [%]")
    cbar.set_ticks(np.arange(-80, 81, 20))
    
    # Plot target bins
    A_og = get_source_A_og(d_tmin)
    ind_tar = build_target_indices(d_tmin, tar)
    overlay_source_target(ax, d_tmin, A_og, ind_tar)
    
    # Map formatting
    geo_map(ax)
    title = f"Effective current anomaly late-season, {tar} {year}"
    ax.set_xticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
    ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(
        [d_tmin.lon[0], d_tmin.lon[1], d_tmin.lat[0], d_tmin.lat[1]],
        crs=ccrs.PlateCarree())
    
    ax.set_title(title)

    outname = plot_outdir / f"flux_diff_{tar}{year}"

    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_clim_flux_panel(paths, week_range=None, shared_colorbar=True):
    """
    Plot climatological mean effective-current fields for all targets

    Parameters
    ----------
    paths : dict
        Path directory contatining "plot_outdir" and "flux_field_dir"
    week_range : tuple(int, int) or None, optional
        Inclusive end-week interval. If None, the late-season interval is used
    shared_colorbar : bool, optional
        If True, all panels use a common color scale
    """
    plot_outdir = paths["plot_outdir"]
    
    if week_range is None:
        week_label = f"{late_week_range[0]}-{late_week_range[1]}"
    else: 
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
            "week_range must be a tuple interval, for example (38, 52).")
            
        week_label = f"{week_range[0]}-{week_range[1]}"
        
    target_names = list(targets.keys())
    
    # Common physical domain for overlays
    d0 = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0)
    target_inds = {tar: build_target_indices(d0, tar) for tar in target_names}
    
    # Compute fluxes and common scale
    out = {}
    vmax_vals = []

    for tar in target_names:
        Fx_clim, Fy_clim, vx, vy = climatology_flux(tar, paths, week_range)

        mag = np.hypot(Fx_clim, Fy_clim)
        finite = np.isfinite(mag)
        
        
        if np.any(finite):
            vmax_tar = np.nanpercentile(mag[finite], 99)
            if np.isfinite(vmax_tar) and vmax_tar >0:
                vmax_vals.append(vmax_tar)
            else:
                vmax_tar = np.nan

        out[tar] = {
            "Fx": Fx_clim,
            "Fy": Fy_clim,
            "vx": vx,
            "vy": vy,
            "vmax": vmax_tar}

    vmax_vals = [v for v in vmax_vals if np.isfinite(v) and v > 0]
    vmax_common = max(vmax_vals)

    # Create 2x2 panel figure
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = axes.ravel()


    for i, (ax, tar) in enumerate(zip(axes, target_names)):
        Fx = out[tar]["Fx"]
        Fy = out[tar]["Fy"]
        vx = out[tar]["vx"]
        vy = out[tar]["vy"]
        
        # Bin centers from bin edges
        x_c = 0.5 * (vx[:-1] + vx[1:])
        y_c = 0.5 * (vy[:-1] + vy[1:])
        Xc, Yc = np.meshgrid(x_c, y_c)
        
        
        mag = np.hypot(Fx, Fy)
        
        # Choose scale
        if shared_colorbar:
            vmax_use = vmax_common
        else:
            vmax_use = out[tar]["vmax"]
            
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax = vmax_use)
        
        # Direction-only arrows, color = magnitude
        eps = 1e-15
        Fx_plot = np.full_like(Fx, np.nan, dtype=float)
        Fy_plot = np.full_like(Fy, np.nan, dtype=float)
    
        valid = np.isfinite(Fx) & np.isfinite(Fy) & np.isfinite(mag) & (
            mag > eps)
        Fx_plot[valid] = Fx[valid] / mag[valid]
        Fy_plot[valid] = Fy[valid] / mag[valid]
    
        # Mask weak arrows
        weak = mag < 0.02 * vmax_use
        Fx_plot[weak] = np.nan
        Fy_plot[weak] = np.nan
    
        mag_plot = mag.copy()
        mag_plot[weak] = np.nan
    
        Q = ax.quiver(
            Xc, Yc,
            Fx_plot, Fy_plot, mag_plot,
            cmap=cmocean.cm.ice_r,
            norm=norm,
            transform=ccrs.PlateCarree(),
            scale=0.8,
            scale_units="xy",
            angles="xy",
            width=0.003,
            headwidth=3,
            headlength=4,
            pivot="middle",
            zorder=0)
        
        # Source and target
        ind_b = target_inds[tar]
        overlay_source_target(ax, d0, A_og, ind_b)
        

        geo_map(ax)
        ax.set_title(tar)
        ax.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())
        ax.set_xticks([-10, 0, 10, 20], crs=ccrs.PlateCarree())
        ax.set_yticks([50, 55, 60, 65, 70], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        
        # Show longitude labels and latitude labels only once
        row = i//2
        col = i%2

        if row == 0:
            ax.tick_params(labelbottom=False)
    
        if col == 1:
            ax.tick_params(labelleft=False)
            
        
        if not shared_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05,
                                      axes_class=plt.Axes)
            cbar = fig.colorbar(Q, cax=cax)
            cbar.set_label("Effective current magnitude [-]")
    
    
    # Shared colorbar
    if shared_colorbar:
        sm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0.0, vmax=vmax_common),
            cmap=cmocean.cm.ice_r,
        )
        sm.set_array([])
    
        fig.subplots_adjust(
            left=0.05,
            right=0.88,
            bottom=0.06,
            top=0.88,
            wspace=0.06,
            hspace=0.04)
        
        cbar = fig.colorbar(
            sm,
            ax=axes.tolist(),
            location="right",
            fraction=0.035,
            pad=0.04)
        
        cbar.set_label("Effective current magnitude [-]")
        
        outname = plot_outdir / (
            f"clim_flux_shared_W{week_label.replace('-', '_')}.png")
    else:
        fig.subplots_adjust(
            left=0.06,
            right=0.95,
            bottom=0.07,
            top=0.88,
            wspace=0.25,
            hspace=0.04)
    
        outname = plot_outdir / (
            f"clim_flux_panelwise_W{week_label.replace('-', '_')}.png")
    
    fig.suptitle(
        "Climatological late-season effective current",
        fontsize=14,
        y=0.92)
    
    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# =============================================================================
#             Trajectory, arrival, and origin plots
# =============================================================================
def plot_backward_traj(year, paths, end_week, num_horizon_weeks, 
                       n_per_bin=100, dt_days=0.05):
    """
    Plot backward-integrated trajectories from all targets for one year.
    The figure is arranged as a 2x2 panel plot with one panel for each target.

    Parameters
    ----------
    year : int
        Analysis year.
    end_week : int, optional
        ISO week number whose Sunday is used as the launch date.
    num_horizon_weeks : int, optional
        Backward integration horizon in weeks.
    n_per_bin : int, optional
        Number of particles released per target bin. Default is 100.
    dt_days : float, optional
        Time step in days used for the backward integration. Default is 0.05.
    """
    plot_outdir = paths["plot_outdir"]
    
    # Create 2x2 panel figure
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.ravel()

    # Loop over suplot axes and target names
    for i, (ax, tar) in enumerate(zip(axes, targets)):
        
        # Compute backward trajectories for this target
        d0, ind_b, x_launch, x_final, traj, time_traj = backward_traj_from_target(
            year=year,
            tar=tar,
            end_week=end_week,
            num_horizon_weeks=num_horizon_weeks,
            n_per_bin=n_per_bin,
            dt_days=dt_days)

        # target bins
        d0.bins_contour(
            ax,
            bin_id=ind_b,
            edgecolor="dodgerblue",
            linewidth=1.0,
            projection=ccrs.PlateCarree())

        # trajectory curves
        ntraj = traj.shape[2]
        for j in range(ntraj):
            ax.plot(
                traj[:, 0, j],
                traj[:, 1, j],
                linewidth=0.5,
                alpha=0.25,
                transform=ccrs.PlateCarree(),
                zorder=2)

        # launch points at target
        ax.scatter(
            x_launch[0, :],
            x_launch[1, :],
            s=8,
            color="green",
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Launch")

        # backward end positions 
        ax.scatter(
            x_final[0, :],
            x_final[1, :],
            s=8,
            color="red",
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Backward end")

        geo_map(ax)
        ax.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())
        ax.set_title(tar)

    fig.suptitle(
        f"Backward trajectories from target regions, {year}, week {end_week}",
        fontsize=14,
        y=0.95)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outname = plot_outdir / (
        f"backward_traj_{year}_W{end_week:02d}_{num_horizon_weeks}w.png")
    fig.savefig(outname, bbox_inches="tight")
    plt.close(fig)


def plot_clim_expected_arrival(paths):
    """
    Plot 2x2 climatological expected-arrival-time maps for all targets.

    Parameters
    ----------
    paths : dict
        Path directory containing "plot_outdir" and "window_ops_dir"
    """
    plot_outdir = paths["plot_outdir"]
    
    target_names = list(targets.keys())

    out = {}
    vmax_vals = []

    # Compute maps and collect common color scale
    for tar in target_names:
        tau_clim, hit_total, d0, ind_b = clim_expected_arrival(tar=tar,
                                                               paths=paths)

        # convert 1D physical-bin vectors to 2D grid fields
        tau_grid = d0.vector_to_matrix(tau_clim)
        hit_grid = d0.vector_to_matrix(hit_total)
        
        # Convert possible MaskedArrays to ordinary arrays.
        # Masked cells become np.nan.
        tau_grid = np.ma.filled(tau_grid, np.nan).astype(float)
        hit_grid = np.ma.filled(hit_grid, np.nan).astype(float)

        finite_tau = np.isfinite(tau_grid)
        if np.any(finite_tau):
            vmax_vals.append(np.nanpercentile(tau_grid[finite_tau], 95))

        out[tar] = {
            "tau_grid": tau_grid,
            "hit_grid": hit_grid,
            "d0": d0,
            "ind_b": ind_b}


    vmax_common = max(vmax_vals)

    # 2x2 plot
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()})
    
    axes = axes.ravel()

    mappable = None

    for i, (ax, tar) in enumerate(zip(axes, target_names)):
        tau_grid = out[tar]["tau_grid"]
        hit_grid = out[tar]["hit_grid"]
        d0 = out[tar]["d0"]
        ind_b = out[tar]["ind_b"]

        # mask boxes with weak climatological target reachability
        finite_hit = np.isfinite(hit_grid) & (hit_grid > 0.0)
        if np.any(finite_hit):
            thr = np.nanpercentile(hit_grid[finite_hit], 20)
            tau_plot = np.where(hit_grid >= thr, tau_grid, np.nan)
        else:
            tau_plot = np.full_like(tau_grid, np.nan)

        pc = ax.pcolormesh(
            d0.vx,
            d0.vy,
            tau_plot,
            cmap=cmocean.cm.tempo,
            vmin=0.0,
            vmax=vmax_common,
            transform=ccrs.PlateCarree())
        
        mappable = pc

        # target bins
        d0.bins_contour(
            ax,
            bin_id=ind_b,
            edgecolor="dodgerblue",
            linewidth=1.0,
            projection=ccrs.PlateCarree())

        geo_map(ax)
        ax.set_extent([lon[0], lon[1], lat[0], lat[1]], crs=ccrs.PlateCarree())
        ax.set_title(tar)
        
        row = i // 2
        col = i % 2
        if row == 0:
            ax.tick_params(labelbottom=False)

        if col == 1:
            ax.tick_params(labelleft=False)
        
    fig.subplots_adjust(
        left=0.05,
        right=0.88,
        bottom=0.06,
        top=0.88,
        wspace=0.10,
        hspace=0.04)    
    
    cbar = fig.colorbar(
        mappable,
        ax=axes.tolist(),
        location="right",
        fraction=0.035,
        pad=0.02)
    
    cbar.set_label("Expected arrival time [days]")
    
    # More colorbar ticks
    ticks = np.linspace(0.0, vmax_common, 8)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.0f}" for t in ticks])

    title_str = "Climatological late-season expected arrival time"
    fname = f"clim_expected_arrivalW{late_week_range[0]}_{late_week_range[1]}.png"

    fig.suptitle(title_str, fontsize=14, y=0.92)
    fig.savefig(plot_outdir / fname, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
def plot_clim_origin(paths, week_range=None):
    """
    Plot climatological final source push-forward density. 

    Parameters
    ----------
    paths : dict
        Path directory containing "plot_outdir" and "window_ops_dir"
    week_range : iterable of int, tuple(int, int), or None, optional
        Inclusive end-week interval. If None, late-season interval is used.
    """

    plot_outdir = paths["plot_outdir"]

    # Choose weeks
    if week_range is None:
        weeks_plot = range(late_week_range[0], late_week_range[1] + 1,
                           week_step)
        week_label = f"{late_week_range[0]}-{late_week_range[1]}"
    
    elif isinstance(week_range, (tuple, list)) and len(week_range) == 2:
        weeks_plot = range(int(week_range[0]), int(week_range[1]) + 1,
                           week_step)
        week_label = f"{week_range[0]}-{week_range[1]}"
    
    else:
        weeks_plot = [int(w) for w in week_range]
        week_label = f"{min(weeks_plot)}-{max(weeks_plot)}"

    target_names = list(targets.keys())

    # Common domain only used for source overlay
    d0_common = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0_common)

    # compute one climatological origin field for each target
    # and collect a common color scale.
    out = {}
    vmax_vals = []

    for tar in target_names:

        # Compute climatological origin likelihood for this target
        origin_clim, hit_clim, d0, ind_b = clim_origin_likelihood(
            tar=tar,
            paths=paths,
            week_range=week_range)

        # Convert 1D physical-bin vector to 2D grid field
        origin_grid = d0.vector_to_matrix(origin_clim)

        # vector_to_matrix may return a MaskedArray.
        # Convert masked cells to NaN before plotting.
        origin_grid = np.ma.filled(origin_grid, np.nan).astype(float)

        # Hide zero values so the whole rectangular domain is not colored
        origin_plot = origin_grid.copy()
        origin_plot[origin_plot <= 0.0] = np.nan

        out[tar] = {
            "origin_grid": origin_plot,
            "hit_clim": hit_clim,
            "d0": d0,
            "ind_b": ind_b}

        # Use only positive finite values for the color scale
        finite = np.isfinite(origin_plot) & (origin_plot > 0.0)
        if np.any(finite):
            vmax_tar = np.nanpercentile(origin_plot[finite], 99)
            if np.isfinite(vmax_tar) and vmax_tar > 0.0:
                vmax_vals.append(vmax_tar)


    # Common upper color limit across all panels
    vmax_common = max(vmax_vals)

    # ---------------------------------------------------------
    # Create 2x2 panel figure
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()})
    
    axes = axes.ravel()

    mappable = None

    for i, (ax, tar) in enumerate(zip(axes, target_names)):

        origin_grid = out[tar]["origin_grid"]
        d0 = out[tar]["d0"]
        ind_b = out[tar]["ind_b"]

        # Plot origin likelihood
        pc = ax.pcolormesh(
            d0.vx,
            d0.vy,
            origin_grid,
            cmap=cmocean.cm.matter,
            vmin=0.0,
            vmax=vmax_common,
            shading="auto",
            transform=ccrs.PlateCarree())
        
        mappable = pc

        # Source outline
        d0.bins_contour(
            ax,
            bin_id=A_og,
            edgecolor="darkorange",
            linewidth=1.3,
            projection=ccrs.PlateCarree())

        # Target outline
        d0.bins_contour(
            ax,
            bin_id=ind_b,
            edgecolor="dodgerblue",
            linewidth=1.3,
            projection=ccrs.PlateCarree())

        # Map formatting
        geo_map(ax)

        ax.set_extent(
            [lon[0], lon[1], lat[0], lat[1]],
            crs=ccrs.PlateCarree())

        ax.set_title(tar)

        # Show longitude labels only on bottom row,
        # and latitude labels only on left column.
        row = i // 2
        col = i % 2

        if row == 0:
            ax.tick_params(labelbottom=False)

        if col == 1:
            ax.tick_params(labelleft=False)

    # ---------------------------------------------------------
    # Shared colorbar
    # ---------------------------------------------------------
    fig.subplots_adjust(
        left=0.05,
        right=0.88,
        bottom=0.06,
        top=0.80,
        wspace=0.12,
        hspace=0.04)

    cbar = fig.colorbar(
        mappable,
        ax=axes.tolist(),
        location="right",
        fraction=0.035,
        pad=0.02)
    
    cbar.set_label("Origin likelihood [-]")

    # More readable colorbar ticks
    ticks = np.linspace(0.0, vmax_common, 8)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.3f}" for t in ticks])

    # Figure title
    fig.suptitle(
        "Climatological late-season origin likelihood",
        fontsize=14,
        y=0.90)

    # Save figure
    outname = plot_outdir / (
        f"clim_origin_likelihood_W{week_label.replace('-', '_')}.png")

    fig.savefig(outname, dpi=250)
    plt.close(fig)

    print(f"[OK] Wrote {outname}")

# =============================================================================
#                   Source-density plot
# =============================================================================
def plot_clim_source_pushforward(paths, week_range=None):
    """
    Plot climatological source push-forward density.

    Parameters
    ----------
    paths : dict
        Path directory containing "plot_outdir" and "window_ops_dir".
    week_range : tuple(int, int) or None, optional
        Inclusive end-week interval. If None, late-season range is used.
    """
    
    plot_outdir = paths["plot_outdir"]

    # Climatological push-forward source density
    rho_clim, d0, A_og = clim_source_pushforward(
        paths=paths,
        week_range=week_range)

    # Convert 1D physical-bin vector to 2D grid field
    rho_grid = d0.vector_to_matrix(rho_clim)
    rho_grid = np.ma.filled(rho_grid, np.nan).astype(float)

    # Build filename label. Do not convert week_range to explicit weeks here.
    if week_range is None:
        week_label = f"{late_week_range[0]}-{late_week_range[1]}"
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
            
        week_label = f"{int(week_range[0])}-{int(week_range[1])}"
        
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 7),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()})
    
    # Use only positive finite densities when choosing color scale
    finite = np.isfinite(rho_grid) & (rho_grid > 0)
    if np.any(finite):
        vmax = np.nanpercentile(rho_grid[finite], 99)
    else:
        vmax = 1.0
    
    rho_plot = rho_grid.copy()
    rho_plot[rho_plot<=0.0] = np.nan
    
    # Plot terminal pushed-forward density
    im = ax.pcolormesh(
        d0.vx,
        d0.vy,
        rho_plot,
        cmap=cmocean.cm.matter,
        vmin=0.0,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree())

    # Source outline
    d0.bins_contour(
        ax,
        bin_id=A_og,
        edgecolor="darkorange",
        linewidth=1.3,
        projection=ccrs.PlateCarree())

    # Outline all target regions
    for tar in targets:
        ind_tar = build_target_indices(d0, tar)
        d0.bins_contour(
            ax,
            bin_id=ind_tar,
            edgecolor="dodgerblue",
            linewidth=1.0,
            projection=ccrs.PlateCarree())
    
    geo_map(ax)
    
    ax.set_extent(
        [d0.lon[0], d0.lon[1], d0.lat[0], d0.lat[1]],
        crs=ccrs.PlateCarree())

    ax.set_title(
        "Climatological late-season terminal density")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%",
                              pad=0.04, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Terminal density [-]")

    outname = plot_outdir / f"clim_source_pushforward_W{week_label.replace('-', '_')}.png"

    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    
    
def plot_clim_source_occ(paths, week_range=None):
    """
    Plot climatological cumulative source occupancy.
    
    Parameters
    ----------
    paths : dict
        Path directory containing ``plot_outdir`` and ``window_ops_dir``.

    week_range : iterable of int, tuple(int, int), or None, optional
        Inclusive end-week interval. If none, late-season interval is used
    """

    plot_outdir = paths["plot_outdir"]

    # Build filename label. Do not convert week_range to explicit weeks here.
    if week_range is None:
        week_label = f"{late_week_range[0]}-{late_week_range[1]}"
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
            
        week_label = f"{int(week_range[0])}-{int(week_range[1])}"
        

    # Compute climatological cumulative occupancy
    occ_clim, d0, A_og = clim_source_cum_occ(
        paths=paths,
        week_range=week_range)

    # Convert 1D physical-bin vector to 2D grid field
    occ_grid = d0.vector_to_matrix(occ_clim)
    occ_grid = np.ma.filled(occ_grid, np.nan).astype(float)


    # Hide zero/nonpositive val. so the whole rectangular domain is not colored
    occ_plot = occ_grid.copy()
    occ_plot[occ_plot <= 0.0] = np.nan

    # Color scale from positive finite values
    finite = np.isfinite(occ_plot) & (occ_plot > 0.0)

    if np.any(finite):
        vmax = np.nanpercentile(occ_plot[finite], 99)
    else:
        vmax = 1.0

    # Plot
    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 7),
        dpi=250,
        subplot_kw={"projection": ccrs.PlateCarree()})

    im = ax.pcolormesh(
        d0.vx,
        d0.vy,
        occ_plot,
        cmap=cmocean.cm.matter,
        vmin=0.0,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree())

    # Source outline
    d0.bins_contour(
        ax,
        bin_id=A_og,
        edgecolor="darkorange",
        linewidth=1.4,
        projection=ccrs.PlateCarree())

    # Outline all target regions
    for tar in targets:
        ind_tar = build_target_indices(d0, tar)
        d0.bins_contour(
            ax,
            bin_id=ind_tar,
            edgecolor="dodgerblue",
            linewidth=1.0,
            projection=ccrs.PlateCarree())

    # Map formatting
    geo_map(ax)

    ax.set_extent(
        [d0.lon[0], d0.lon[1], d0.lat[0], d0.lat[1]],
        crs=ccrs.PlateCarree())

    # Colorbar label
    cbar_label = "Residence time [days]"

    ax.set_title("Climatological late-season residence time")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right",
        size="3.5%",
        pad=0.04,
        axes_class=plt.Axes)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)

    # More readable colorbar ticks
    ticks = np.linspace(0.0, vmax, 8)
    cbar.set_ticks(ticks)

    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])

    outname = plot_outdir / (
        f"clim_source_occupancy_W{week_label.replace('-', '_')}.png")

    fig.savefig(outname, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Wrote {outname}")
