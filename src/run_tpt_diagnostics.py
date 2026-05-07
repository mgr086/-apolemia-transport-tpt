# =============================================================================
#                         run_tpt_diagnostics
# =============================================================================
"""
Driver script for finite-time transition path diagnostics

This script coordinates the full diagnostics workflow: 

    1. Compute source-to-target TPT metrics, full-window operators, and
    effective-current fields. 
    2. Compute whole-system transport distances relative to climatology
    3. Produce geometry, cumulative-transition-rate, anomaly,
    effectice-current, source-density, origin-likelihood, and trajectory plots.
"""

from config import OUTPUT_DIR, late_week_range

from diagnostics_core import (
    main_metrics,
    analysis_setup,
    whole_system_distance,
    mean_travel)

from diagnostics_plots import (
    plot_source_target,
    plot_gates,
    plot_clim_cum_k,
    plot_whole_system_distance,
    plot_source_target_rel_clim, 
    plot_backward_traj,
    plot_clim_expected_arrival,
    plot_flux_map,
    plot_clim_flux_map,
    plot_flux_map_anomaly,
    plot_clim_flux_panel,
    plot_clim_source_pushforward,
    plot_clim_origin,
    plot_clim_source_occ)

from datetime import datetime
import gc    

# =============================================================================
#                         Run configuration
# =============================================================================

RUN_NAME = "Run0505_1741"
RUN_METRICS = False
RUN_ANOMALY = False
RUN_STANDARD_PLOTS = False
RUN_FLUX_PLOTS = False
RUN_SOURCE_DENSITY_PLOTS = False
RUN_HEAVY_PLOTS = True


# None means that the plotting functions use "late_week_range"
default_week_range = None
late_season_range = late_week_range

# Settings for backward-trajectory plots.
backward_traj_year = 2021
backward_end_week = 45
backward_horizon_weeks = 36

# Example years and targets used for individual effectice-current maps
flux_examples = [(2024, "Bergen"), (2001, "Bergen")]

# =============================================================================
#                       Path setup
# =============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if RUN_NAME is None:
    stamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"Run{stamp}"
else:
    run_name = RUN_NAME    

outdir = OUTPUT_DIR / run_name
outdir.mkdir(parents=True, exist_ok=True)

# Folder for plots
plot_outdir = outdir / "plots"
plot_outdir.mkdir(parents=True, exist_ok=True)

# Folder for weekly flux fields
flux_field_dir = outdir / "flux_fields"
flux_field_dir.mkdir(parents=True, exist_ok=True)

# Folder for metrics
metrics_dir = outdir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

# Folder for multi-step operator
window_ops_dir = outdir / "multi_step"
window_ops_dir.mkdir(parents=True, exist_ok=True)

paths = {
    "outdir": outdir,
    "plot_outdir": plot_outdir,
    "flux_field_dir": flux_field_dir,
    "metrics_dir": metrics_dir,
    "window_ops_dir": window_ops_dir}

# =============================================================================
#                     Main workflow
# =============================================================================
def main():
    """
    Run the selected diagnostics workflow
    """
    # Compute and save yearly TPT metrics, window operators, and flux fields.
    if RUN_METRICS:
        main_metrics(paths)
    
    setup = None
    distance_result = None
    
    # Whole-system distance diagnostics use the saved full-window opertors
    if RUN_ANOMALY or RUN_STANDARD_PLOTS:
        setup = analysis_setup(paths)
        distance_result = whole_system_distance(setup, paths)
    
    # Standard geometry, cumulative-rate, mean-travel, and anomaly plots
    if RUN_STANDARD_PLOTS:
        plot_source_target(paths)
        plot_gates(paths)
        plot_clim_cum_k(paths)
        
        mean_travel(paths, week_range = None)
        mean_travel(paths, week_range = late_season_range)
        
        if distance_result is not None:
            plot_whole_system_distance(distance_result, paths)
            plot_source_target_rel_clim(distance_result, paths=paths)
    
    # Release memory before heavier spatial plotting.
    if setup is not None:
        del setup
    gc.collect()
    
    # Effective-current maps and anomalies for selected year/target pairs
    if RUN_FLUX_PLOTS:
        plotted_targets = set()
        
        for year, target in flux_examples:
            plot_flux_map(year, target, paths, week_range = default_week_range)
            plot_flux_map_anomaly(year, target, paths, 
                                  week_range = default_week_range)
            
            plotted_targets.add(target)
            
        for target in sorted(plotted_targets):
            plot_clim_flux_map(target, paths, week_range=default_week_range)
        
        plot_clim_flux_panel(paths, week_range = default_week_range, 
                                 shared_colorbar = True)
            
        plot_clim_flux_panel(paths, week_range = default_week_range, 
                                 shared_colorbar = False)
        
        
    # Source-density diagnostics
    if RUN_SOURCE_DENSITY_PLOTS:
        plot_clim_source_pushforward(paths, week_range = default_week_range)
        plot_clim_source_occ(paths, week_range= default_week_range)
    
    # Heavier diagnostics: Backward trajectories, origin likelihood, and
    # expected arrival-time maps. 
    if RUN_HEAVY_PLOTS:
        #plot_backward_traj(year = backward_traj_year, paths = paths, 
        #                   end_week=backward_end_week, 
        #                   num_horizon_weeks=backward_horizon_weeks)
        
        plot_clim_origin(paths, week_range=default_week_range)
        #plot_clim_expected_arrival(paths)
    
if __name__ == "__main__":
    main()