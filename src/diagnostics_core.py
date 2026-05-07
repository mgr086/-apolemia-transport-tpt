# =============================================================================
#                     Diagnostics core
# =============================================================================
"""
Core routines for finite-time transition path diagnostics.

This module computes TPT metrics, full-window transport operators, 
anomaly diagnostics, and cores for plotting routines. The main inputs are 
yearly sequences of augmented time-inhomogeneous transition matatrices 
produces by "build_titm.py"
"""

import numpy as np
from pathlib import Path

import pandas as pd
from datetime import datetime, timedelta

from pygtm_miron.physical import physical_space
from pytpt_finite_helfmann import tpt as finite_tpt
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import time
import scipy.io as sio


from config import (
    T, K_win, late_week_range, lon, lat, spatial_dis, all_years, clim_years,
    targets, s_minlon, s_maxlon, s_minlat, s_maxlat, DEFAULT_MAX_WORKERS,
    latmin_g1, latmax_g1, lonpos_g1, latmin_g2, latmax_g2, lonpos_g2,
    Pa_dir, DATA_DIR, init_dist_A, init_dist_U, week_step)

from Tbarrier_hallergroup import ( 
    convert_meters_per_second_to_deg_per_day, 
    interpolant_unsteady, integration_dFdt)

# =============================================================================
#                           Geometry helpers
# =============================================================================

def build_target_indices(d0, tar):
    """
    Return physical-bin indices for one configured target region

    Parameters
    ----------
    d0 : physical_space
        Physical-space domain used to locate the target bins.
    tar : str
        Target names. Must be a key in "Targets"

    Returns
    -------
    ind_b : ndarray
        Unique valid physical-bin indices belonging to the target region.

    """
    # B is target position
    B = targets[tar]

    dx = d0.vx[1] - d0.vx[0]    
    dy = d0.vy[1] - d0.vy[0]

    t0_lon, t0_lat = B[0], B[1]     # One target point
    nstrip = B[4]                   # Number of target points
    
    # one step in minus direction
    lons = t0_lon + B[2] * dx * np.arange(nstrip)
    lats = t0_lat + B[3] * dy * np.arange(nstrip)
    
    # Find idx of region    
    ind_b = d0.find_element(lons, lats)
    ind_b = np.unique(ind_b[ind_b >= 0]).astype(int) # Clean
    
    return ind_b

def datenum_from_date(dt: datetime) -> int:
    """
    Convert a Python datetime object to MATLAB datenum convention. 

    Parameters
    ----------
    dt : datetime
        Date or datetime to convert.

    Returns
    -------
    int
        Date represented in MATLAB-compatible day count.

    """
    return datetime.toordinal(dt) - 366


def gate_from_domain(d, lat_min, lat_max, lon_pos):
    """
    Return bin indices on the west and east sides of a vertical pathway gate.

    Parameters
    ----------
    d : physical_space
        Physical-space domain.
    lat_min, lat_max: float
        Latitude bounds of the gate.
    lon_pos : float
        Longitude used to place the gate between neighbouring grid columns.

    Returns
    -------
    west_gate, east_gate : ndarray
        Physical-bin indices on the western and eastern sides of the gate.
    """
    # Physical-bin center
    xB = np.mean(d.coords[d.bins, 0], 1)
    yB = np.mean(d.coords[d.bins, 1], 1)

    # Convert to one-dimensional arrays for masking
    x = np.asarray(xB, dtype=float).ravel()
    y = np.asarray(yB, dtype=float).ravel()
    
    # Unique bin-center longitudes define the grid columns
    ux = np.unique(np.round(x, 10))
    ux.sort()

    # Find the grid-column pair that brackets lon_pos
    j = np.searchsorted(ux, lon_pos, side="right") - 1
    if j < 0 or j + 1 >= len(ux):
        raise ValueError(f"Could not place gate at lon_pos={lon_pos}.")
        
    lon_west = ux[j]
    lon_east = ux[j + 1]
    
    dx = lon_east - lon_west
    tol = 0.49 * dx

    # Restrict the gate to the requested latitude band
    band = (y >= lat_min) & (y <= lat_max)
    
    # Select bins west and east of the gate
    west_mask = band & (np.abs(x - lon_west) <= tol)
    east_mask = band & (np.abs(x - lon_east) <= tol)

    west_gate = np.where(west_mask)[0].astype(int)
    east_gate = np.where(east_mask)[0].astype(int)
    
    return west_gate, east_gate

def get_source_A_og(d0):
    """
    Return source-region indices in the original physical-bin numbering

    Parameters
    ----------
    d0 : physical_space
        Physical-space domain.

    Returns
    -------
    A_og : ndarray
        Unique valid bin indices of source region.

    """
    # Bin centers of the original domain
    x_c = 0.5 * (d0.vx[:-1] + d0.vx[1:])
    y_c = 0.5 * (d0.vy[:-1] + d0.vy[1:])
    Xc, Yc = np.meshgrid(x_c, y_c)

    # Original-domain bin ID for each bin center
    ids0 = d0.find_element(Xc.ravel(), Yc.ravel())

    # Mask for Source A box
    maskA0 = (
        (Xc.ravel() >= s_minlon) & (Xc.ravel() <= s_maxlon) &
        (Yc.ravel() >= s_minlat) & (Yc.ravel() <= s_maxlat))

    # Keep valid original-domain bin IDs only
    A_og = np.unique(ids0[maskA0])
    A_og = A_og[A_og >= 0].astype(int)

    return A_og

# =============================================================================
#                           Yearly TPT metrics
# =============================================================================
def compute_metrics_one_year(year: int, paths) -> dict:
    """
    Compute TPT metrics and effective-current fields for one analysis year. 

    Parameters
    ----------
    year : int
        Analysis year.
    paths : dict
        Path directories.

    Returns
    -------
    dict
        Status directory with runtime information and potential error details.

    """
    t0 = time.time()
    try:
        # OUtput directories
        flux_field_dir = paths["flux_field_dir"]
        metrics_dir = paths["metrics_dir"]
        window_ops_dir = paths["window_ops_dir"]
        
        # Load kernels
        pkl_cur = Pa_dir / f"P_ains_{year}.pkl"
        P_cur = [np.asarray(P, dtype=float) for P in pd.read_pickle(pkl_cur)]
        
        # Need kernelds from previous years
        Ksup_need = K_win - 1
        pkl_prev = Pa_dir / f"P_ains_{year-1}.pkl"
        P_prev_full = [
            np.asarray(P, dtype=float) for P in pd.read_pickle(pkl_prev)]
        P_prev = P_prev_full[-Ksup_need:]
        
        # Complete kernel
        P_ains = P_prev + P_cur
        
        # ---- Create domain (per-process; avoids pickling issues) ----
        d0 = physical_space(lon, lat, spatial_dis)
        
        # Create d_tmin for later
        d_tmin = d0
        
        # Bin centers of physical bins
        xB = np.mean(d_tmin.coords[d_tmin.bins, 0], axis=1)
        yB = np.mean(d_tmin.coords[d_tmin.bins, 1], axis=1)
        N_phys = len(d_tmin.bins)
        
        
        # -------- Source A -----------
        A_og = get_source_A_og(d0)
        ind_a = A_og.copy()
        
        # End week of target year
        week52_end = datetime.fromisocalendar(year, 52, 7)
        
        # Indices of bins of gates
        west_idxg1, east_idxg1 = gate_from_domain(
            d_tmin, latmin_g1, latmax_g1, lonpos_g1)
        west_idx_g2, east_idx_g2 = gate_from_domain(
            d_tmin, latmin_g2, latmax_g2, lonpos_g2)
        
        # Store the full-window operators once per year
        Pfull_hist = []
        weeks_window_hist = []
        window_dates_hist = []
        Pwindow_hist = []
        
        # Loop over every target region
        for tar in targets:
            save_window_ops = (tar ==list(targets.keys())[0])
            
            # Lists for storing values
            Ktot_hist = []
            weeks_lbl_metrics = []
            gate_share_hist=[]
            k_cumnorm_hist = []
            k_cum_hist=[]
            k_steps_hist=[]
            mean_travel_hist = []
            gate1_tot_hist=[]
            gate2_tot_hist=[]
            
            # Find target indices
            ind_b = build_target_indices(d0, tar)
            
            # Sliding windows over the combined sequence
            n_windows = len(P_ains) - K_win + 1
            
            for times in range(n_windows):
                end_dt = week52_end - timedelta(days=times*T)
                
                # ISO week number of this window endpoint
                _,week,_ = end_dt.isocalendar()
                
                
                end = len(P_ains) - times
                start = end - K_win
                P_window_fwd = P_ains[start:end]
                
                if save_window_ops:
                    P_window_save = [
                        np.asarray(Pk, dtype=np.float32) 
                        for Pk in P_window_fwd]
                    
                    Pwindow_hist.append(P_window_save)
                    
                    P_full = np.array(P_window_fwd[0], dtype=float, copy=True)
                    for Pk in P_window_fwd[1:]:
                        P_full = P_full @ Pk
                    
                    Pfull_hist.append(P_full)
                    weeks_window_hist.append(int(week))
                    window_dates_hist.append(np.datetime64(end_dt.date()))
                
                
                # Forward kernel sequence used in finite-time TPT
                P_window_fwd = P_ains[start:end]

                #  ----------------- TPT method ------------------
                K = len(P_window_fwd) # num of kernels in this window
                N = K+1                 # Number of time nodes
                
                # Define P's as a function. Required for finite.tpt
                def P(n: int) -> np.ndarray:
                    return P_window_fwd[n]
                
                # Definin transition-region indices
                S = P(0).shape[0]   # Tot number of states in aug system
                AB = np.union1d(ind_a, ind_b) # Union of source and target ind
                
                # All bins not in A or B
                ind_C = np.setdiff1d(np.arange(S, dtype=int), AB)
                
                # Initial density
                nirvana = S-1   # Index of nirvana
                init_dens = np.zeros(S, dtype=float)
                
                if init_dist_U:
                    # Uniform over physical bins
                    init_dens[:nirvana] = 1.0/float(nirvana) 
                    
                elif init_dist_A:
                    ind_a_phys = np.asarray(ind_a, dtype=int)
                    ind_a_phys = ind_a_phys[(ind_a_phys >= 0)
                                            & (ind_a_phys < nirvana)]
                    init_dens[ind_a_phys] = 1.0 / float(ind_a_phys.size)
                
                init_dens[nirvana] = 0.0 # No initial mass in nirvana
                init_dens /= init_dens.sum()    # Normalize for safety
                
                # Initialize TPT
                tpt = finite_tpt(P, N, ind_a, ind_b, ind_C, init_dens)
                
                tpt.density()                       # Density
                tpt.backward_transitions()          # Time-reversed P
                tpt.forward_committor()             # Committors
                tpt.backward_committor()
                
                # reactive current
                current, eff_current, fx,fy = tpt.reac_current(xB, yB, N_phys)
                
                # Effective current per step
                eff_steps = tpt.eff_current[:tpt.N-1, :, :] # shape (K, S, S)
                
                rate, _ = tpt.transition_rate()     # Transition rates
                k_steps_raw = rate[1:, 1].copy()    # Inflow to B each step
                
                # Convert to "since release"
                k_steps = k_steps_raw[::-1].copy()
                k_steps_hist.append(k_steps)
                
                # Total transition rate into B over window
                k_Btot = float(np.nansum(k_steps))
                
                # Cumulative reactive trasition rate over the window
                k_cum = np.nancumsum(k_steps)
                k_cum_hist.append(k_cum)
                
                # Normailze to [0,1] to compare across weeks/targets
                k_cum_norm = k_cum /k_cum[-1]
                k_cumnorm_hist.append(k_cum_norm)
                
                # Normalization factor of reactive density
                tpt.reac_norm_factor()
                
                # Mean transition time: Mean length in time steps
                t_steps = tpt.mean_transition_length() 
                t_travel = T*t_steps
                mean_travel_hist.append(t_travel)
                
                # Gate flux by step
                gate_flux_by_stepg1 = np.zeros(tpt.N-1, dtype=float)
                gate_flux_by_step_g2 = np.zeros(tpt.N-1, dtype=float)
                
                for n in range(tpt.N-1):
                    gate_flux_by_stepg1[n] = float(
                        eff_steps[n][np.ix_(west_idxg1, east_idxg1)].sum())
                    gate_flux_by_step_g2[n] = float(
                        eff_steps[n][np.ix_(west_idx_g2, east_idx_g2)].sum())
                
                # Stepwise route share between DC and NASC
                sumD = float(np.nansum(gate_flux_by_stepg1))
                sumS = float(np.nansum(gate_flux_by_step_g2))
                den = sumD + sumS
                
                # Window-integrated gate share
                gate_share = (sumD/den) if den>0 else np.nan
                gate1_tot = sumD
                gate2_tot = sumS
                
                # Append to list for later plotting
                Ktot_hist.append(k_Btot)
                weeks_lbl_metrics.append(week)
                gate_share_hist.append(gate_share)
                gate1_tot_hist.append(gate1_tot)
                gate2_tot_hist.append(gate2_tot)
                
                # Time-integrated effective current over the full window
                fx_sum = np.sum(fx, axis=0)
                fy_sum = np.sum(fy, axis=0)
                
                # Convert vector fields to 2D grid fields
                Fx = d_tmin.vector_to_matrix(fx_sum)
                Fy = d_tmin.vector_to_matrix(fy_sum)
                
                save_dir = flux_field_dir / str(year) / tar
                save_dir.mkdir(parents=True, exist_ok=True)
                
                np.savez_compressed(
                    save_dir/ f"fields_W{week:02d}.npz",
                    week=int(week),
                    Fx=Fx.astype(np.float32),
                    Fy=Fy.astype(np.float32),
                    vx=d0.vx.astype(np.float32),
                    vy=d0.vy.astype(np.float32),
                    ind_a=np.asarray(ind_a, dtype=np.int32),
                    ind_b=np.asarray(ind_b, dtype=np.int32),
                    west_idx=np.asarray(west_idxg1, dtype=np.int32),
                    east_idx=np.asarray(east_idxg1, dtype=np.int32),
                    end_date=np.datetime64(end_dt.date()))
        
            # Save arrays for years comparison
            result = {
                "weeks": weeks_lbl_metrics[::-1],                 
                "Ktot": Ktot_hist[::-1], 
                "gate_share": gate_share_hist[::-1],
                "gate1_tot": gate1_tot_hist[::-1],
                "gate2_tot": gate2_tot_hist[::-1],
                "k_cumnorm": k_cumnorm_hist[::-1],
                "mean_travel": mean_travel_hist[::-1],
                "k_cum" : k_cum_hist[::-1],
                "k_steps" : k_steps_hist[::-1]}
            
            # Outdir
            pd.to_pickle(result, metrics_dir/ f"metrics_{tar}{year}.pkl")
    
        # Save full-window operators once per year
        window_obj = {
            "weeks": np.asarray(weeks_window_hist[::-1], dtype=int),
            "end_date": np.asarray(window_dates_hist[::-1]),
            "Pfull": Pfull_hist[::-1],
            "Pwins": Pwindow_hist[::-1]}
        pd.to_pickle(window_obj, window_ops_dir / f"window_ops_{year}.pkl")    
    
        dt = time.time() - t0
        return {'year': year, 'ok': True, 'seconds': dt}
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        return {'year': year, 'ok': False, 'seconds': dt, 
                'error': str(e), 'traceback': tb}

def main_metrics(paths):
    """
    Compute and save source-to-target diagnostics for all analysis years.
    
    The function runs "compute_metrics_one_year" for each year in "all_years",
    using parallel worker processes up to "DEFAULT_MAX_WORKERS". The resulting
    metric files, full-window operators, and effetice-current fields are saved
    through the output directories specified in "paths". 

    Parameters
    ----------
    paths : dict
        Path directory containing output for metrics, flux fields,
        and full-window operators
    """
    # Analysis years are fixed by the project configuration
    years = all_years
    if not years:
        print("[ERROR] No P_ains files found. Run the builder first.")
        return
    
    # Limit the number of workers 
    max_workers = min(DEFAULT_MAX_WORKERS, len(years))
    print(f"[INFO] Running metric computation for {len(years)} years"
          f"with max_workers={max_workers}")
    
    # Store one independent metric computation per analysis year
    results = []
    
    # Run one independent metric computation per analysis year
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(compute_metrics_one_year, year, paths)
                for year in years]
        
        # Report results as soon as each year finished
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get('ok'):
                print(f"[OK] year={r['year']} in {r['seconds']:.1f}s")
            else:
                print(f"[FAIL] year={r['year']} after"
                      f"{r['seconds']:.1f}s: {r.get('error')}")
                print(r.get('traceback', ''))
                
    # Print final success/failure summary
    ok = sum(1 for r in results if r.get('ok'))
    fail = len(results) - ok
    
    print(f'[INFO] Metric computation done. ok={ok}, fail={fail}')

# =============================================================================
#               Window operators and transport distance
# =============================================================================

def compute_pi(setup, ref_years, week):
    """
    Compute a week-specific climatological occupancy weight pi. 

    For each reference year, a uniform density over physical bins is
    propagated through the finite-time window ending at "week". In-domian 
    occupancy is accumulates over all steps and averaged over reference years. 

    Parameters
    ----------
    setup : dict
        Output from "analysis_setup"
    ref_years : list[int]
        Years used to form the climatological reference
    week : int
        End-week label of the transport window.

    Returns
    -------
    pi : np.ndarray
        Week-specific climatological occupancy weight over physical bins.
    """
    N = setup["N"]
    windows_by_year = setup["windows_by_year"]

    occ = np.zeros(N, dtype=float)

    for yref in ref_years:
        week_map = windows_by_year[yref]

        if int(week) not in week_map:
            continue

        P_window_fwd = week_map[int(week)]   # list of K_win forward kernels

        dens = np.zeros(N + 1, dtype=float)
        dens[:N] = 1.0 / N   # uniform over physical bins

        # accumulate occupancy over the whole finite-time transport horizon
        for P in P_window_fwd:
            dens = dens @ P
            dens[N] = 0.0
            occ += dens[:N]

    pi = occ / occ.sum()
    return pi

def load_yearly_window_operators(year, paths):
    """
    Load saved full-window operators for one analysis year.

    Parameters
    ----------
    year : int
        Analysis year.
    paths: dict
        Path directory

    Returns
    -------
    weeks : np.ndarray
        End-week labels for the saved full-window operators.
    Pfull : list[np.ndarray]
        Full multi-step augmented operators for each window.
    Pwins : list[list[ndarray]]
        One-step augmented transition-matrix sequence for each window
    """
    window_ops_dir = paths["window_ops_dir"]
    
    # Build filename
    f = window_ops_dir / f"window_ops_{year}.pkl"

    # Load saved object
    obj = pd.read_pickle(f)

    # Extract weeks and full-window operators
    weeks = np.asarray(obj["weeks"], dtype=int)
    Pfull = [np.asarray(P, dtype=float) for P in obj["Pfull"]]
    Pwins = [[np.asarray(Pk, dtype=float) for Pk in win] for win in obj["Pwins"]]

    # Safety: ensure matching lengths
    n = min(len(weeks), len(Pfull), len(Pwins))

    return weeks[:n], Pfull[:n], Pwins[:n]


def compute_Pmean(ref_years, paths):
    """
    Compute climatological mean full-window operators

    Parameters
    ----------
    ref_years : list[int]
        Years included in the climatology.
    paths : dict
        Path directory.

    Returns
    -------
    Pmean_win : list[ndarray]
        Mean full-window augmented operators.
    common_weeks : ndarray
        End-week labels associated with "Pmean_win".

    """
    ops_by_year = {}
    weeks_sets = []

    for yref in ref_years:
        weeks_y, Pfull_y,_ = load_yearly_window_operators(yref, paths)

        # Map each saved end week to its corresponding full-window operator
        ops_by_year[yref] = {int(w): P for w, P in zip(weeks_y, Pfull_y)}
        weeks_sets.append(set(int(w) for w in weeks_y))

    # Keep only week labels that exist in every reference year
    common_weeks = np.array(sorted(set.intersection(*weeks_sets)), dtype=int)

    Pmean_win = []
    for w in common_weeks:
        # Average the full-window operator for the same end week across years
        mats_w = [ops_by_year[yref][int(w)] for yref in ref_years]
        Pmean_win.append(np.mean(np.stack(mats_w, axis=0), axis=0))

    return Pmean_win, common_weeks

def whole_system_distance(setup, paths):
    """
    Compute whole-system transport distances from full-window operators.
    
    For each analysis year and aligned end-week, the saved full-window operator
    is compared with the corresponding climatological mean full-window operator
    using "two_part_dist"
    
    Parameters
    ----------
    setup : dict
        Setup dictionary returned by "analysis_setup.
    paths : dict
        Path directory contatining "window_ops_dir".

    Returns
    -------
    out : dict
        Directory contatining weekly distance components, 
        yearly mean distances, late-season mean distances, valid week labels,
        and the configured window length.

    """
    # Load setup information
    ny = setup["ny"]
    Kseg = setup["Kseg"]
    week_axis = setup["week_axis"]
    late_week_range = setup["late_week_range"]

    # Storage for stepwise distances
    step_tot = np.full((ny, Kseg), np.nan, dtype=float)
    step_out = np.full((ny, Kseg), np.nan, dtype=float)
    step_route = np.full((ny, Kseg), np.nan, dtype=float)

    # Ensure the plotting axis is a NumPy integer array
    week_axis = np.asarray(week_axis, dtype=int)
    week_to_idx = {int(w): j for j, w in enumerate(week_axis)}
    
    ref_years = clim_years[:]
    
    # Compute climatological mean full-window operators ONCE
    Pmean_win, weeks_ref = compute_Pmean(ref_years, paths)
    map_ref = {int(w): P for w, P in zip(weeks_ref, Pmean_win)}
    
    pi_by_week = {}
    for week in map_ref.keys():
        pi_by_week[int(week)] = compute_pi(setup, ref_years, int(week))
    
    
    # Loop over all analysis years
    for i, y in enumerate(all_years):

        # Load saved full-window operators for the analysis year
        weeks_y, Pfull_y,_ = load_yearly_window_operators(y, paths)

        # Map week label -> operator for year y and climatology
        map_y = {int(w): P for w, P in zip(weeks_y, Pfull_y)}

        # Only compare weeks available in both year y and the climatology
        common_weeks = sorted(set(map_y.keys()) & set(map_ref.keys()))

        for week in common_weeks:
            # Uniform weight over physical bins.
            pi_ref = pi_by_week[int(week)]
            
            Pwin_y = map_y[week]
            Pwin_ref = map_ref[week]

            dT, dO, dR = two_part_dist(Pwin_y, Pwin_ref, pi_ref)
            
            j = week_to_idx.get(int(week))
            step_tot[i, j] = dT
            step_out[i, j] = dO
            step_route[i, j] = dR

    # Full-year average over valid saved weeks only
    valid_mask = np.any(np.isfinite(step_tot), axis=0)  
    d_tot = np.nanmean(step_tot[:, valid_mask], axis=1)

    # Late-season mask
    week_min, week_max = late_week_range
    late_mask = (week_axis >= week_min) & (week_axis <= week_max)

    # Keep only late-season weeks that also have saved full-window operators
    late_mask_valid = late_mask & valid_mask
    d_tot_late = np.nanmean(step_tot[:, late_mask_valid], axis=1)

    # Package output
    out = {
        "step_tot": step_tot,
        "step_out": step_out,
        "step_route": step_route,
        "d_tot": d_tot,
        "d_tot_late": d_tot_late,
        "late_mask": late_mask_valid,
        "valid_weeks": week_axis[valid_mask],
        "window_len": K_win
    }

    return out

def two_part_dist(P1, P2, pi):
    """
    Compute a two-part distance between augmented transition matrices.
    
    The distance has one contribution from differences in outflow probability
    and one contribution from differences in conditional in-domain routing. 

    Parameters
    ----------
    P1, P2 : ndarray
        Augmented transition operators with the absorbing out-of-domain state
        in the final row and column. 
    pi : ndarray
        Climatological occupancy weight

    Returns
    -------
    d_total: float
        Sum of the outflow and routing distances
    d_out: float
        Weighted outflow-distance component
    d_route: float
        Weighted in-domain routing-distance component
    """
    
    N = P1.shape[0]-1   # Number of physical bins
    
    Po1 = P1[:N, :N]    # Open domain matrix, excluding nirvana
    Po2 = P2[:N, :N]

    Pw1 = P1[:N, -1]    # Outflow to nirvana
    Pw2 = P2[:N, -1]
    
    s1 = 1.0 - Pw1      # Survival in the open domain over one step
    s2 = 1.0 - Pw2
    
    # Outflow distance
    d_out = float(np.sum(pi*np.abs(Pw1-Pw2)))
    
    # Condition on staying in domain
    good = (s1 > 1e-12) & (s2 > 1e-12)
    Ph1 = np.zeros_like(Po1, dtype=float)
    Ph2 = np.zeros_like(Po2, dtype=float)
    Ph1[good, :] = Po1[good, :] / s1[good, None]
    Ph2[good, :] = Po2[good, :] / s2[good, None]
    
    # Row-wise L1 difference 
    row_l1 = np.sum(np.abs(Ph1 - Ph2), axis = 1)
    
    # Downweight rows where either year has small survival
    m = np.minimum(s1, s2)
    
    # Weigted routing distance
    d_route = float(np.sum(pi*m*row_l1))
    
    # Combined distance
    d_total = d_out + d_route
    
    return d_total, d_out, d_route

def analysis_setup(paths):
    """
    Build shared analysis metadata for the anomaly diagnostics

    Parameters
    ----------
    paths : dict
        Path directory containing "window_ops_dir".

    Returns
    -------
    out : dict
        Setup dictionary with the physical dimension, source indices,
        common week axis, saved TM per year, and the configured window length.

    """
    
    # Reconstruct A_og
    d0_ref = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0_ref)

    windows_by_year = {}

    ny = len(all_years)
    N = None
    week_axis = None

    for y in all_years:
        weeks_y, Pfull_y, Pwins_y = load_yearly_window_operators(y, paths)

        if len(weeks_y) == 0:
            raise ValueError(f"No saved window operators found for year {y}.")

        # Store week -> transition-window mapping
        windows_by_year[y] = {
            int(w): win for w, win in zip(weeks_y, Pwins_y)}

        # Infer N once from the saved full-window operator
        if N is None:
            N = Pfull_y[0].shape[0] - 1

        # Use the first year's saved weeks as the common plotting axis
        if week_axis is None:
            week_axis = np.asarray(weeks_y, dtype=int)

    out = {
        "ny": ny,
        "Kseg": len(week_axis),
        "late_week_range": late_week_range,
        "N": N,
        "A_og": A_og,
        "week_axis": week_axis,
        "windows_by_year": windows_by_year,
        "window_len": K_win}

    return out

def leave_one_out_mean(values, years):
    """
    Compute a year-specific leve-one-out climatological reference mean. 
    
    For years included in "clim_years", the reference exclued the year itself. 
    For years outside "clim_years", the reference is the mean over all 
    climatology years. 

    Parameters
    ----------
    values : array-like
        One yearly value per analysis year.
    years : array-like of int
        Years corresponding to "values".

    Returns
    -------
    ref: no.ndarray.
        Year-specific referance means. 
    """
    # Safety
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=int)
    
    # Array for storing
    ref = np.full_like(values, np.nan, dtype=float)
    
    # Loop over each year and compute its own leave-one-out reference mean
    for i,y in enumerate(years):
        # Leave-one-out scheme
        if y in clim_years:
            ref_mask = np.isin(years, [yy for yy in clim_years if yy != y])
        else:
            ref_mask = np.isin(years, clim_years)
    
        # Compute mean if reference set is not empty
        if np.any(ref_mask):
            ref[i] = np.nanmean(values[ref_mask])
    
    return ref

# =============================================================================
#                 Source-to-target summary diagnostics
# =============================================================================

def gate_share(g1, g2, week_range=None):
    """
    Compute the window-integrated share of flux passing through gate 1

    Parameters
    ----------
    g1, g2 : array-like
        Gate-integrated flux values for gate 1 and gate 2.
    week_range : tople[int, int], optional
        Inclusive week interval used to restrict calc. The default is None.

    Returns
    -------
    val : float
        "sum(g1) / (sum(g1) + sum(g2))" over the selcted weeks.

    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    
    weeks = week_step * np.arange(1, len(g1) + 1, dtype=int)
    
    # Keep only entries where both gate fluxes and the week label are finite
    mask = np.isfinite(g1) & np.isfinite(g2) & np.isfinite(weeks)
        
    # Optional week restrriction
    if week_range is not None:
        wmin, wmax = week_range
        mask &= (weeks >= wmin) & (weeks <= wmax)
    
    g1_use = g1[mask]
    g2_use = g2[mask]
    
    # Compute the window-integrated share of transport crossing gate 1
    denom = np.nansum(g1_use) + np.nansum(g2_use)
    val = np.nansum(g1_use) / denom
    
    return val

def load_year_target_metric(tar, year, metric, paths):
    """
    Load one saved metric time series for a target and analysis year

    Parameters
    ----------
    tar : str
        Target name, e.g. 'Bergen'
    year : int
        Analysis year
    metric : str
        Metric key in "metrics_{tar}{year}.pkl"
    paths : dict
        Path directory

    Returns
    -------
    arr : ndarray
        Metric values truncated to match the avilable week labels
    
    weeks : ndarray
        stored end-week labels truncated to match "arr"
    """
    metrics_dir = paths["metrics_dir"]
    
    # Build the filename
    f = metrics_dir / f"metrics_{tar}{year}.pkl"
    
    # Load the saved metrics
    obj = pd.read_pickle(f)
    
    # Extract the requested metric
    arr = np.asarray(obj[metric], dtype=float)
    
    # Used stored week labels
    weeks = np.asarray(obj["weeks"], dtype=int)
        
    # Ensure arr and weeks have matching length in case of inconsistency
    n = min(len(arr), len(weeks))
    
    # Return truncated arrays with consistend length
    return arr[:n], weeks[:n]

def sum_target_metric(x, week_range = None):
    """
    Sum a matric time series over all weeks or a selected week interval. 
    
    Parameters
    ----------
    x : array-like
        One-dimensional metric time series.
    week_range : tuple(int, int), optional
        Inclusive week interval (wmin, wmax). If given, only values with
        weeks in this range are used.

    Returns
    -------
    x_sum : float
        Sum of the selcted metric values
    """
    
    # Convert metric to Numpy Array
    x = np.asarray(x, dtype=float)
    
    weeks = week_step * np.arange(1, len(x) +1, dtype=int)
    
    # Keep only finite values
    mask = np.isfinite(x) & np.isfinite(weeks)
    
    # Restrict to a chosen week interval
    if week_range is not None:
        wmin, wmax = week_range
        mask &= (weeks >= wmin) & (weeks <= wmax)
    
    # Apply the mask 
    x_use = x[mask]

    x_sum = np.nansum(x_use)
    
    # Return summary statistics
    return x_sum

def add_source_target_diagnostics(summary_df, paths):
    """
    Add yearly source-to-target diagnostic summaries to a dataframe
    
    For each target and analysis year, this function computes the full-year and
    late-season summaries of the total transition rate and gate share. It also
    computes relative deviations from the climatological mean. 
    
    Parameters
    ----------
    summary_df : pandas.DataFrame
        Dataframe containing one row per analysis year.
    paths : dict
        Path directory contatining "metrics_dir"

    Returns
    -------
    summary_df : pandas.DataFrame
        The input dataframe with added diagnostic summary and climatological 
        anomaly columns
    """
    metric_keys = ["Ktot", "gate_share"]
    
    new_cols = {}           # Dict for storing
    years_arr = np.array(all_years)
    
    # Loop over each target location
    for tar in targets:
        
        # Loop over each transport metric to summarize
        for metric in metric_keys:
            
            full_vals = []  # Lists for full-year summaries
            late_vals = []  # Lists for late-season summaries
            
            # Loop over the analysis year
            for y in years_arr:
                
                if metric == "Ktot":
                
                    # Load the stored metric for this targetn and year
                    arr, weeks = load_year_target_metric(
                        tar, int(y), "Ktot", paths)
    
                    # Compute full-year summary
                    val_full = sum_target_metric(arr, week_range=None)
    
                    # Compute late-season summary
                    val_late = sum_target_metric(
                        arr, week_range = late_week_range)
                    
                elif metric =="gate_share":
                    # Load the stored metric for this target and year
                    g1, weeks1 = load_year_target_metric(
                        tar, int(y), "gate1_tot", paths)
                    g2, weeks2 = load_year_target_metric(
                        tar, int(y), "gate2_tot", paths)
                    
                    n = min(len(g1), len(g2), len(weeks1), len(weeks2))
                    g1 = g1[:n]
                    g2 = g2[:n]
                    
                    val_full = gate_share(g1, g2,week_range=None)
                    val_late = gate_share(g1, g2, week_range=late_week_range)
                    
                    
                # Store the full-year summaries for this year
                full_vals.append(val_full)
                    
                # Store the late-season summaries for this year
                late_vals.append(val_late)
            
            # Store the completed summary columns for this target/metric
            full_arr = np.asarray(full_vals, dtype=float)
            late_arr = np.asarray(late_vals, dtype=float)
        
            new_cols[f"{metric}_{tar}"] = full_arr
            new_cols[f"{metric}_late_{tar}"] = late_arr
            
            # ----- Relative deviation from climatological mean -----
            eps = 1e-12
            
            clim_mask = np.isin(years_arr, clim_years)
            
            # Compute climatological mean
            #ref_full = leave_one_out_mean(full_arr, years_arr)
            #ref_late = leave_one_out_mean(late_arr, years_arr)
            ref_full = np.nanmean(full_arr[clim_mask])
            ref_late = np.nanmean(late_arr[clim_mask])
            
            # Compute relative anomaly
            rel_full = (full_arr-ref_full) / (np.maximum(
                np.abs(ref_full), eps))
            rel_late = (late_arr-ref_late) / (np.maximum(
                np.abs(ref_late), eps))
            
                
            # Store the anomalies 
            new_cols[f"{metric}_clim_{tar}"] = rel_full
        
            new_cols[f"{metric}_clim_late_{tar}"] = rel_late
                
    summary_df = pd.concat( [summary_df, 
            pd.DataFrame(new_cols, index=summary_df.index)],axis=1)

    # Return the dataframe with the new summary columns included
    return summary_df

def mean_travel(paths, week_range=None):
    """
    Export average mean transition time for each target.
    
    The average is computes over all available analysis years and selected
    end weeks. Results are written to "mean_travel_.csv" for the full year or
    "mean_travel_late_season.csv" for a restricted week_range

    Parameters
    ----------
    paths : dict
        Path directory containing "outdir" and "metrics_dir"
    week_range : tuple(int, int) or None, optional
        Inclusive week interval. 
    """
    outdir = paths["outdir"]
    metrics_dir = paths["metrics_dir"]
    suffix =""
    
    # Output dir
    csv_dir = outdir
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []   # Storage

    # Loop through each target region
    for tar in targets:
        values = []         # Storage
        
        # Loop through all years
        for year in all_years:
            
            # Load saved metrics file
            f = metrics_dir / f"metrics_{tar}{year}.pkl"
            if not f.exists():
                print(f"[export_mean_travel_target_average_csv]"
                      f" Missing file: {f}")
                continue

            obj = pd.read_pickle(f)
            
            # Extract mean travel time
            mean_travel = np.asarray(obj.get("mean_travel", []), dtype=float)
            
            # Extract the corresponding end-week numbers
            weeks = np.asarray(obj.get("weeks", []), dtype=int)

            # Make sure arrays have matching length
            n = min(len(mean_travel), len(weeks))
            mean_travel = mean_travel[:n]
            weeks = weeks[:n]

            # Filter by week range if requested
            if week_range is not None:
                wmin, wmax = week_range
                mask = (weeks >= wmin) & (weeks <= wmax)
                mean_travel = mean_travel[mask]
                suffix = "late_season"

            # Keep only finite values
            mean_travel = mean_travel[np.isfinite(mean_travel)]
            if mean_travel.size == 0:
                continue
            
            # Add remaining mean travel values from this year to target list
            values.extend(mean_travel.tolist())
        
        # Compute average mean travel time for this target
        avg_val = float(np.mean(values)) if len(values) > 0 else np.nan
        
        # Append one output row for this target
        rows.append({
            "target": tar,
            "mean_travel": avg_val})
        
    # Convert list of rows to pandas DataFrame
    df = pd.DataFrame(rows)
    
    # Write to csv
    save_path = csv_dir / f"mean_travel_{suffix}.csv"
    df.to_csv(save_path, index=False)

    return df

# =============================================================================
#                       Effective-current fields
# =============================================================================
def mean_flux(year, tar, paths, week_range=None):
    """
    Compute the mean integrated effective-current field for one year 
    and target. 

    Parameters
    ----------
    year : int
        Analysis year.
    tar : str
        Target name. 
    paths : dict
        Path directory contatining "flux_field_dir"
    week_range : tuple[int, int], optional
        End-weeks included in the average. If None, the late-season is used

    Returns
    -------
    Fx_mean, Fy_mean : ndarray
        Mean zonal and meridional flux field over the selected weeks
    vx, vy : ndarray
        Grid coordinates associated with the flux field

    """
    flux_field_dir = paths["flux_field_dir"]
    
    #Lists for storing weekly flux components
    Fx_list = []
    Fy_list = []
    
    # Grid coordinates
    vx = None
    vy = None
    
    # Use late season by default
    if week_range is None:
        week_min, week_max = late_week_range
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
        week_min, week_max = int(week_range[0]), int(week_range[1])
    
    # Selected end-weeks, inclusive of both endpoints.
    weeks = range(week_min, week_max + 1, week_step)

    # Loop over the selected weeks
    for w in weeks:
        # Load the weekly field file for this year, target and week
        f = Path(flux_field_dir) / str(year) / tar / f"fields_W{int(w):02d}.npz"
        dat = np.load(f, allow_pickle=True)
    
        
        # Store the weekly flux components
        Fx_list.append(np.asarray(dat["Fx"], dtype=float))
        Fy_list.append(np.asarray(dat["Fy"], dtype=float))
        
        # Read the grid coordinates from the first sucessfully loaded file
        if vx is None:
            vx = np.asarray(dat["vx"], dtype=float)
            vy = np.asarray(dat["vy"], dtype=float)

    
    # Stack weekly fields and compute the mean across weeks
    Fx_mean = np.nanmean(np.stack(Fx_list, axis=0), axis=0)
    Fy_mean = np.nanmean(np.stack(Fy_list, axis=0), axis=0)

    # Return the mean flux field and the corresponding grid coordinates
    return Fx_mean, Fy_mean, vx, vy


def climatology_flux(tar, paths, week_range=None):
    """
    Compute the climatologicam mean effective-current field for one target. 
    
    Parameters
    ----------
    tar : str
        Target name. 
    paths : dict
        Paths directory containing "flux_field_dir"
    weeks_use : tupe(int, int), optional
        End-weeks included for each climatology year. If None late-season is
        used

    Returns
    -------
    Fx_clim, Fy_clim : ndarray
        Climatological mean zonal and meridional effective-current fields
    vx, vy : ndarray
        Grid coordinates associated with the flux field
    """
    
    # Lists for stroing yearly mean flux fields
    Fx_years = []
    Fy_years = []
    
    # Grid coordinates
    vx = None
    vy = None
    
    # Loop over the climatology years
    for y in clim_years:
        # Compute the mean flux field for this year over the selcted weeks
        Fx_y, Fy_y, vx_y, vy_y = mean_flux(
            y, tar, paths, week_range = week_range)
        
        
        # Store the yearly mean flux fields
        Fx_years.append(Fx_y)
        Fy_years.append(Fy_y)
        
        # Read the grid coordinates form the first valid yearly field
        if vx is None:
            vx = vx_y
            vy = vy_y
    
    # Stack yearly mean fields and compute the climatological mean
    Fx_clim = np.nanmean(np.stack(Fx_years, axis=0), axis=0)
    Fy_clim = np.nanmean(np.stack(Fy_years, axis=0), axis=0)
    
    return Fx_clim, Fy_clim, vx, vy


# =============================================================================
#                     Arrival-time diagnostics
# =============================================================================
def expected_arrival(P_seq, ind_b):
    """
    Compute conditional first-arrival time to a target over one finite window.
    
    For each physical starting bin, the distribution is propagated through 
    "P_seq". 

    Parameters
    ----------
    P_seq : list[ndarray]
        Sequence of augmented transition matrices over the finite-time window.
    ind_b : array-like
        Target-bin indices in the physical state space.

    Returns
    -------
    tau_days : ndarray
        Expected first-arrival time in days for each physical starting bin.
        Shape (N_phys,).
    hit_prob : ndarray
        Probability of reaching the target within the window from each physical
        starting bin. Shape (N_phys,).
    """
    P_seq = [np.asarray(P, dtype=float) for P in P_seq]
    S = P_seq[0].shape[0]       
    N_phys = S - 1                  # last state = nirvana
    min_hit_prob = 1e-8
    
    # Boolean mask for extraction of target probability mass
    ind_b = np.asarray(ind_b, dtype=int)
    target_mask = np.zeros(S, dtype=bool)
    target_mask[ind_b] = True

    tau_days = np.full(N_phys, np.nan, dtype=float)
    hit_prob = np.zeros(N_phys, dtype=float)
    
    # Loop over each physical bin
    for i in range(N_phys):
        # Start from one physical bin
        p = np.zeros(S, dtype=float)
        p[i] = 1.0

        num = 0.0
        den = 0.0
        
        # Propagate the distribution through the finite-time window
        for step, P in enumerate(P_seq, start=1):
            # propagate one step
            p = p @ P

            # newly arrived mass in target at this step
            arrived = float(np.sum(p[target_mask]))

            if arrived > 0.0:
                t_days = step * T           # Arrival time in physical days
                num += t_days * arrived
                den += arrived

                # remove target mass so later 
                # arrivals correspond to first arrival
                p[target_mask] = 0.0
                
        hit_prob[i] = den
        
        # Conditional mean first-arrival time
        if den >= min_hit_prob:
            tau_days[i] = num / den

    return tau_days, hit_prob


def clim_expected_arrival(tar, paths):
    """
    Compute climatological conditional expected arrival time for one target.
    
    The function averages late-season expected-arrival maps over all analysis
    years and finite-time windows. 
    
    Parameters
    ----------
    tar : str
        Target name.
    paths : dict
        Path directory containing "windows_ops_dir".

    Returns
    -------
    tau_clim : ndarray
        Hit-probability-weighted climatological expected arrival time in days.
    den : ndarray
        Accumulated hit-probability weights.
    d0 : physical_space
        Physical-space domain.
    ind_b : ndarray
        Target-bin indices.
    """
    
    # Build physical space and target 
    d0 = physical_space(lon, lat, spatial_dis)
    ind_b = build_target_indices(d0, tar)
    min_total_hit = 1e-6    # Min. acc. hit probability required
    
    tau_list = []       # Storage
    hit_list = []
    
    # Loop over all analysis years and saved finite-time windows
    for year in all_years:
        weeks_y, _, Pwins_y = load_yearly_window_operators(year, paths)
        
        for week, P_window_fwd in zip(weeks_y, Pwins_y):
            # Keep only late-season windows.
            wmin, wmax = late_week_range
            if not (wmin <= int(week) <= wmax):
                continue
            
            # Compute expected first-arrival time and target-hit prob. 
            tau_days, hit_prob = expected_arrival(
                P_seq=P_window_fwd, ind_b=ind_b)

            tau_list.append(np.asarray(tau_days, dtype=float))
            hit_list.append(np.asarray(hit_prob, dtype=float))

    num = np.zeros_like(np.asarray(tau_list[0], dtype=float))
    den = np.zeros_like(num)
    
    # Average expected arrival times using hit probability as the weight. 
    for tau, hit in zip(tau_list, hit_list):
        valid = np.isfinite(tau) & np.isfinite(hit) & (hit > 0.0)
        num[valid] += tau[valid] * hit[valid]
        den[valid] += hit[valid]
    
    # Compute the conditional climatological expected arrival time
    tau_clim = np.full_like(num, np.nan, dtype=float)
    keep = den >= min_total_hit
    tau_clim[keep] = num[keep]/den[keep]
    

    return tau_clim, den, d0, ind_b


# =============================================================================
#                   source push-forward and occupancy
# =============================================================================
def source_pushforward(year, end_week, paths):
    """
    Push a uniform source distribution forward to a selected end-week.

    The initial distribution is placed uniformly on the source region
    at the beginning of the transport window. The distribution is then
    pushed forward through the sequence of transition matrices ending
    at ``end_week``.

    Parameters
    ----------
    year : int
        Analysis year.
    end_week : int
        ISO end-week of the transport window.
    paths : dict
        Path directory containing "windows_ops_dir"

    Returns
    -------
    rho_end : ndarray
        Final in-domain density on physical bins
    d0 : physical_space
        Physical-space domain.
    A_og : ndarray
        Source-bin indices.
    """
    
    # Build space
    d0 = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0)
    N_og = len(d0.bins)   
    
    # Uniform initial density on source
    rho = np.zeros(N_og, dtype=float)
    rho[A_og] = 1.0 / len(A_og)
    
    # Load transition matrices
    weeks_y, _, Pwins_y = load_yearly_window_operators(year, paths)
    
    # Find the requesteed end-week
    matches = np.where(weeks_y == int(end_week))[0]

    j = int(matches[0])
    P_window_fwd = Pwins_y[j]
    
    # Augmented dimension
    S = P_window_fwd[0].shape[0]
    nirvana = S - 1
    
    # Uniform initial density on source
    rho = np.zeros(S, dtype=float)
    rho[A_og] = 1.0 / len(A_og)
    rho[nirvana] = 0.0

    # Push forward through the saved finite-time horizon window
    for P in P_window_fwd:
        rho = rho @ P

    # Keep only physical bins
    rho_end = np.asarray(rho[:N_og], dtype=float)

    return rho_end, d0, A_og


def clim_source_pushforward(paths, week_range=None):
    """
    Compute climatological final source push-forward density.

    A uniform source distribution is pushed through every selected climatology
    window, and the final in-domain densities are averaged. 

    Parameters
    ----------
    paths : dict
        Path directory containing "window_ops_dir".
    week_range : tuple(int, int), optional
        End-weeks to include. If None, the late-season inverval is used

    Returns
    -------
    rho_clim : ndarray
        Mean final in-domain source density.
    d0_ref : physical_space
        Physical-space domain.
    A_ref : ndarray
        Source-bin indices.
    """
    ref_years = clim_years
    
    # Use late season by default
    if week_range is None:
        week_min, week_max = late_week_range
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
            
        week_min, week_max = int(week_range[0]), int(week_range[1])
    
    # Selected end-weeks, inclusive of both endpoints.
    weeks = range(week_min, week_max + 1, week_step)
    
    rho_list = []

    d0_ref = None
    A_ref = None

    for year in ref_years:
        for week in weeks:
            
            # Push forward
            rho_end, d0, A_og = source_pushforward(year=year, end_week=week,
                paths=paths)
            
            # Store final in-domain density
            rho_list.append(rho_end)
            d0_ref = d0
            A_ref = A_og
        
    # Stack all final densities
    rho_clim = np.nanmean(np.stack(rho_list, axis=0), axis=0)

    return rho_clim, d0_ref, A_ref

def source_cum_occ(year, end_week, paths):
    """
    Compute cumulative source occupancy over one transport window.

    A uniform source density is propagated through the selected transport 
    window. After each step, the in-domain density is accumalted and
    multiplied by "T" to obtain occupancy in particle-days. 
    
    Parameters
    ----------
    year : int
        Analysis year.
    end_week : int
        ISO end-week of the transport window.
    paths : dict
        Path dictionary containing ``window_ops_dir``.

    Returns
    -------
    occ_days : ndarray
        Cumulative occupancy over physical bins, multiplied by ``T``.
        Units are particle-days.
    d0 : physical_space
        Physical-space domain.
    A_og : ndarray
        Source-bin indices.
    """

    # Build physical domain and source indices
    d0 = physical_space(lon, lat, spatial_dis)
    A_og = get_source_A_og(d0)
    N_og = len(d0.bins)

    # Load saved transition-window operators
    weeks_y, _, Pwins_y = load_yearly_window_operators(year, paths)

    # Find requested end-week
    matches = np.where(weeks_y == int(end_week))[0]

    j = int(matches[0])
    P_window_fwd = Pwins_y[j]

    # Augmented state dimension
    S = P_window_fwd[0].shape[0]
    nirvana = S - 1

    # Initial density: uniform over source bins
    rho = np.zeros(S, dtype=float)
    rho[A_og] = 1.0 / float(len(A_og))
    rho[nirvana] = 0.0

    # Accumulated in-domain occupancy
    occ = np.zeros(N_og, dtype=float)

    # Push density forward and accumulate physical occupancy
    for P in P_window_fwd:
        rho = rho @ P

        # Physical in-domain density after this step
        rho_phys = np.asarray(rho[:N_og], dtype=float)

        occ += rho_phys

        # Keep nirvana from re-entering the calculation
        rho[nirvana] = 0.0

    # Convert sum over biweekly steps to particle-days
    occ_days = T * occ

    return occ_days, d0, A_og


def clim_source_cum_occ(paths, week_range=None):
    """
    Compute climatological cumulative source occupancy.

    Cumulative source-occupancy fields are computed for all selected
    climatology windows and then averaged. 

    Parameters
    ----------
    paths : dict
        Path dictionary containing ``window_ops_dir``.
    week_range : tuple(int, int) or None, optional
        End-weeks to include. If None, late-season week range is used

    Returns
    -------
    occ_clim : ndarray
        Mean cumulative source occupancy over physical bins.
    d0_ref : physical_space
        Physical-space domain.
    A_ref : ndarray
        Source-bin indices.
    """

    ref_years = clim_years
    
    # Use late season by default
    if week_range is None:
        week_min, week_max = late_week_range
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
            
        week_min, week_max = int(week_range[0]), int(week_range[1])
    
    # Selected end-weeks, inclusive of both endpoints.
    weeks = range(week_min, week_max + 1, week_step)

    occ_list = []

    d0_ref = None
    A_ref = None
    
    # Loop over climatology years and selected end-weeks. 
    for year in ref_years:
        for week in weeks:
            occ, d0, A_og = source_cum_occ(
                year=year,
                end_week=week,
                paths=paths)

            occ_list.append(occ)
            d0_ref = d0
            A_ref = A_og
        
    # Average over all selected year/window combinations.
    occ_clim = np.nanmean(np.stack(occ_list, axis=0), axis=0)

    return occ_clim, d0_ref, A_ref

# =============================================================================
#                   Origin and trajectory diagnostics
# =============================================================================
def backward_traj_from_target(year, tar, end_week, num_horizon_weeks,
                              n_per_bin=25, dt_days=0.05):
    """
    Integrate synthetic particles backward from one target region.
    
    Particles are released in the target bins at the sunday of "end_week" and 
    integrated backward for "K_win_plot" weeks using the gridded
    velocity field. 
    
    Parameters
    ----------
    year : int
        Analysis year. The function expects "NAO{year}.mat" in "DATA_DIR".
    tar : str
        Target name. Must be a key in "targets".
    end_week : int
        ISO week number whose Sunday is used as the trajectory launch time.
    num_horizon_weeks : int
        Backward integration horizon in weeks. Default is 36.
    n_per_bin : int, optional
        Number of particles released in each target bin. Default is 25.
    dt_days : float, optional
        Time step in days used for the backward integration. Default is 0.05.
    
    Returns
    -------
    d0 : physical_space
        Physical-space domain.
    ind_b : ndarray
        Indices of the target bins.
    x_launch : ndarray, shape (2, N)
        Particle positions at launch time.
    x_final : ndarray, shape (2, N)
        Particle positions at the end of the backward integration.
    traj : ndarray, shape (Nt, 2, N)
        Full trajectory array. The first axis is time, the second contains
        longitude and latitude, and the third indexes particles.
    time_traj : ndarray
        Time array used in the backward integration.
    """
    
    # --------- Domain and target bins -----------
    d0 = physical_space(lon, lat, spatial_dis)
    
    # Build target
    ind_b = build_target_indices(d0, tar)
    
    # -------- Sample release points inside target bins -------
    rng = np.random.default_rng(1234)

    dx = d0.vx[1] - d0.vx[0]
    dy = d0.vy[1] - d0.vy[0]

    pts = []

    for b in ind_b:
        # bin center
        xc = np.mean(d0.coords[d0.bins[b], 0])
        yc = np.mean(d0.coords[d0.bins[b], 1])

        # random jitter inside the bin
        jx = (rng.random(n_per_bin) - 0.5) * 0.9 * dx
        jy = (rng.random(n_per_bin) - 0.5) * 0.9 * dy

        x = xc + jx
        y = yc + jy

        pts.append(np.vstack([x, y]))
    
    # Stack all release points into one array of shape (2, N)
    x0 = np.hstack(pts)

    # ------- loead yearly velocity file -----------
    mat_path = DATA_DIR / f"NAO{year}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {mat_path}")

    mat_file = sio.loadmat(mat_path)

    U_ms = mat_file["u"].astype(np.float32, copy=False)
    V_ms = mat_file["v"].astype(np.float32, copy=False)
    x = mat_file["x"].astype(np.float32, copy=False)
    y = mat_file["y"].astype(np.float32, copy=False)
    t_sec = mat_file["t"]

    del mat_file
    
    # Build the velocity field grid
    X, Y = np.meshgrid(x, y)
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)

    # Convert velocities from m/s to degrees/day
    U, V = convert_meters_per_second_to_deg_per_day(X, Y, U_ms, V_ms)
    del U_ms, V_ms
    
    # Replace NaNs with zero so the interpolants can built saftely
    U[np.isnan(U)] = 0.0
    V[np.isnan(V)] = 0.0

    # Build unsteady interpolants 
    interpolant = interpolant_unsteady(X, Y, U, V, method="cubic")
    Interpolant_u = interpolant[0]
    Interpolant_v = interpolant[1]

    del U, V, interpolant

    # Convert model time from seconds since 1970 to day format
    t_days_since_1970 = t_sec / 86400.0
    matlab_1970 = datenum_from_date(datetime(1970, 1, 1))
    t_data = t_days_since_1970 + matlab_1970

    # -------- Backward launch time and horizon ---------    
    # Launch date = Sunday of selected ISO week
    launch_date = datetime.fromisocalendar(year, end_week, 7)
    t_launch = datenum_from_date(launch_date)

    # Backward integration horizon
    days_back = 7 * num_horizon_weeks
    t_start = t_launch - days_back

    # descending time array for backward integration
    time_traj = np.arange(t_launch, t_start - dt_days, -dt_days)
    
    # ------- Integrate backward -------------
    traj = np.zeros((len(time_traj), 2, x0.shape[1]), dtype=float)
    traj[0] = x0
    
    periodic = [False, False, False]
    x_cur = x0.copy()
    for k in range(1, len(time_traj)):
        # Integrate one time step backward
        t_pair = np.array([time_traj[k - 1], time_traj[k]], dtype=float)
        out = integration_dFdt(t_pair, x_cur, X, Y, Interpolant_u,
            Interpolant_v, periodic, bool_unsteady=True, time_data = t_data,
            verbose=False)
        
        traj_step = out[0]       # shape (2, 2, N)
        x_cur = traj_step[-1]    # shape (2, N), final positions at t_pair[-1]

        traj[k] = x_cur
        
    # Launch positions at the target and final backward positions
    x_launch = traj[0]
    x_final = traj[-1]

    return d0, ind_b, x_launch, x_final, traj, time_traj


def origin_likelihood(year, end_week, tar, paths):
    """
    Compute within-window origin likelihood for one target and window
    
    For each physical startin bin, the function estimates the probability
    of reaching the selected target at least once during the transport window.

    Parameters
    ----------
    year : int
        Analysis year.
    end_week : int
        End-week label for the transport window.
    tar : str
        Target name.
    paths : dict
        Path directory containing "windows_ops_dir.

    Returns
    -------
    origin : ndarray
        Normalized origin-likelihood field over physical bins.
    hit_prob : ndarray
        First-hit probability for each starting bin.
    d0 : physical_space
        Physical- space domain.
    ind_b : ndarray
        Target-bin indices.
    """
    # Build physical space and target
    d0 = physical_space(lon, lat, spatial_dis)
    ind_b = build_target_indices(d0, tar)
    N_og = len(d0.bins)
    
    # Load the saved finite-time windows for this year
    weeks_y, _, Pwins_y = load_yearly_window_operators(year, paths)
    
    # Select the window ending at the requested ISO week.
    matches = np.where(weeks_y == int(end_week))[0]

    j = int(matches[0])
    P_window_fwd = Pwins_y[j]
    
    # Augmented state-space size.
    S = P_window_fwd[0].shape[0]
    nirvana = S - 1

    ind_b = np.asarray(ind_b, dtype=int)
    ind_b = ind_b[(ind_b >= 0) & (ind_b < nirvana)]

    target_mask = np.zeros(S, dtype=bool)
    target_mask[ind_b] = True
    
    # First-hit probability for each physical startin bin
    hit_prob = np.zeros(N_og, dtype=float)
    
    # Loop over all physical bins
    for i in range(N_og):
        # Uniform initial distribution
        p = np.zeros(S, dtype=float)
        p[i] = 1.0

        total_hit = 0.0
        
        # Propagate through the finite-time window
        for P in P_window_fwd:
            p = p @ P
            
            # Prob. mass that first reaches the target at this step
            arrived = float(np.nansum(p[target_mask]))

            if arrived > 0.0:
                total_hit += arrived
                
                # Remove target mass so it is only counted once
                p[target_mask] = 0.0 

            p[nirvana] = 0.0 # Remove out-of-domain mass
            
        hit_prob[i] = total_hit

    # Do not treat the target itself as an origin region
    hit_prob[ind_b] = 0.0
    
    # Uniform-prior origin likelihood.
    # Since all starting bins are assumed equally likely, the posterior
    # origin map is proportional to the first-hit probability.
    origin = hit_prob.copy()
    
    # Normalize to obatin a probability distributions 
    total = np.nansum(origin)
    if total > 0.0:
        origin = origin / total
    else:
        origin[:] = np.nan

    return origin, hit_prob, d0, ind_b

def clim_origin_likelihood(tar, paths, week_range=None):
    """
    Compute climatological within-window origin likelihood for one target.

    The function averages first-hit probabilities over selceted climatology 
    years and end-weeks, then normalize the mean hit-probability field to 
    obtain an origin-likelihood map

    Parameters
    ----------
    tar : str
        Target name.
    paths : dict
        Path directory containing "windows_ops_dir".
    week_range : tuple(int, int), optional
        Inclusive end-week interval "(week_min, week_max)". If None, 
        late-season week range is used.

    Returns
    -------
    origin_clim : ndarray
        Normalized climatological origin-likelihood field over physical bins.
    hit_clim : ndarray
        Climatological mean first-hit probability over each physical starting
        bin.
    d0_ref : physical_space
        Physical-space domain.
    ind_b_ref : ndarray
        Target-bin indices.
    """

    ref_years = clim_years
    
    # Use late season by default
    if week_range is None:
        week_min, week_max = late_week_range
    else:
        if not isinstance(week_range, tuple) or len(week_range) != 2:
            raise ValueError(
                "week_range must be a tuple interval, for example (38, 52).")
            
        week_min, week_max = int(week_range[0]), int(week_range[1])
    
    # Selected end-weeks, inclusive of both endpoints.
    weeks = range(week_min, week_max + 1, week_step)

    hit_list = []

    d0_ref = None
    ind_b_ref = None
    
    # Loop over climatology years and selected end-weeks
    for year in ref_years:
        for week in weeks:
            _, hit_prob, d0, ind_b = origin_likelihood(
                year=year,
                end_week=week,
                tar=tar,
                paths=paths)

            hit_list.append(hit_prob)

            d0_ref = d0
            ind_b_ref = ind_b

    # Average first-hit probabilites over all selected years and windows   
    hit_clim = np.nanmean(np.stack(hit_list, axis=0), axis=0)
    
    origin_clim = hit_clim.copy()
    
    # Normalize to obatin a probability distribution
    total = np.nansum(origin_clim)
    if total > 0.0:
        origin_clim = origin_clim / total

    return origin_clim, hit_clim, d0_ref, ind_b_ref

