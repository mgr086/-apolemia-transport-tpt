# =============================================================================
#                Time-inhomogeneous transition matrix builder
# =============================================================================
"""
Build yearly sequences of augmented time-inhomogeneous transition matrices. 

This script reads processed MATLAB velocity fields, interpolates the surface
velocity fields, initializes synthetic particles in each valid spatial bin, 
advects them over fixed interval length T, estimates open-system transition
matrices from the resulting particle displacements, and augments each matrix
with an abosrbing out-of-domain state. 

For each analysis year, the script saves one sequence of augmented transition
matrices ordered from oldest to newest. These matries are used as input to the 
finite-time transition path theory and transport-diagnostics. 
"""

# Standard library
import os
import sys
import gc
import csv
import time
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Limit BLAS threading to avoid oversubscription during multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Default number of worker processes for parallel year-by-year computation
DEFAULT_MAX_WORKERS = min(7, os.cpu_count() or 1)

# Third-party libraries
import numpy as np
import pandas as pd
import scipy.io as sio

# Project configuration
from config import (
    SRC_DIR, THIRD_PARTY_DIR, DATA_DIR, T, m, lon, lat,
    spatial_dis, matrix_years, Kseg, Pa_dir)


# Third-party transport and discretization tools
from pygtm_miron.dataset import trajectory
from pygtm_miron.matrix import matrix_space
from pygtm_miron.physical import physical_space
from Tbarrier_hallergroup import (
    convert_meters_per_second_to_deg_per_day, 
    interpolant_unsteady,integration_dFdt_final)

# Write diagnostic tests for augmented transition matrices
TEST_TITM = True

# Add local project and third-party source directories to the Python path
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(THIRD_PARTY_DIR))

# Folder for transition matrices
Pa_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
#                              Helpers
# =============================================================================
def build_file_years():
    """
    Collect existing yearly MATLABk velocity fields
    
    For each year in "matrix_years", this function expects a file named
    "NAO{year}.mat" in "DATA_DIR". 
    
    Returns
    -------
    list[tuple[Path, int]]
        List of (file_path, analysis_year) pairs for files that exist on disk. 
    """
    data_dir = DATA_DIR
    file_years = []
    
    for year in matrix_years:
        # Construct filename
        filename = data_dir / f"NAO{year}.mat"
        file_years.append((filename, year))
        
    existing = [(filename, year) for filename, year 
                in file_years if filename.exists()]
    missing_years = [year for filename, year 
                     in file_years if not filename.exists()]
        
    if missing_years:
        print(f'[INFO] Missing analysis years: {missing_years}')
   
    return existing
        

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

def build_fixed_augmented_from_tm(tm, N_og):
    """
    Embed a reduced open-system transition matrix into the original fixed
    domain and append an absorbing out-of-domain state
    
    Active bins are mapped back to their original indices. Missing transition
    mass is assigned to the aborbing state
    
    Parameters
    ----------
    tm: matrix_space
        Transition-matrix object containing "P", "fo", and "domain.id_og"
    N_og: int
        Number of bins in the original bin domain
        
    Returns
    -------
    numpy.ndarray
        Augmented transition matrix of shape (N_og +1, N_og +1)
    """
    # Mapping from tm current bins -> original bin indices
    id_og = np.asarray(tm.domain.id_og, dtype=int)

    # Embed P into fixed-size matrix
    P_full = np.zeros((N_og, N_og))
    P_full[np.ix_(id_og, id_og)] = tm.P

    # Outflow for fixed-size state space
    # Default: whatever mass is missing from the row goes to nirvana/outside
    fo_full = 1.0 - P_full.sum(axis=1)
    
    # Numerical safeguard against small negative round-off errors
    fo_full[fo_full < 0] = 0.0

    # For bins that exist in tm, use tm.fo directly
    if tm.fo is not None and len(tm.fo) == len(id_og):
        fo_full[id_og] = tm.fo

    # Build augmented (nirvana absorbing)
    P_ain_full = np.zeros((N_og + 1, N_og + 1))
    P_ain_full[:N_og, :N_og] = P_full 
    
    # top right block: transition from O to w (outflow)
    P_ain_full[:N_og, -1] = fo_full
    P_ain_full[-1, -1] = 1.0    # Bottom right: Make it row-stochastic

    return P_ain_full

def tests_in(P_a):
    """
    Run basic consistency check on an augmented transition matrix. Checks
    non-negative entries, absorbing final state, row-stochasticity. 
    
    Parameters
    ----------
    P_a : numpy.ndarray
        Augmented transition matrix with the absorbing out-of-domain state as
        the final row and final column.

    Returns
    -------
    list[str | int]
        Returns `["ok"]` if all tests pass. Otherwise returns a list of failes
        tests
    """
    
    # Empty list for storing tests that fails
    bad = []   
    
    # -------------- Non-negativity --------------------
    # Check that we only have positive probabilities
    tol = 1e-14
    if np.all(P_a >= -tol):
        pass
    else:
        bad.append(1)
        print("Non-negativity: some entries are negative.")
        print("Min entry:", P_a.min())
    
    
    # --------- Abosrbing out-of-domain state ---------------
    if not (np.allclose(P_a[-1, :-1], 0.0, atol=1e-12) and 
            np.isclose(P_a[-1, -1], 1.0, atol=1e-12)):
        bad.append(2)
        
    # ------------------ Row-stochasticity ------------------------
    # Check that all rows sums to 1, i.e., closed domain, no leakage
    row_sums = P_a.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-12):
        pass
    else:
        bad.append(3)
        print("Row-stochasticity: FAIL")
        print("Row sums min / max:", row_sums.min(), row_sums.max())
    
    if len(bad) == 0:
        bad = ["ok"]
    
    return bad

# =============================================================================
#                  Build titm
# =============================================================================
def run_one_year(filename: str, year: int) -> dict:
    """
    Build and save the augmented transition-matrix sequence for one anaysis
    year. 

    Parameters
    ----------
    filename : str
        Path to the yearly MATLAB velocity field
    year : int
        Analysis year associated with the velocity field.

    Returns
    -------
    dict
        Dictonary cotaining the year, input file, success flag, runtime in sec, 
        and potential error message.

    """
    t0 = time.time()
    try:
        # ---- Create domain (per-process; avoids pickling issues) ----
        d0 = physical_space(lon, lat, spatial_dis)
        N_og = len(d0.bins)

        # Per-year CSV to avoid concurrent writes to the same file
        if TEST_TITM:
            results_file = Pa_dir / "titm_tests" / f'test_titm{year}.csv'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = [
                'End week','Active bins', 'Min in bins',
                'Mean in bins', "Max in bins", 'Tests']
            
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames)
                writer.writeheader()
            
        
        # Load yearly velocity fields
        mat_file = sio.loadmat(filename)

        U_ms = mat_file['u'].astype(np.float32, copy=False)   # (Ny, Nx, Nt)
        V_ms = mat_file['v'].astype(np.float32, copy=False)   # (Ny, Nx, Nt)
        
        # Make x and y true 1D coordinate arrays
        x = np.asarray(mat_file['x'], dtype=np.float32).ravel()   # (Nx,)
        y = np.asarray(mat_file['y'], dtype=np.float32).ravel()   # (Ny,)
        
        # Make time a 2D row vector
        t_sec = np.asarray(mat_file['t'], dtype=float)
        
        del mat_file
        # Compute meshgrid of dataset
        X, Y = np.meshgrid(x, y) # array (NY, NX), array (NY, NX)
        X = X.astype(np.float32, copy=False)
        Y = Y.astype(np.float32, copy=False)

        # Convert veolicty from m/s to degrees/day
        U,V = convert_meters_per_second_to_deg_per_day(X, Y, U_ms, V_ms)
        
        del U_ms, V_ms

        # -----------------Interpolate velocity-----------------
        # Set nan values to zero, so that we can apply interpolant. 
        # Interpolant does not work if the array contains nan values. 
        U[np.isnan(U)] = 0
        V[np.isnan(V)] = 0
        
        
        if U.shape[:2] != (len(y), len(x)):
            raise ValueError(
                f"Grid/velocity mismatch: U.shape[:2]={U.shape[:2]}, "
                f"but expected {(len(y), len(x))} from y,x lengths."
            )
        
        # Interpolate velocity data using cubic spatial interpolation
        Interpolant = interpolant_unsteady(X, Y, U, V, method = "cubic")

        Interpolant_u = Interpolant[0] # RectangularBivariateSpline-object
        Interpolant_v = Interpolant[1] # RectangularBivariateSpline-object
        
        del U, V, Interpolant

        # Periodic boundary conditions
        periodic_x = False # bool
        periodic_y = False # bool
        periodic_t = False # bool
        periodic = [periodic_x, periodic_y, periodic_t]

        # Unsteady velocity field
        bool_unsteady = True # bool

        # -------------- Grid setup  --------------
        # bin centers 
        x_c = 0.5*(d0.vx[:-1] + d0.vx[1:])
        y_c = 0.5*(d0.vy[:-1] + d0.vy[1:])

        # Grid of bin centers
        Xc, Yc = np.meshgrid(x_c, y_c)

        # Stack to satisfy flow map input
        pts = np.vstack([Xc.ravel(), Yc.ravel()])  # (2, nbins_grid)

        # Find bin_id for each bin. Bin_id == -1 means outside/invalid/land
        bin_id = d0.find_element(pts[0], pts[1])
        valid = bin_id >= 0     # Only valid bin if it has an id
        pts = pts[:, valid]     # Keep only points of valid bins
        bin_id = bin_id[valid]  # id of the bins that are valid

        # replicate m particles per bin
        jitter_frac = 0.9      # <=1, Jitter amplitude as fraction of bin size

        # Create a random number with a fixed seed. 
        rng = np.random.default_rng(12346)  

        # bin sizes
        dx = d0.vx[1] - d0.vx[0]
        dy = d0.vy[1] - d0.vy[0]

        reps = np.full(pts.shape[1], m, dtype=int)  # Avoid bug
        reps[0] = m - 1  # make exactly one bin have m-1 particles instead of m
        pts_rep = np.repeat(pts, reps, axis=1)  

        # Add random jitter
        # Shift to -0.5 and 0.5 to push particle in both direction
        jx = (rng.random(pts_rep.shape[1])-0.5)*jitter_frac*dx
        jy = (rng.random(pts_rep.shape[1])-0.5)*jitter_frac*dy

        pts_rep[0,:] += jx
        pts_rep[1, :] +=jy

       
        x0 = pts_rep.astype(np.float32, copy=False)
  
        del pts, bin_id, valid, reps, pts_rep, jx, jy


        # ----------- Time setup for synthetic trajectories ---------------
        t_days_since_1970 = t_sec/86400
        matlab_1970 = datenum_from_date(datetime(1970, 1, 1))
        t_data = t_days_since_1970 + matlab_1970

        week52_end = datetime.fromisocalendar(year, 52, 7) # Sunday of week 52

        tN_traj = datenum_from_date(week52_end)     # End of week 52
          
        # Outdir path for saved augmented transition matrices 
        P_ains_pkl = Pa_dir / f"P_ains_{year}.pkl"
        
        Psa_in = [] # Empty list for storing augmented time-inhomogeneous P
        
        N_state_expected = None  # fixed dimension check

        # Step backward through the year and build one TM per interval
        for _ in range(Kseg):

            #d_tmin = copy.deepcopy(d0)
            d_tmin = d0
            
            t0_traj = tN_traj -T

            dt_traj = .05      # integration time step in days
            time_traj = np.arange(t0_traj, tN_traj + dt_traj, dt_traj)
            tN_this = tN_traj        # Time of this window
            tN_traj -= T        # Shift end time T days backwards
            
            # Week counter and week updater
            # Convert back to datetime
            end_date = datetime.fromordinal(int(tN_this) + 366)  
            iso_year, week, _ = end_date.isocalendar()

            week_data = {}  # Empty dict for storing week_data values
            week_data["End week"] = week


            yT = integration_dFdt_final(time_traj, x0, X, Y, 
                                        Interpolant_u,Interpolant_v,
                                          periodic,bool_unsteady,
                                          t_data, verbose=False)

            
            # --------- Transition matrix ---------------
            # Only the initial and final positions are needed from the flow map
            x_0 = x0[0, :].copy()
            y_0 = x0[1, :].copy()
            x_T = yT[0, :].copy()
            y_T = yT[1, :].copy()

            del yT     # Free memory

            data_in = trajectory(x=None, y=None, t=None, ids=None)
            data_in.T = T
            data_in.x0, data_in.y0, data_in.xt, data_in.yt = x_0, y_0, x_T, y_T

            # ------ create matrix object and fill the transition matrix ----
            tm_in = matrix_space(d_tmin)
            tm_in.fill_transition_matrix(data_in)

            week_data["Active bins"] = tm_in.N
            week_data["Min in bins"] = tm_in.M.min()
            week_data["Max in bins"] = tm_in.M.max()
            week_data["Mean in bins"] = round(np.mean(tm_in.M))
            
            # ------------- Augmented transition matrix ----------------------

            # Build augmented time-inhomogeneous transition matrix 
            P_ain = build_fixed_augmented_from_tm(
                tm_in, N_og).astype(np.float32, copy=False)
            
            del x_0, y_0, x_T, y_T
            del data_in
            del tm_in
            
            gc.collect()
            
            # Save augmented time-inhom. tm for later (new -> old)
            Psa_in.append(P_ain)

            # --------- Tests on augmented transition matrix -----------------
            bad_in = tests_in(P_ain)
            week_data["Tests"] = bad_in

            # Dimension consistency
            if N_state_expected is None:
                N_state_expected = P_ain.shape[0]
            elif P_ain.shape[0] != N_state_expected:
                raise ValueError(f"Time-inhom kernel dimension changed:"
                                 f"{P_ain.shape[0]} vs {N_state_expected}")
                
            if TEST_TITM:
                with open(results_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames)
                    writer.writerow(week_data)
            
            
        # ------------ Save transition matrices --------------------
        if not Psa_in:
            raise ValueError("No transition matrices were built" 
                             f"for year {year}. Check time window logic.")
        
        # Augmented tm, reverse list so it is oldest -> newest
        P_ains = Psa_in[::-1]  

        # Save P_ains (List of augmented time-inhomogeneous transition matrix)
        pd.to_pickle(P_ains,  P_ains_pkl)
        
        dt = time.time() - t0
        return {'year': year, 'file': filename, 'ok': True, 'seconds': dt}
    
    except Exception as e:
        dt = time.time() - t0
        return {
            'year': year,
            'file': filename,
            'ok': False,
            'seconds': dt,
            'error': str(e),
            'traceback': traceback.format_exc()}
    

# =============================================================================
#                                Main
# =============================================================================
def main():
    """
    Build augmented transition-matrix sequence for all available analysis years.
    """
    file_years = build_file_years()

    if not file_years:
        print("[ERROR] No input .mat files found." 
              "Check PROJECT_ROOT and filenames.")
        return

    max_workers = min(DEFAULT_MAX_WORKERS, len(file_years))
    print(f"[INFO] Running {len(file_years)}"
          "files with max_workers={max_workers}")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one_year, str(fn), year) 
                for (fn, year) in file_years]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get('ok'):
                print(f"[OK] year={r['year']} ({Path(r['file']).name})"
                      f" in {r['seconds']:.1f}s")
            else:
                print(f"[FAIL] year={r['year']} ({Path(r['file']).name}) after"
                      f"{r['seconds']:.1f}s: {r.get('error')}")
                print(r.get('traceback',''))

    ok = sum(1 for r in results if r.get('ok'))
    fail = len(results) - ok
    print(f'[INFO] Done. ok={ok}, fail={fail}')

# run
if __name__ == "__main__":
    main()
