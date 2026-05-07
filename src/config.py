# =============================================================================
#                             config
# =============================================================================
import sys
from pathlib import Path
import numpy as np

# =============================================================================
#                             User choices
# =============================================================================
T = 14                      # Transition time [days]
transport_horizon = 36      # Finite-time transport horizon [weeks]
m = 1000                    # Particles per bin
late_week_range= (38, 52)   # Late-season definiton [ISO week]

# Spatial grid spacing in degrees
#grid_spacing = 1

# Computational domain
lon = [-15, 20]
lat = [50, 72]

# Maximum number of parallel worker processes used for yearly diagnostics
DEFAULT_MAX_WORKERS = 7     

# =============================================================================
#                         Derived values
# =============================================================================
# Spatial grid spacing in degrees
#spatial_dis = int((lon[1]-lon[0])/grid_spacing)+1
spatial_dis = (np.abs(lon[0]) + np.abs(lon[1]))

# Number of transition matrices in one finite-time window
K_win = (7*transport_horizon)//T

# Step size in ISO week between consecutive transition matrices
week_step = T//7

# Kernels per analysis year
Kseg = 52 //week_step

# =============================================================================
#                         TPT setup
# =============================================================================
# Initial distribution choice
init_dist_A = True
init_dist_U = False

# Years of which transition matrices are built
matrix_years = list(range(1996,2025))

# Apolemia years
apolemia_years = [1997, 2001, 2021, 2022, 2023, 2024]

# Years used in the actual analysis
all_years = list(range(1997, 2025))
#clim_years = [y for y in all_years if y not in apolemia_years]
clim_years = all_years[:]

# Target region
# [lon0, lat0, dlon_factor, dlat_factor, n_points]
# lon0, lat0 = starting pos, dlon_factor = move east, dlat_factor = move north
# n_points = maximum number of bins
targets = {
    "Bergen": [5, 62, 0, -0.5, 4],
    "Tromsø": [18.5, 70.5, -1, -0.7, 3],
    "Brønnøysund": [12.5, 66.5, -1, -0.9, 3],
    "Skagerrak": [8.0, 58.2, 0.7, 0.2, 4]}

# Source region
s_minlon = -7
s_maxlon = -5
s_minlat = 58
s_maxlat = 61

# Orkney-Shetland section pathway gate
lonpos_g1 = -0.5
latmin_g1 = 57
latmax_g1 = 60

# Northeast of Shetland section pathway gate
lonpos_g2 = 0.5
latmin_g2 = 60
latmax_g2 = 64

# =============================================================================
#                                   Paths
# =============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

THIRD_PARTY_DIR = SRC_DIR / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

DATA_DIR = SRC_DIR /"data"

OUTPUT_DIR = SRC_DIR / "Output"
Pa_dir = OUTPUT_DIR / f"titm{m}"
