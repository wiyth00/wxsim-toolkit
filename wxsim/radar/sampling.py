"""WSR-88D sampling geometry: polar gates, beam height/width, scan strategy, ground clutter."""
import numpy as np

RE_EFF = 4.0 / 3.0 * 6371.0          # effective earth radius, km (standard refraction)
BEAMWIDTH_DEG = 0.95
VCP212 = [0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5]


class PolarGrid:
    def __init__(self, max_range_km=230.0, gate_km=0.25, az_res_deg=0.5):
        self.ng = int(round(max_range_km / gate_km))
        self.na = int(round(360 / az_res_deg))
        self.gate_km, self.az_res = gate_km, az_res_deg
        self.r = (np.arange(self.ng) + 0.5) * gate_km                   # gate centers
        self.az = (np.arange(self.na) + 0.5) * az_res_deg                # radial centers (deg, clockwise from N)
        A, R = np.meshgrid(np.deg2rad(self.az), self.r, indexing='ij')   # shape (na, ng)
        self.R = R
        self.X, self.Y = R * np.sin(A), R * np.cos(A)                    # km east / north of the radar

    def edges_xy(self):
        re = np.arange(self.ng + 1) * self.gate_km
        ae = np.deg2rad(np.arange(self.na + 1) * self.az_res)
        A, R = np.meshgrid(ae, re, indexing='ij')
        return R * np.sin(A), R * np.cos(A)


def beam_height(r_km, elev_deg):
    """Beam-center height above the radar (km) with 4/3-earth refraction."""
    e = np.deg2rad(elev_deg)
    return np.sqrt(r_km ** 2 + RE_EFF ** 2 + 2 * r_km * RE_EFF * np.sin(e)) - RE_EFF


def beam_half_depth(r_km):
    return r_km * np.tan(np.deg2rad(BEAMWIDTH_DEG / 2))


def ground_clutter(grid, rng, strength=1.0):
    """Speckled near-radar clutter / AP returns (dBZ, NaN where none), strongest inside ~25 km."""
    R = grid.R
    from scipy.ndimage import gaussian_filter
    # clumpy: clutter follows terrain/buildings, so gate hits cluster in patches
    patch = gaussian_filter(rng.standard_normal(R.shape), (2.0, 3.0))
    patch = (patch - patch.mean()) / patch.std()
    prob = (0.42 * np.exp(-R / 7.0) + 0.0015 * np.exp(-R / 60.0)) * (patch > 0.3)
    hit = rng.random(R.shape) < prob * strength
    db = 8 + 30 * rng.random(R.shape) * np.exp(-R / 18.0)
    return np.where(hit, db, np.nan)
