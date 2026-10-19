"""SIMULATED SCENARIO: Apalachee Bay sou'easter, Dec 15–19, 2026.

v3 is a reproducible, physics-informed *parametric* gust model, not an NWP
forecast or observational analysis. Low-level winds, radii, inversions and
mixing depths are prescribed proxies; no tropical pressure–wind relation.
Public v2 interfaces and peak.npz / frames.npz keys are retained.
"""
import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates
from scipy.interpolate import PchipInterpolator

MS2MPH = 2.23694
KT2MS = 0.514444
DX = 3000.0
HERE = os.path.dirname(os.path.abspath(__file__))
PEAK = os.path.join(HERE, 'peak.npz')
FRAMES = os.path.join(HERE, 'frames.npz')
DAY = ['Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MODEL_VERSION = 'v3.0-size3x'
STORM_SIZE_SCALE = 3.0  # Scale cyclone lengths relative to the original; retain intensity and track.
SEED = 1217
START, END, STEP = 12.0, 114.0, 0.1
# An optional aligned NPZ must contain elevation_m and roughness_m on X/Y.
TERRAIN_FILE = os.path.join(HERE, 'terrain.npz')

m = Basemap(projection='lcc', lat_0=30.5, lon_0=-84.0, lat_1=28.5, lat_2=33.5,
            width=1.5e6, height=1.45e6, resolution='i', area_thresh=100)
WP = np.array([
    (12,-96.2,25.4,1006),(24,-93.2,25.5,1003),(30,-90.6,26.2,997),
    (36,-88.7,26.9,982),(42,-87.0,27.9,962),(48,-85.5,29.0,947),
    (51,-84.85,29.55,939),(54,-84.2,30.05,941),(60,-83.9,30.7,946),
    (66,-84.2,31.2,951),(72,-84.6,31.6,955),(78,-84.9,31.4,960),
    (84,-84.7,31.0,963),(90,-84.2,30.9,965),(96,-83.4,31.0,968),
    (102,-82.4,31.2,972),(108,-81.3,31.4,975),
    (120,-78.5,32.3,980),(132,-75.5,33.8,983)])
RMAX = np.array([(12,150),(30,120),(36,90),(42,65),(48,50),(54,45),
                 (60,55),(66,75),(78,110),(90,150),(132,185)])
# Maximum low-level (~850 mb) jet speed, m/s: independent scenario parameters.
LLJ = np.array([(12,22),(30,31),(36,40),(42,47),(48,55),(51,59),
                (54,58),(60,55),(66,51),(78,45),(90,40),(108,36),(132,29)])
# Low-level BASE tangential 34 / 50 / 64 kt radii, km, NE/SE/SW/NW.
# These are NOT 10 m sustained-wind radii or operational hurricane radii.
# Thresholds exceeding the time-dependent LLJ maximum are inactive (returned 0).
RADIUS_HOURS = np.array([12,30,42,51,60,78,96,108,132])
RADII = np.array([
 [[400,310,250,340],[220,190,170,200],[170,155,150,165]],
 [[430,350,280,380],[210,180,150,190],[130,110,100,120]],
 [[450,380,310,410],[200,160,135,185],[110,85,75,100]],
 [[460,400,320,430],[220,175,145,200],[115,90,75,105]],
 [[510,440,370,490],[250,220,180,240],[135,110,90,130]],
 [[590,520,440,560],[340,290,250,325],[185,150,135,175]],
 [[650,580,490,620],[390,340,290,370],[225,195,180,215]],
 [[680,620,510,640],[430,385,320,410],[250,225,200,240]],
 [[720,660,540,680],[480,420,360,450],[280,255,220,270]],
], dtype=float)
STING = dict(lon=-83.6, lat=30.9, hour=59., duration_h=1.7,
             along_km=85., across_km=23., azimuth=125., boost_ms=6.)
CONVECTION = dict(cells=150, first_h=40., last_h=51., width_min_km=5.,
                  width_max_km=15., max_gust_min_mph=62., max_gust_max_mph=102.)
# (peak hour, latitude, e-folding lifetime): transient bow/line-end vortices.
BOWS = [(43.6,26.1,1.1),(46.0,28.3,1.4),(48.2,29.8,1.0)]
RIDGE = [(-84.4,34.7),(-83.6,35.1),(-83.0,35.5),(-82.1,35.9),
         (-81.5,36.3),(-80.8,36.8),(-80.3,37.1)]

# Scale the cyclone, keeping local downbursts/sting jets and geographic masks fixed.
RMAX = RMAX.astype(float)
RMAX[:,1] *= STORM_SIZE_SCALE
RADII *= STORM_SIZE_SCALE

th = WP[:,0]
_lon = PchipInterpolator(th, WP[:,1]); _lat = PchipInterpolator(th, WP[:,2])
_p = PchipInterpolator(th, WP[:,3]); _R = PchipInterpolator(RMAX[:,0],RMAX[:,1])
_jet = PchipInterpolator(LLJ[:,0], LLJ[:,1])
_rad = PchipInterpolator(RADIUS_HOURS, RADII, axis=0)

@dataclass(frozen=True)
class Member:
    """Smooth coherent perturbations, not independent noise at every time."""
    track_dx_km: float = 0.
    track_dy_km: float = 0.
    pressure_delta_mb: float = 0.
    radius_scale: float = 1.
    timing_h: float = 0.
    sting_dx_km: float = 0.
    sting_dy_km: float = 0.
    sting_timing_h: float = 0.
    sting_scale: float = 1.
    inversion_scale: float = 1.
    seed: int = SEED

CONTROL = Member()

def pressure(t): return float(_p(np.clip(t,th[0],th[-1])))
def center_ll(t):
    t = np.clip(t,th[0],th[-1])
    return float(_lon(t)), float(_lat(t))
def center_xy(t): return m(*center_ll(t))
def track_xy(hours): return np.array([center_xy(h) for h in hours])
def hlabel(h):
    # Nearest hour, including day rollover; keeps the v2 short label contract.
    h = int(np.floor(float(h) + .5))
    return f'{DAY[min(h//24,len(DAY)-1)]} {h%24:02d}Z'

def smooth(x0,x1,v):
    s = np.clip((v-x0)/(x1-x0),0,1)
    return s*s*(3-2*s)

xs = np.arange(0,m.xmax+DX,DX); ys = np.arange(0,m.ymax+DX,DX)
X,Y = np.meshgrid(xs,ys)
LON,LAT = m(X,Y,inverse=True)
pts = np.c_[X.ravel(),Y.ravel()]
land = np.zeros(X.shape,bool)
for (px,py),typ in zip(m.coastpolygons,m.coastpolygontypes):
    inside = Path(np.c_[px,py]).contains_points(pts).reshape(X.shape)
    if typ == 1: land |= inside
    elif typ == 2: land &= ~inside
X = X.astype(np.float32); Y = Y.astype(np.float32)
LON = LON.astype(np.float32); LAT = LAT.astype(np.float32)
dcoast = (distance_transform_edt(land)*DX/1000.).astype(np.float32)

def ellipse_gauss(lon0,lat0,a_km,b_km,az_deg):
    x0,y0 = m(lon0,lat0)
    az = np.radians(az_deg)
    dx,dy = X-x0,Y-y0
    along = dx*np.sin(az)+dy*np.cos(az)
    across = -dx*np.cos(az)+dy*np.sin(az)
    return np.exp(-((along/(a_km*1000))**2+(across/(b_km*1000))**2))

wedge = ellipse_gauss(-82.9,34.3,270,130,40)
rp = np.array([m(lo,la) for lo,la in RIDGE])
dmin = np.full(X.shape,1e12)
for (x0,y0),(x1,y1) in zip(rp[:-1],rp[1:]):
    sx,sy = x1-x0,y1-y0
    tt = np.clip(((X-x0)*sx+(Y-y0)*sy)/(sx*sx+sy*sy),0,1)
    dmin = np.minimum(dmin,np.hypot(X-(x0+tt*sx),Y-(y0+tt*sy)))
ridge = np.exp(-(dmin/20000.)**2).astype(np.float32)
gmask = smooth(-84.5,-82.5,LON)*smooth(29.5,31.5,LAT)*(1-smooth(35.5,37.,LAT))
gdecay = np.where(land,np.exp(-dcoast/90.),1.)
roughness = np.where(land,.05,.002).astype(np.float32)
slope_x = slope_y = None
TERRAIN_MODE = 'coarse land/water roughness + schematic ridge proxy (no DEM)'
if os.path.exists(TERRAIN_FILE):
    with np.load(TERRAIN_FILE) as terrain:
        z = terrain['elevation_m']; roughness = terrain['roughness_m']
        if z.shape != X.shape or roughness.shape != X.shape:
            raise ValueError('terrain.npz must be aligned to X/Y; shape mismatch')
        if not np.isfinite(z).all() or not np.isfinite(roughness).all() or (roughness<=0).any() or (roughness>=10).any():
            raise ValueError('Terrain must be finite, with 0 < roughness_m < 10')
        slope_y,slope_x = np.gradient(gaussian_filter(z.astype(float),2),DX)
    TERRAIN_MODE = 'supplied aligned DEM slopes and roughness; no resolved gap flow'
# Log-profile exposure relative to an open-land reference, not fixed gust factors.
exposure = np.clip(np.log(10/roughness)/np.log(150/roughness)/.66,.65,1.08)


def mixing_gust(llj_ms,inversion_k,mixing_depth_m,jet_height_m=900.,exposure_factor=1.):
    """Return (3-second gust proxy, 2-minute mean proxy, coupling), m/s.

    Bulk Ri uses a prescribed 500 m inversion layer and 0.45*LLJ shear.
    Ri and mixed depth limit momentum transfer. The gust is mean + 2.8 sigma;
    this is an explicit tunable closure, not a fitted gust-duration conversion.
    """
    ri = (9.81/300.)*np.maximum(inversion_k,0)*500/np.maximum((.45*llj_ms)**2,25)
    coupling = np.clip(mixing_depth_m/jet_height_m,0,1)/(1+4*ri)
    mean = llj_ms*(.27+.48*coupling)*exposure_factor
    sigma = llj_ms*(.015+.055*coupling)*exposure_factor
    return mean+2.8*sigma,mean,coupling


def quadrant_radii(t,member=CONTROL):
    tt = np.clip(t-member.timing_h,th[0],th[-1])
    radii = np.asarray(_rad(tt))*member.radius_scale
    vmax = float(_jet(tt))*intensity_scale(tt,member)
    return np.where(np.array([34,50,64])[:,None]*KT2MS < vmax,radii,0.)

def intensity_scale(t,member):
    # Pressure perturbations change kinetic-energy scale modestly; the control
    # uses prescribed LLJ speeds, not pressure mapped to tropical intensity.
    dp = max(1012-pressure(t),4.)
    return np.sqrt(max(dp-member.pressure_delta_mb,4.)/dp)

def angular_radius(values,theta):
    # Smoothly interpolate between quadrant centers without seams at 0/360.
    q = ((theta-np.pi/4)%(2*np.pi))/(np.pi/2)
    lo = np.floor(q).astype(np.int8); f = q-lo
    # theta increases CCW: NE, NW, SW, SE.
    vals = np.asarray(values)[[0,3,2,1]]
    f = .5-.5*np.cos(np.pi*f)
    return vals[lo]*(1-f)+vals[(lo+1)%4]*f


def base_jet(t,member=CONTROL):
    tt = float(np.clip(t-member.timing_h,th[0],th[-1]))
    cx,cy = center_xy(tt)
    # Offset ramps during development, then remains coherent through the loop.
    ramp = .35+.65*smooth(12,60,tt)
    cx += 1000*member.track_dx_km*ramp; cy += 1000*member.track_dy_km*ramp
    dx,dy = X-cx,Y-cy
    r = np.maximum(np.hypot(dx,dy)/1000.,.1)
    theta = np.arctan2(dy,dx)
    rm = float(_R(tt))*member.radius_scale
    vmax = float(_jet(tt))*intensity_scale(tt,member)
    radii = quadrant_radii(t,member)
    # Continuous piecewise profile, with exact low-level threshold anchors.
    V = vmax*np.minimum(r/rm,1)
    prev_r = np.full(X.shape,rm,dtype=np.float32); prev_v = vmax
    for k,kt in ((2,64),(1,50),(0,34)):
        if vmax <= kt*KT2MS: continue
        rr = angular_radius(radii[k],theta).astype(np.float32)
        rr = np.maximum(rr,prev_r+3.*STORM_SIZE_SCALE)
        vv = kt*KT2MS
        between = (r>prev_r)&(r<=rr)
        interp = prev_v+(vv-prev_v)*np.clip((r-prev_r)/(rr-prev_r),0,1)
        V = np.where(between,interp,V)
        prev_r,prev_v = rr,vv
    V = np.where(r>prev_r,prev_v*np.exp(-(r-prev_r)/(230.*STORM_SIZE_SCALE)),V)
    # A curled cold-conveyor/bent-back-front maximum; core fills, outer arc grows.
    maturity = smooth(36,54,tt)*(1-.65*smooth(96,120,tt))
    angle_dist = np.arctan2(np.sin(theta-np.radians(215)),np.cos(theta-np.radians(215)))
    hook_radius = rm*(1.65+.30*np.cos(theta)) + 35*STORM_SIZE_SCALE
    hook = np.exp(-((r-hook_radius)/(45*STORM_SIZE_SCALE+.20*rm))**2-(angle_dist/1.65)**2)*maturity
    V += 8.*hook
    # Rotate tangential flow into frictional inflow and add translation & high/low gradient.
    tx,ty = -dy/(r*1000),dx/(r*1000)
    ix,iy = -dx/(r*1000),-dy/(r*1000)
    inflow = np.where(land,np.radians(28),np.radians(16))
    cx0,cy0 = center_xy(tt-.5); cx1,cy1 = center_xy(tt+.5)
    ut,vt = (cx1-cx0)/3600.,(cy1-cy0)/3600.
    translation = .45*np.exp(-r/(500.*STORM_SIZE_SCALE))
    gt = smooth(48,62,tt)*(1-smooth(100,114,tt))
    gx = -.707*11.*gt*gmask*gdecay; gy = .707*11.*gt*gmask*gdecay
    u = V*(np.cos(inflow)*tx+np.sin(inflow)*ix)+translation*ut+gx
    v = V*(np.cos(inflow)*ty+np.sin(inflow)*iy)+translation*vt+gy
    return u,v,r,theta,hook,tt


def sting_weight(t,member=CONTROL):
    tt = t-member.timing_h-member.sting_timing_h
    az = np.radians(STING['azimuth'])
    x0,y0 = m(STING['lon'],STING['lat'])
    # Descending jet moves southeast at 40 km/h; track perturbations move it too.
    dist = (tt-STING['hour'])*40000
    x0 += dist*np.sin(az)+1000*(member.track_dx_km+member.sting_dx_km)
    y0 += dist*np.cos(az)+1000*(member.track_dy_km+member.sting_dy_km)
    along = (X-x0)*np.sin(az)+(Y-y0)*np.cos(az)
    across = -(X-x0)*np.cos(az)+(Y-y0)*np.sin(az)
    return np.exp(-((along/(STING['along_km']*1000))**2+(across/(STING['across_km']*1000))**2
                    +((tt-STING['hour'])/STING['duration_h'])**2))

_NOISE = {}; _CELLS = {}

def correlated_noise(t,seed):
    if seed not in _NOISE:
        rng = np.random.default_rng(seed)
        fields = []
        for _ in range(2):
            f = gaussian_filter(rng.standard_normal(X.shape),4)
            fields.append((f/f.std()).astype(np.float32))
        _NOISE[seed] = fields
    a,b = _NOISE[seed]
    # Smooth time correlation and spatial correlation; deterministic for arbitrary call order.
    return np.clip(a*np.cos(t/1.8)+b*np.sin(t/1.8),-2.5,2.5)


def convective_cells(seed):
    if seed not in _CELLS:
        rng = np.random.default_rng(seed+77)
        cells = []
        for _ in range(CONVECTION['cells']):
            birth = rng.uniform(CONVECTION['first_h'],CONVECTION['last_h'])
            lat = rng.uniform(25.7,30.2)
            lon = -83.+.45*(birth-45)+.4*(lat-28.2)
            x,y = m(lon,lat)
            width = rng.uniform(CONVECTION['width_min_km'],CONVECTION['width_max_km'])
            cells.append((birth,x,y,rng.uniform(.65,1.6),width,
                          rng.uniform(CONVECTION['max_gust_min_mph'],CONVECTION['max_gust_max_mph']),
                          rng.uniform(-.25,.25)))
        _CELLS[seed] = cells
    return _CELLS[seed]


def convective_gust(t,member=CONTROL):
    tt = t-member.timing_h
    # Bowing line background remains sub-severe; localized cells carry the extremes.
    bow = .20*np.cos((LAT-28.2)*3)
    lonf = -83.+.45*(tt-45)+.4*(LAT-28.2)+bow
    active = np.exp(-((tt-45.5)/5.5)**4)
    latmask = smooth(25.,25.8,LAT)*(1-smooth(29.9,30.5,LAT))
    envelope = np.exp(-((LON-lonf)/.22)**2)*latmask*active
    g = 43.*envelope
    for birth,x,y,life,width,peak,tilt in convective_cells(member.seed):
        age = tt-birth
        if age < 0 or age > life: continue
        cx = x+age*46000+member.track_dx_km*1000
        cy = y+age*tilt*18000+member.track_dy_km*1000
        reach = 45000.
        j0 = max(0,int((cy-reach)/DX)); j1 = min(Y.shape[0],int((cy+reach)/DX)+1)
        i0 = max(0,int((cx-reach)/DX)); i1 = min(X.shape[1],int((cx+reach)/DX)+1)
        if i1<=i0 or j1<=j0: continue
        dx,dy = X[j0:j1,i0:i1]-cx,Y[j0:j1,i0:i1]-cy
        # width is cross-track FWHM: 5–15 km, ~2–5 cells at this resolution.
        cross = dy-tilt*dx
        a = np.exp(-(dx/12000)**2-(cross/(width*1000/1.665))**2)
        amp = np.sin(np.pi*age/life)**.6
        # Treat as a total gust proxy; do not double-count background wind.
        cell = peak*a*amp
        g[j0:j1,i0:i1] = np.maximum(g[j0:j1,i0:i1],cell)
    # Bow-apex and line-end vortex gust maxima, also translating with the line.
    for hour,lat0,lifetime in BOWS:
        weight = np.exp(-((tt-hour)/lifetime)**2)
        lat0 += .04*(tt-hour)
        lon0 = -83.+.45*(tt-45)+.4*(lat0-28.2)
        # Coherent member offsets apply to both downbursts and bow vortices.
        x0,y0 = m(lon0,lat0)
        lon0,lat0 = m(x0+member.track_dx_km*1000,y0+member.track_dy_km*1000,inverse=True)
        g = np.maximum(g,78.*ellipse_gauss(lon0,lat0,24,9,80)*active*weight)
    return g.astype(np.float32),envelope


def instant_fields(t,member=CONTROL,texture=True):
    u,v,r,theta,hook,tt = base_jet(t,member)
    sj = sting_weight(t,member)
    speed = np.hypot(u,v)+STING['boost_ms']*member.sting_scale*sj
    # Wedge persists throughout the loop; post-frontal mixing is mainly W/SW.
    post = smooth(-.1,.7,-np.cos(theta))*smooth(36,54,tt)*(1-.7*wedge)
    marine = (~land)*smooth(30.5,32.,LAT)*smooth(-.4,.5,np.cos(theta))
    inversion = (7.*wedge + 2.8*marine)*member.inversion_scale
    inversion *= 1-.95*sj
    depth = 700.+550.*post-540.*wedge-360.*marine
    depth = np.maximum(depth,100.)
    depth = depth*(1-sj)+1350.*sj
    gust,mean,coupling = mixing_gust(speed,inversion,depth,exposure_factor=exposure)
    # No fake DEM claim. The fallback ridge proxy is weak and mixing-dependent.
    if slope_x is None:
        terrain_factor = 1+.10*ridge*coupling
    else:
        along_slope = (slope_x*u+slope_y*v)/np.maximum(np.hypot(u,v),1.)
        terrain_factor = 1+np.clip(2.*along_slope,-.12,.20)*coupling
    gust *= terrain_factor; mean *= terrain_factor
    if texture:
        gust *= 1+.025*correlated_noise(tt,member.seed)*(.3+.7*coupling)
    conv,envelope = convective_gust(t,member)
    g = np.maximum(gust*MS2MPH,conv)
    return dict(gust=np.maximum(g,5.).astype(np.float32),mean_ms=mean,
                llj_ms=speed,inversion_k=inversion,mixing_depth_m=depth,
                coupling=coupling,sting=sj,ccb=hook,convective_mph=conv)


def vortex_gust(t):
    # Compatibility entry point: surface background gust without convection/sting boost.
    u,v,r,theta,hook,tt = base_jet(t)
    post = smooth(-.1,.7,-np.cos(theta))*smooth(36,54,tt)*(1-.7*wedge)
    marine = (~land)*smooth(30.5,32.,LAT)*smooth(-.4,.5,np.cos(theta))
    gust,_,_ = mixing_gust(np.hypot(u,v),7*wedge+2.8*marine,
                          np.maximum(700+550*post-540*wedge-360*marine,100),
                          exposure_factor=exposure)
    return gust*MS2MPH

def inst_gust(t,member=CONTROL): return instant_fields(t,member)['gust']


def sample(arr,lon,lat):
    x,y = m(lon,lat)
    if not (0<=x<=X[0,-1] and 0<=y<=Y[-1,0]):
        return float('nan')
    return float(map_coordinates(arr,[[y/DX],[x/DX]],order=1,mode='nearest')[0])


def signature():
    with open(__file__,'rb') as source:
        digest = hashlib.sha256(source.read())
    if os.path.exists(TERRAIN_FILE):
        with open(TERRAIN_FILE,'rb') as fh:
            for chunk in iter(lambda: fh.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def member_peak(member=CONTROL,step=STEP,frames=False,grid_stride=1):
    """Full-time integration; optional grid_stride only subsamples stored fields."""
    shape = X[::grid_stride,::grid_stride].shape
    peak = np.zeros(shape,np.float32); times = np.full(shape,START,np.float32)
    ft,F = [],[]
    station_t,station_g,station_mean = [],[],[]
    if frames:
        from station_catalog import ST
        sp = np.array([m(lo,la) for _,_,la,lo in ST])
        coords = [sp[:,1]/DX,sp[:,0]/DX]
    for i,t in enumerate(np.arange(START,END+step/2,step)):
        t = float(round(t,6))
        fields = instant_fields(t,member)
        g = fields['gust'][::grid_stride,::grid_stride]
        if frames:
            station_t.append(t)
            station_g.append(map_coordinates(fields['gust'],coords,order=1,mode='nearest'))
            station_mean.append(map_coordinates(fields['mean_ms']*MS2MPH,coords,order=1,mode='nearest'))
        update = g>peak
        peak = np.maximum(peak,g); times = np.where(update,t,times)
        if frames and 30<=t<=112 and abs(t*2-round(t*2))<1e-6:
            ft.append(t); F.append(g.astype(np.float16))
        if member==CONTROL and i%200==0:
            print(f'control {t:.1f} h / {END:.0f} h',flush=True)
    if frames:
        np.savez_compressed(os.path.join(HERE,'station_series.npz'),
                            hours=station_t,gust_mph=station_g,mean_mph=station_mean,
                            signature=signature(),simulated=True)
    return peak,times,np.array(ft),np.array(F)


def compute():
    peak,times,ft,F = member_peak(frames=True)
    meta = dict(signature=signature(),model_version=MODEL_VERSION,simulated=True,step_h=STEP)
    np.savez_compressed(PEAK,G=peak,T=times,**meta)
    np.savez_compressed(FRAMES,ft=ft,F=F,**meta)


def cache_valid(path):
    try:
        with np.load(path) as d: return str(d['signature'])==signature()
    except (OSError,ValueError,KeyError): return False

# Existing renderers import G/T. Explicit skip allows fast isolated physics tests.
if os.environ.get('APALACHEE_SKIP_CACHE')=='1':
    G = T = None
else:
    if '--recompute' in sys.argv or not all(cache_valid(p) for p in (PEAK,FRAMES,os.path.join(HERE,'station_series.npz'))):
        compute()
    with np.load(PEAK) as d: G,T = d['G'],d['T']

if __name__=='__main__':
    print('SIMULATED SCENARIO',MODEL_VERSION)
    print('min pressure',min(pressure(t) for t in np.arange(12,114,.1)))
    print('max gust overall',round(float(G.max()),1),'land',round(float(G[land].max()),1),
          'water',round(float(G[~land].max()),1))
    for name,lon,lat in [('TLH',-84.353,30.396),('VLD',-83.277,30.783),('ABY',-84.194,31.535),
      ('JAX',-81.688,30.494),('SAV',-81.202,32.127),('CHS',-80.041,32.899),
      ('ATL',-84.428,33.637),('MCO',-81.309,28.429),('TPA',-82.533,27.975),
      ('BHM',-86.754,33.563),('MOB',-88.243,30.691),('SSI',-81.391,31.152)]:
        print(name,round(sample(G,lon,lat),1),hlabel(sample(T,lon,lat)))
