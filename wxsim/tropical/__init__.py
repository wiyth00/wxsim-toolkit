"""NHC-style tropical cyclone products (Phase 1: storm JSON, 5-day cone, TCM, TCP).

Modules
  storm     - tc_*.json loader/validator, units, time and geometry helpers (shared by every product)
  coast     - coastal breakpoints (data/tc_breakpoints.json) and breakpoint-to-breakpoint segments
  cone      - 5-day track-forecast cone graphic      python -m wxsim.tropical.cone <json>
  local     - zoomed office-area inland warning map          python -m wxsim.tropical.local <tcwarn json>
  windmap   - deterministic peak sustained/gust swath        python -m wxsim.tropical.windmap <tcwind json>
  products  - TCM forecast/advisory + TCP public advisory   python -m wxsim.tropical.products <json>

Planned (not implemented): two.py (Tropical Weather Outlook hatched areas), windprob.py (perturbed-track
Holland-profile ensemble -> wind-speed probabilities and arrival times, consuming storm.Storm), and a bridge
from a landfalling Storm to wxsim.env / wxsim.radar.
"""


def render(cfg, outdir):
    """render_all.py entry point for tc_*.json: cone graphic + text products."""
    from . import cone, products
    if cfg['advisory'].get('center', 'NHC') != 'WPC':
        cone.render(cfg, outdir)
    products.render(cfg, outdir)
