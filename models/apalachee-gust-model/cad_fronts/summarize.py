"""Summarize verified size experiments without recomputing their fields."""
import csv
import json
from pathlib import Path
import numpy as np
import model as fm

HERE = Path(__file__).resolve().parent


def run():
    summaries, stations = [], []
    for name in ('size_2p5x', 'size_3x'):
        case = HERE / name
        meta = json.loads((case / 'summary.json').read_text())
        with np.load(case / 'peak.npz') as d:
            peak = d['G']
        with np.load(case / 'fields.npz') as d:
            hours = d['hours']
            front = d['F_K_100km_3h'][:, 3:-3, 3:-3]
            ix = int(np.argmin(np.abs(hours - 48)))
            f48 = front[ix]
            theta48 = d['theta_K'][ix]
            cfg = fm.Config(**meta['config'])
            flow48 = fm.flow(float(hours[ix]), cfg)
            surface48 = fm.surface_gust(float(hours[ix]), theta48, flow48, cfg)
            nga = {}
            for sid, lon, lat in [('KATL', -84.428, 33.637), ('KAHN', -83.327, 33.949), ('KGVL', -83.830, 34.272), ('KABY', -84.194, 31.535), ('KTLH', -84.353, 30.396)]:
                nga[sid] = dict(theta_K=round(fm.bg.sample(theta48, lon, lat), 2),
                    upper_speed_ms=round(fm.bg.sample(np.hypot(flow48['upper_u_ms'], flow48['upper_v_ms']), lon, lat), 2),
                    lower_transport_speed_ms=round(fm.bg.sample(np.hypot(flow48['u_ms'], flow48['v_ms']), lon, lat), 2),
                    gust_proxy_mph=round(fm.bg.sample(surface48['gust_mph'], lon, lat), 2),
                    inversion_proxy_K=round(fm.bg.sample(surface48['inversion_K'], lon, lat), 2))
            metrics = dict(radius_scale=meta['config']['radius_scale'],
                pressure_min_mb=meta['pressure_min_mb'],
                central_pressure_parameter_min_mb=meta['pressure_central_parameter_min_mb'],
                quebec_high_mb=meta['pressure_background']['high_center_mb'],
                grid_peak_gust_mph=round(meta['peak_gust_max_mph'], 2),
                land_peak_gust_mph=round(meta['peak_land_max_mph'], 2),
                event_gust_footprint_km2={str(k): int((peak >= k).sum() * 9) for k in (40, 58, 74, 100)},
                interior_frontogenesis_minmax_K_100km_3h=[float(front.min()), float(front.max())],
                hour48_frontogenesis_minmax_K_100km_3h=[float(f48.min()), float(f48.max())],
                hour48_frontogenesis_p01_p99_K_100km_3h=np.percentile(f48, [1, 99]).tolist(),
                hour48_diagnostics=nga,
                theta_minmax_K=meta['stored_theta_minmax_K'],
                source_fingerprint=meta['source_fingerprint'], simulated=True)
        summaries.append(metrics)
        with np.load(case / 'station_series.npz') as d:
            station_ids = d['station_ids']
            gust = d['gust_mph'].max(axis=0)
            hours = d['hours']
            peak_hours = hours[d['gust_mph'].argmax(axis=0)]
            stations.append({str(sid): (round(float(g), 2), round(float(h), 3))
                             for sid, g, h in zip(station_ids, gust, peak_hours)})
    payload = dict(simulated=True, scenarios=summaries,
        scope='Dry, parameterized CAD with 1049 mb off-domain Quebec high, shallow barrier flow, inversion and explicit restoring source. Only circular cyclone component is in radial gradient balance; no moist two-way atmospheric dynamics.',
        area_definition='Projected grid area inside the original Southeast map only; not the full storm footprint.',
        gust_definition='Surface gust closure target, not time-resolved observed 3-second gusts; complete station series, no synthetic reporting/outage layer.')
    (HERE / 'comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    from station_catalog import ST
    with (HERE / 'station_comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'gust_2p5x_mph', 'peak_hour_2p5x', 'gust_3x_mph', 'peak_hour_3x', 'simulated'])
        for sid, name, *_ in ST:
            writer.writerow([sid, name, *stations[0][sid], *stations[1][sid], True])
    print(json.dumps(payload, indent=2))
    for sid in ('KBHM', 'KMOB', 'KATL', 'KTLH', 'KJAX', 'KVLD'):
        print(sid, stations[0][sid], stations[1][sid])


if __name__ == '__main__':
    # Importing the catalog only requires its parent directory, not the model.
    import sys
    sys.path.insert(0, str(HERE.parent))
    run()
