#!/usr/bin/env python3
"""Fast manufactured checks for the dry frontogenesis model.

Run from any directory with ``python realistic_fronts/validate.py``. Optional
``--case PATH`` checks one saved model case without recomputing it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model as fm


def _affine_grid(n: int = 25, dx: float = 1000.0):
    # The model sampler maps projected meters to indices by x / DX, y / DX.
    axis = np.arange(n, dtype=float) * dx
    return np.meshgrid(axis, axis), dx


class FrontogenesisTests(unittest.TestCase):
    def setUp(self):
        (self.x, self.y), self.dx = _affine_grid()
        self.theta0 = 290.0
        self.gx, self.gy = 2.0e-5, -3.0e-5
        self.theta = self.theta0 + self.gx * self.x + self.gy * self.y

    def test_uniform_theta_has_zero_frontogenesis(self):
        zero = np.zeros_like(self.x)
        result = fm.frontogenesis(np.full_like(self.x, 290.0), zero, zero, self.dx)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(np.max(np.abs(result))), 0.0)

    def test_rigid_rotation_has_zero_frontogenesis(self):
        omega = 2.0e-4
        u, v = -omega * self.y, omega * self.x
        result = fm.frontogenesis(self.theta, u, v, self.dx)
        self.assertLess(float(np.max(np.abs(result[2:-2, 2:-2]))), 2.0e-12)

    def test_compressive_and_extensional_strain_sign_and_units(self):
        rate = 1.7e-4  # s^-1
        grad = 4.0e-5  # K m^-1
        theta = self.theta0 + grad * self.x
        compressive = fm.frontogenesis(theta, -rate * self.x, rate * self.y, self.dx)
        extensional = fm.frontogenesis(theta, rate * self.x, -rate * self.y, self.dx)
        target = rate * grad  # K m^-1 s^-1
        np.testing.assert_allclose(compressive[2:-2, 2:-2], target, rtol=2e-5, atol=1e-14)
        np.testing.assert_allclose(extensional[2:-2, 2:-2], -target, rtol=2e-5, atol=1e-14)

    def test_shear_term_uses_both_gradient_components(self):
        shear = 2.5e-4  # s^-1
        theta = self.theta0 + self.gx * self.x + self.gy * self.y
        result = fm.frontogenesis(theta, shear * self.y, np.zeros_like(self.y), self.dx)
        expected = -(self.gx * self.gy * shear) / np.hypot(self.gx, self.gy)
        np.testing.assert_allclose(result[2:-2, 2:-2], expected, rtol=2e-5, atol=1e-14)


class BalancedFlowTests(unittest.TestCase):
    def test_pressure_profile_limits_and_original_minimum(self):
        cfg = fm.Config(2.5)
        t = 51.0
        flow = fm.flow(t, cfg)
        radius = np.asarray(flow["radius_m"], dtype=float)
        pressure = np.asarray(flow["p_mb"], dtype=float)
        pc = fm.bg.pressure(t)
        self.assertAlmostEqual(pc, 939.0, places=5)
        self.assertGreaterEqual(float(pressure.min()), pc - 1e-4)
        self.assertLessEqual(float(pressure.max()), 1012.0 + 1e-4)
        self.assertLess(float(pressure.min() - pc), 1e-4)

        rmax = float(flow["rmax_m"])
        far_index = np.unravel_index(np.argmax(radius), radius.shape)
        expected_far = pc + (1012.0 - pc) * np.exp(-rmax / radius[far_index])
        self.assertAlmostEqual(float(pressure[far_index]), float(expected_far), delta=0.003)

        # Finite-difference the 2-D pressure field in an annulus around R.
        py, px = np.gradient(pressure, fm.DX, fm.DX, edge_order=2)
        grad_pa_m = np.hypot(px, py) * 100.0
        annulus = (radius > 0.85 * rmax) & (radius < 1.15 * rmax)
        ratio = grad_pa_m[annulus] / np.asarray(flow["dpdr_pa_m"])[annulus]
        self.assertGreater(ratio.size, 100)
        self.assertLess(abs(float(np.median(ratio)) - 1.0), 0.08)

    def test_gradient_wind_satisfies_radial_balance(self):
        flow = fm.flow(51.0, fm.Config(3.0))
        radius = np.asarray(flow["radius_m"], dtype=float)
        rmax = float(flow["rmax_m"])
        speed = np.asarray(flow["gradient_wind_ms"], dtype=float)
        dpdr = np.asarray(flow["dpdr_pa_m"], dtype=float)
        f = float(flow["f_s"])
        rho = fm.AIR_DENSITY_KG_M3
        annulus = (radius > 0.5 * rmax) & (radius < 1.5 * rmax)
        lhs = np.square(speed[annulus]) / radius[annulus] + f * speed[annulus]
        rhs = dpdr[annulus] / rho
        rel = np.abs(lhs - rhs) / np.maximum(rhs, 1e-12)
        self.assertLess(float(np.max(rel)), 2e-6)
        self.assertTrue(np.isfinite(speed).all())
        self.assertGreaterEqual(float(speed.min()), 0.0)


class AdvectionTests(unittest.TestCase):
    def _small_model_grid(self):
        (x, y), dx = _affine_grid(n=25, dx=1000.0)
        return x.astype(np.float32), y.astype(np.float32), dx

    def test_constant_field_is_preserved(self):
        x, y, dx = self._small_model_grid()
        cfg = fm.Config(2.5)
        with patch.object(fm, "X", x), patch.object(fm, "Y", y), patch.object(fm, "DX", dx):
            result = fm.advect_theta(np.full(x.shape, 291.25, dtype=np.float32), 40.0, 120.0,
                                     cfg, ambient_boundary=False)
        np.testing.assert_array_equal(result, np.full(x.shape, 291.25, dtype=np.float32))

    def test_one_step_is_bounded_and_does_not_wrap_domain_edges(self):
        x, y, dx = self._small_model_grid()
        cfg = fm.Config(2.5)
        theta = np.full(x.shape, 270.0, dtype=np.float32)
        theta[:, :3] = 300.0
        constant_flow = {
            "u_ms": np.full(x.shape, 25.0, dtype=np.float32),
            "v_ms": np.zeros(x.shape, dtype=np.float32),
        }
        with patch.object(fm, "X", x), patch.object(fm, "Y", y), patch.object(fm, "DX", dx), \
                patch.object(fm, "flow", return_value=constant_flow):
            result = fm.advect_theta(theta, 40.0, 120.0, cfg, ambient_boundary=False)
        self.assertGreaterEqual(float(result.min()), 270.0 - 1e-5)
        self.assertLessEqual(float(result.max()), 300.0 + 1e-5)
        # Positive u advects east; the west departure point exits the domain.
        # Nearest/open edge handling retains the west value instead of wrapping
        # the cold east edge into the west boundary as a periodic sampler would.
        self.assertGreater(float(result[12, 0]), 299.0)
        self.assertLess(float(result[12, -1]), 271.0)

    def test_frontogenesis_is_not_total_gradient_tendency(self):
        x, y, dx = self._small_model_grid()
        cfg = fm.Config(2.5)
        theta = (290.0 + 4.0 * np.exp(-((x - 12_000.0) ** 2 + (y - 12_000.0) ** 2) / (2 * 4_000.0**2))).astype(np.float32)
        zero = np.zeros(x.shape, dtype=np.float32)
        constant_flow = {"u_ms": zero, "v_ms": zero}
        with patch.object(fm, "X", x), patch.object(fm, "Y", y), patch.object(fm, "DX", dx), \
                patch.object(fm, "flow", return_value=constant_flow):
            f = fm.frontogenesis(theta, zero, zero, dx)
            after = fm.advect_theta(theta, 40.0, 120.0, cfg, ambient_boundary=False)
        self.assertLess(float(np.max(np.abs(f))), 1e-12)  # no deformation source
        self.assertGreater(float(np.max(np.abs(after - theta))), 1e-6)  # diffusion still changes |grad theta|

    def test_compressive_transport_grows_linear_theta_gradient(self):
        x, y, dx = self._small_model_grid()
        cfg = fm.Config(2.5)
        # Use a deliberately resolved manufactured gradient so float32 theta
        # storage does not dominate the one-step tendency difference.
        rate, grad, dt = 5.0e-5, 2.0e-3, 120.0
        xc = float(x[0, x.shape[1] // 2])
        yc = float(y[y.shape[0] // 2, 0])
        theta = (290.0 + grad * x).astype(np.float32)
        u = (-rate * (x - xc)).astype(np.float32)
        v = (rate * (y - yc)).astype(np.float32)
        strain_flow = {"u_ms": u, "v_ms": v}
        with patch.object(fm, "X", x), patch.object(fm, "Y", y), patch.object(fm, "DX", dx), \
                patch.object(fm, "flow", return_value=strain_flow):
            initial_f = fm.frontogenesis(theta, u, v, dx)
            after = fm.advect_theta(theta, 40.0, dt, cfg, ambient_boundary=False)
        # Exclude the boundary closure and the immediate stencil edge.
        ix = slice(5, -5)
        g0 = np.gradient(theta, dx, dx, edge_order=2)[1][10, ix]
        g1 = np.gradient(after, dx, dx, edge_order=2)[1][10, ix]
        expected = grad * np.exp(rate * dt)
        np.testing.assert_allclose(g1, expected, rtol=5e-4, atol=1e-9)
        measured_tendency = float(np.mean((g1 - g0) / dt))
        diagnosed = float(np.mean(initial_f[10, ix]))
        self.assertAlmostEqual(measured_tendency / diagnosed, 1.0, delta=0.01)


def _check_case(case: Path) -> list[str]:
    """Return archive validation failures; this never launches a simulation."""
    errors: list[str] = []
    required = (case / "fields.npz", case / "peak.npz", case / "station_series.npz", case / "summary.json")
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        return [f"{case}: missing {', '.join(missing)}"]
    try:
        with np.load(required[0]) as z:
            fields = {key: z[key] for key in z.files}
        with np.load(required[1]) as z:
            peak = {key: z[key] for key in z.files}
        with np.load(required[2]) as z:
            stations = {key: z[key] for key in z.files}
        summary = json.loads(required[3].read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{case}: cannot read archive: {exc}"]

    expected_hours = fm.HOURS
    hours = np.asarray(fields.get("hours", []), dtype=float)
    if hours.shape != expected_hours.shape or not np.allclose(hours, expected_hours, atol=1e-4, rtol=0):
        errors.append(f"{case}: expected 165 half-hour frames from hour 30 through 112")
    expected_shape = (len(hours),) + fm.X.shape
    for key in ("theta_K", "F_K_100km_3h", "gust_mph", "p_mb", "u_ms", "v_ms"):
        value = fields.get(key)
        if value is None or value.shape != expected_shape:
            errors.append(f"{case}: {key} shape does not match frame/grid dimensions")
        elif not np.isfinite(value).all():
            errors.append(f"{case}: {key} contains nonfinite values")
    for key in ("G", "T"):
        value = peak.get(key)
        if value is None or value.shape != fm.X.shape:
            errors.append(f"{case}: peak {key} shape does not match model grid")
        elif not np.isfinite(value).all():
            errors.append(f"{case}: peak {key} contains nonfinite values")

    if "theta_K" in fields and np.isfinite(fields["theta_K"]).all():
        lo, hi = np.percentile(fields["theta_K"], [0, 100])
        if lo < 260.0 or hi > 320.0:
            errors.append(f"{case}: theta range [{lo:.2f}, {hi:.2f}] K is outside broad physical bounds")
    if "F_K_100km_3h" in fields and fields["F_K_100km_3h"].ndim == 3:
        interior = np.ones(fm.X.shape, dtype=bool)
        interior[:3, :] = interior[-3:, :] = False
        interior[:, :3] = interior[:, -3:] = False
        front = fields["F_K_100km_3h"][:, interior]
        if not (np.any(front > 0.001) and np.any(front < -0.001)):
            errors.append(f"{case}: signed frontogenesis lacks positive or negative interior values above 0.001")
        if (all(key in fields and fields[key].shape == expected_shape
                for key in ("theta_K", "u_ms", "v_ms")) and hours.size
                and all(np.any(np.isclose(hours, target, atol=1e-4)) for target in (54.0, 78.0))):
            for target in (54.0, 78.0):
                index = int(np.argmin(np.abs(hours - target)))
                calculated = fm.frontogenesis(fields["theta_K"][index], fields["u_ms"][index],
                                              fields["v_ms"][index]) * fm.FRAME_SCALE
                saved = fields["F_K_100km_3h"][index]
                if not np.allclose(saved, calculated, rtol=1e-5, atol=1e-4):
                    errors.append(f"{case}: archived frontogenesis does not reproduce from theta/wind at hour {target:g}")
                    break

    if ("G" in peak and "gust_mph" in fields and fields["gust_mph"].ndim == 3
            and fields["gust_mph"].shape[0] > 0 and np.isfinite(peak["G"]).all()):
        frame_max = np.max(fields["gust_mph"], axis=0)
        # Frame gusts are float16; allow one local half-precision ULP at the
        # largest representable frame value when comparing to float32 peaks.
        tol = float(np.max(np.abs(np.spacing(fields["gust_mph"].astype(np.float16))).astype(np.float32)))
        if np.any(peak["G"] + tol < frame_max):
            errors.append(f"{case}: peak gust falls below a stored frame gust beyond float16 tolerance")
        start = float(summary.get("integration_start_hour", summary.get("theta_initial_time_hour", 12.0)))
        end = float(summary.get("integration_end_hour", 112.0))
        if "T" in peak and hours.size and (np.nanmin(peak["T"]) < start - 1e-3 or np.nanmax(peak["T"]) > end + 1e-3):
            errors.append(f"{case}: peak time lies outside the integration interval")

    sh = np.asarray(stations.get("hours", []), dtype=float)
    sg = stations.get("gust_mph")
    sm = stations.get("mean_mph")
    start = float(summary.get("integration_start_hour", summary.get("theta_initial_time_hour", 12.0)))
    end = float(summary.get("integration_end_hour", 112.0))
    expected_station_count = int(round((end - start) / 0.1)) + 1
    if (sh.size != expected_station_count or not np.isclose(sh[0], start, atol=2e-5)
            or not np.isclose(sh[-1], end, atol=2e-5)
            or not np.allclose(np.diff(sh), 0.1, atol=2e-5, rtol=0)):
        errors.append(f"{case}: station samples are not spaced every six minutes")
    if sg is None or sm is None or sg.shape != sm.shape or sg.ndim != 2 or sg.shape[0] != sh.size or sg.shape[1] != 69:
        errors.append(f"{case}: station series dimensions do not match 69 stations and sample times")
    elif not np.isfinite(sg).all() or not np.isfinite(sm).all():
        errors.append(f"{case}: station series contains nonfinite values")

    scale = summary.get("config", {}).get("radius_scale")
    expected_scale = 3.0 if "3x" in case.name else 2.5
    if scale != expected_scale:
        errors.append(f"{case}: configured radius scale {scale!r} does not match case name ({expected_scale:g}x)")
    if not str(summary.get("pressure", "")).startswith("original WP central pressure"):
        errors.append(f"{case}: metadata does not state that original central pressure is preserved")
    fingerprints = [summary.get("source_fingerprint"), fields.get("source_fingerprint"),
                   peak.get("source_fingerprint"), stations.get("source_fingerprint")]
    if any(fp is None or str(np.asarray(fp).item()) != fm._source_fingerprint() for fp in fingerprints):
        errors.append(f"{case}: source fingerprints are missing or stale")

    if "p_mb" in fields and fields["p_mb"].size and np.isfinite(fields["p_mb"]).all():
        if abs(float(np.min(fields["p_mb"])) - 939.0) > 0.5:
            errors.append(f"{case}: stored pressure minimum does not preserve the original 939 mb track minimum")
    if abs(float(summary.get("pressure_min_mb", np.inf)) - 939.0) > 1e-4:
        errors.append(f"{case}: summary pressure minimum does not preserve the original 939 mb track minimum")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=Path, default=[],
                        help="validate a saved case directory (may be repeated); does not recompute")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    failures = 0 if result.wasSuccessful() else 1
    for case in args.case:
        errors = _check_case(case.expanduser().resolve())
        if errors:
            failures += 1
            for error in errors:
                print(f"FAIL: {error}", file=sys.stderr)
        else:
            print(f"PASS: {case} archive is consistent")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
