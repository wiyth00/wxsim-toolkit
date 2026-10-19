#!/usr/bin/env python3
"""Fast manufactured checks for the dry CAD and frontogenesis model.

Run from any directory with ``python cad_fronts/validate.py``. Optional
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
    def test_pressure_profile_keeps_central_curve_and_reports_total_minimum(self):
        cfg = fm.Config(2.5)
        t = 51.0
        flow = fm.flow(t, cfg)
        radius = np.asarray(flow["radius_m"], dtype=float)
        pressure = np.asarray(flow["p_mb"], dtype=float)
        pc = fm.bg.pressure(t)
        self.assertAlmostEqual(pc, 939.0, places=5)
        # The canonical WP central-pressure parameter remains 939 mb. Adding
        # a sloping background can move the actual grid minimum modestly below
        # that value; it must be reported, not clipped back onto the curve.
        self.assertGreaterEqual(float(pressure.min()), pc - 9.0)
        self.assertLessEqual(float(pressure.min()), pc + 9.0)
        self.assertLess(float(pressure.max()), fm.CAD_HIGH_CENTER_MB)

        # At the storm center the total field must attain the canonical WP
        # central parameter. The pressure anomaly must relax toward the
        # background away from the center (a reversed exp(-R/r) profile puts
        # the low on the far field instead).
        cx, cy, _ = fm._center(t)
        center_index = np.unravel_index(np.argmin(np.square(fm.X - cx)
                                                   + np.square(fm.Y - cy)), fm.X.shape)
        self.assertAlmostEqual(float(pressure[center_index]), pc, delta=1.0)

        rmax = float(flow["rmax_m"])
        far_index = np.unravel_index(np.argmax(radius), radius.shape)
        expected_far = (float(fm.background_pressure(fm.X[far_index], fm.Y[far_index]))
                        - (float(flow["background_pressure_center_mb"]) - pc)
                        * (1.0 - np.exp(-rmax / radius[far_index])))
        self.assertAlmostEqual(float(pressure[far_index]), float(expected_far), delta=0.003)

        # Remove the background before checking the circular component's
        # derivative; total pressure also contains the CAD high gradient.
        circular_deficit_pa = (np.asarray(fm.background_pressure(fm.X, fm.Y))
                               - pressure) * 100.0
        py, px = np.gradient(circular_deficit_pa, fm.DX, fm.DX, edge_order=2)
        grad_pa_m = np.hypot(px, py)
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


class CADPhysicsTests(unittest.TestCase):
    def test_quebec_high_and_southeast_background_gradient(self):
        high = float(fm.background_pressure(fm.CAD_HIGH_X, fm.CAD_HIGH_Y))
        self.assertAlmostEqual(high, 1049.0, places=5)
        nga_x, nga_y = fm.bg.m(-83.5, 33.5)
        nga = float(fm.background_pressure(nga_x, nga_y))
        self.assertLess(nga, 1049.0)

        # Compare the model's total local pressure gradient with a control
        # flow in which the Quebec high is replaced by its flat far field.
        pbg = np.asarray(fm.background_pressure(fm.X, fm.Y), dtype=float)
        py, px = np.gradient(pbg, fm.DX, fm.DX, edge_order=1)
        i, j = np.unravel_index(np.argmin(np.square(fm.bg.LON + 83.5)
                                           + np.square(fm.bg.LAT - 33.5)), fm.X.shape)
        local_gradient = float(np.hypot(px[i, j], py[i, j]))
        self.assertGreater(local_gradient, 1e-7)
        def disabled_high(x, y, t=51.0, config=None):
            del t, config
            xx, yy = np.broadcast_arrays(np.asarray(x), np.asarray(y))
            return np.full(xx.shape, fm.BACKGROUND_FAR_MB, dtype=float)
        cfg = fm.Config(2.5, cad_enabled=True)
        high_flow = fm.flow(51.0, cfg)
        with patch.object(fm, "background_pressure", side_effect=disabled_high):
            flat_flow = fm.flow(51.0, cfg)
        def local_pressure_gradient(field):
            gy, gx = np.gradient(np.asarray(field, dtype=float), fm.DX, fm.DX, edge_order=1)
            return float(np.hypot(gx[i, j], gy[i, j]))
        high_total_gradient = local_pressure_gradient(high_flow["p_mb"])
        flat_total_gradient = local_pressure_gradient(flat_flow["p_mb"])
        self.assertGreater(abs(high_total_gradient - flat_total_gradient), 1e-7)

    def test_hydrostatic_slp_ridge_is_excluded_from_upper_geostrophic_wind(self):
        enabled = fm.Config(2.5, cad_enabled=True)
        disabled = fm.Config(2.5, cad_enabled=False)
        pressure_on = fm.background_pressure(fm.X, fm.Y, config=enabled)
        pressure_off = fm.background_pressure(fm.X, fm.Y, config=disabled)
        ridge = np.asarray(pressure_on - pressure_off, dtype=float)
        mask = fm.cad_mask(fm.X, fm.Y)
        core = mask > 0.8
        outside = mask < 0.05
        core_ridge = float(np.mean(ridge[core]))
        self.assertGreater(core_ridge, 0.2)
        # The Gaussian edge tapers smoothly beyond the operational core mask;
        # residual pressure there should remain small relative to the ridge.
        self.assertLess(float(np.max(np.abs(ridge[outside]))), 0.1 * core_ridge)

        # Raising the shallow ridge depth changes total SLP, but must leave
        # the high-only upper background wind exactly invariant.
        with patch.object(fm, "CAD_DEPTH_MAX_M", 1600.0):
            pressure_deep = fm.background_pressure(fm.X, fm.Y, config=enabled)
            deep_wind = fm._background_geostrophic_wind(35.0, enabled)
        shallow_wind = fm._background_geostrophic_wind(35.0, enabled)
        control_wind = fm._background_geostrophic_wind(35.0, disabled)
        self.assertGreater(float(np.mean(np.asarray(pressure_deep - pressure_off)[core])),
                           1.8 * float(np.mean(ridge[core])))
        for got, expected in zip(shallow_wind, control_wind):
            np.testing.assert_array_equal(got, expected)
        for got, expected in zip(deep_wind, shallow_wind):
            np.testing.assert_array_equal(got, expected)

        # Independent analytic geostrophic wind for the broad Gaussian high.
        # The maximum Gaussian pressure-gradient magnitude is A/S/sqrt(e),
        # which bounds wind speed at this fixed latitude.
        f = 2.0 * fm.OMEGA_S * np.sin(np.deg2rad(35.0))
        amp_pa = (fm.CAD_HIGH_CENTER_MB - fm.BACKGROUND_FAR_MB) * 100.0
        scale = fm.BACKGROUND_SCALE_M
        rx, ry = fm.X - fm.CAD_HIGH_X, fm.Y - fm.CAD_HIGH_Y
        shape = np.exp(-0.5 * (rx * rx + ry * ry) / (scale * scale))
        dpdx = -amp_pa * rx / (scale * scale) * shape
        dpdy = -amp_pa * ry / (scale * scale) * shape
        analytic_u = -dpdy / (fm.AIR_DENSITY_KG_M3 * f)
        analytic_v = dpdx / (fm.AIR_DENSITY_KG_M3 * f)
        np.testing.assert_allclose(shallow_wind[0], analytic_u, rtol=0.004, atol=0.03)
        np.testing.assert_allclose(shallow_wind[1], analytic_v, rtol=0.004, atol=0.03)
        speed = np.hypot(shallow_wind[0], shallow_wind[1])
        analytic_bound = amp_pa / scale / np.sqrt(np.e) / (fm.AIR_DENSITY_KG_M3 * f)
        self.assertTrue(np.isfinite(speed).all())
        self.assertLessEqual(float(np.max(speed)), analytic_bound * 1.05)

    def test_corridor_selects_eastern_appalachians_and_north_georgia(self):
        samples = {
            "southern_appalachians": (-82.4, 34.4),
            "north_georgia": (-83.5, 33.5),
            "west_alabama": (-87.0, 33.0),
        }
        mask = {}
        for name, (lon, lat) in samples.items():
            x, y = fm.bg.m(lon, lat)
            mask[name] = float(fm.cad_mask(x, y))
        self.assertGreater(mask["southern_appalachians"], 0.9)
        self.assertGreater(mask["north_georgia"], 0.75)
        self.assertLess(mask["west_alabama"], 0.4)

    def test_low_level_transport_follows_barrier_and_is_finite(self):
        fl = fm.flow(51.0, fm.Config(2.5, cad_enabled=True))
        mask = np.asarray(fl["cad_mask"])
        active = mask > 0.9
        self.assertGreater(int(active.sum()), 0)
        for key in ("u_ms", "v_ms", "upper_u_ms", "upper_v_ms"):
            self.assertTrue(np.isfinite(fl[key]).all(), key)
        # Positive NE-to-SW along the barrier is the (+0.8,+0.6) axis;
        # the prescribed low-layer jet must project southwestward.
        along_barrier = 0.8 * np.asarray(fl["u_ms"]) + 0.6 * np.asarray(fl["v_ms"])
        self.assertLess(float(np.median(along_barrier[active])), 0.0)

    def test_cad_restoring_source_cools_scalar_after_transport_step(self):
        cfg_cad = fm.Config(2.5, cad_enabled=True)
        cfg_control = fm.Config(2.5, cad_enabled=False)
        initial = (290.0 + 1e-5 * (fm.Y - np.mean(fm.Y))).astype(np.float32)
        zero_wind = {"u_ms": np.zeros(fm.X.shape, dtype=np.float32),
                     "v_ms": np.zeros(fm.X.shape, dtype=np.float32)}
        with patch.object(fm, "flow", return_value=zero_wind):
            with_cad = fm.advect_theta(initial, 40.0, 120.0, cfg_cad,
                                       ambient_boundary=False)
            without_cad = fm.advect_theta(initial, 40.0, 120.0, cfg_control,
                                          ambient_boundary=False)
        core = fm.cad_mask(fm.X, fm.Y) > 0.9
        self.assertTrue(np.isfinite(fm.theta_source(initial, cfg_cad)).all())
        self.assertGreater(float(np.max(fm.theta_source(initial, cfg_cad))), 0.0)
        self.assertEqual(float(np.max(np.abs(fm.theta_source(initial, cfg_control)))), 0.0)
        self.assertLess(float(np.mean(with_cad[core])), float(np.mean(without_cad[core])))

    def test_stability_limits_same_upper_wind_and_mean_never_exceeds_gust(self):
        cfg = fm.Config(2.5, cad_enabled=True)
        fl = fm.flow(51.0, cfg)
        mask = np.asarray(fl["cad_mask"]) > 0.9
        ambient = fm._ambient_theta(fm.X, fm.Y, 51.0, cfg)
        cold = np.full(fm.X.shape, 270.0, dtype=np.float32)
        warm = ambient.astype(np.float32)
        same_upper = dict(fl)
        same_upper["u_ms"] = np.full(fm.X.shape, 10.0, dtype=np.float32)
        same_upper["v_ms"] = np.zeros(fm.X.shape, dtype=np.float32)
        cold_surface = fm.surface_gust(51.0, cold, same_upper, cfg)
        warm_surface = fm.surface_gust(51.0, warm, same_upper, cfg)
        self.assertLess(float(np.mean(cold_surface["gust_mph"][mask])),
                        float(np.mean(warm_surface["gust_mph"][mask])))
        for surface in (cold_surface, warm_surface):
            self.assertTrue(np.isfinite(surface["gust_mph"]).all())
            self.assertTrue(np.isfinite(surface["mean_mph"]).all())
            self.assertTrue(np.all(surface["mean_mph"] <= surface["gust_mph"] + 1e-6))

    def test_strong_inversion_is_confined_to_the_cad_cold_pool(self):
        cfg = fm.Config(2.5, cad_enabled=True)
        fl = fm.flow(51.0, cfg)
        ambient = fm._ambient_theta(fm.X, fm.Y, 51.0, cfg)
        cold_pool = np.full(fm.X.shape, 270.0, dtype=np.float32)
        core = np.asarray(fl["cad_mask"]) > 0.9
        outside = np.asarray(fl["cad_mask"]) < 0.1
        cold = fm.surface_gust(51.0, cold_pool, fl, cfg)
        neutral = fm.surface_gust(51.0, ambient, fl, cfg)
        self.assertGreater(float(np.mean(cold["inversion_K"][core])),
                           float(np.mean(neutral["inversion_K"][core])))
        self.assertLess(float(np.mean(neutral["inversion_K"][outside])), 4.0)
        self.assertGreater(float(np.mean(neutral["mixing_depth_m"][outside])), 850.0)

    def test_horizontal_frontogenesis_remains_kinematic_separate_from_source(self):
        cfg = fm.Config(2.5, cad_enabled=True)
        theta = fm.initial_theta(40.0, cfg)
        fl = fm.flow(40.0, cfg)
        source = fm.theta_source(theta, cfg)
        f = fm.frontogenesis(theta, fl["u_ms"], fl["v_ms"])
        mask = fm.cad_mask(fm.X, fm.Y).astype(np.float64)
        ambient = fm._ambient_theta(fm.X, fm.Y, 0.0, cfg).astype(np.float64)
        target = ambient + mask * (fm.CAD_THETA_CORE_K - ambient)
        expected_source = mask * (target - theta.astype(np.float64)) / fm.CAD_RESTORE_TAU_S
        np.testing.assert_allclose(source, expected_source, rtol=2e-6, atol=1e-12)
        self.assertTrue(np.isfinite(f).all())


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
        # Keep the h48-78 cold-pool contrast requested for the scenario. This
        # is potential temperature only; it does not establish 2-m air
        # temperature or an exact freezing-temperature observation.
        if hours.size and np.any((hours >= 48.0) & (hours <= 78.0)):
            try:
                station_ids, sx, sy = fm._stations()
                atl_i, aby_i = station_ids.index("KATL"), station_ids.index("KABY")
                contrast_frames = np.where((hours >= 48.0) & (hours <= 78.0))[0]
                contrasts = []
                for frame in contrast_frames:
                    atl_theta = fm._sample(fields["theta_K"][frame],
                                           sx[[atl_i]], sy[[atl_i]])[0]
                    aby_theta = fm._sample(fields["theta_K"][frame],
                                           sx[[aby_i]], sy[[aby_i]])[0]
                    contrasts.append(float(aby_theta - atl_theta))
                if not contrasts or min(contrasts) < 10.0:
                    errors.append(f"{case}: h48-78 Atlanta-to-Albany potential-temperature contrast falls below 10 K")
            except (ValueError, IndexError, KeyError) as exc:
                errors.append(f"{case}: cannot verify h48-78 Atlanta/Albany theta contrast: {exc}")
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
    pressure_note = str(summary.get("pressure", ""))
    if "WP central-pressure parameter" not in pressure_note or "total minimum can shift" not in pressure_note:
        errors.append(f"{case}: pressure metadata does not distinguish central parameter from total grid pressure")
    fingerprints = [summary.get("source_fingerprint"), fields.get("source_fingerprint"),
                   peak.get("source_fingerprint"), stations.get("source_fingerprint")]
    if any(fp is None or str(np.asarray(fp).item()) != fm._source_fingerprint() for fp in fingerprints):
        errors.append(f"{case}: source fingerprints are missing or stale")

    canonical_min = float(summary.get("pressure_central_parameter_min_mb", np.inf))
    if abs(canonical_min - 939.0) > 1e-4:
        errors.append(f"{case}: canonical WP central-pressure parameter minimum is not 939 mb")
    grid_min = float(summary.get("pressure_grid_min_mb", np.inf))
    if not (930.0 <= grid_min <= 948.0):
        errors.append(f"{case}: actual total-pressure grid minimum {grid_min:.2f} mb is outside the stated plausible range")
    if not np.isclose(float(summary.get("pressure_min_mb", np.inf)), grid_min, atol=1e-6):
        errors.append(f"{case}: pressure_min_mb does not report the actual total-pressure grid minimum")
    if "p_mb" in fields and fields["p_mb"].size and np.isfinite(fields["p_mb"]).all():
        frame_min = float(np.min(fields["p_mb"]))
        if not (930.0 <= frame_min <= 948.0):
            errors.append(f"{case}: stored total-pressure frame minimum {frame_min:.2f} mb is outside the stated plausible range")
        elif abs(frame_min - grid_min) > 1.0:
            errors.append(f"{case}: stored pressure frames disagree with reported total-pressure grid minimum")
    cad_meta = summary.get("cad", {})
    source_note = str(cad_meta.get("thermal_source", ""))
    if "explicit Q" not in source_note or "excludes Q" not in source_note:
        errors.append(f"{case}: metadata does not distinguish thermal restoring source Q from horizontal F")
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
