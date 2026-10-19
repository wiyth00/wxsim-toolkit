"""Meaningful physical/ensemble/cache invariants for the fictional scenario."""
import json
import os
import unittest
import numpy as np
import stormmodel as sm
from stations import station_reports
from ensemble import generate_members


class PhysicsChecks(unittest.TestCase):
    def test_canon(self):
        for t,lo,la,p in sm.WP:
            self.assertAlmostEqual(sm.pressure(t),p)
            np.testing.assert_allclose(sm.center_ll(t),(lo,la),atol=1e-10)
        self.assertAlmostEqual(sm.pressure(51),939.)
        self.assertAlmostEqual(sm.pressure(54),941.)

    def test_stability_reduces_surface_gust(self):
        neutral,mean,c0 = sm.mixing_gust(40.,0.,1100.)
        stable,_,c1 = sm.mixing_gust(40.,7.,180.)
        self.assertLess(stable,neutral*.6)
        self.assertLess(c1,c0)
        self.assertGreaterEqual(neutral,mean)
        gusts = [sm.mixing_gust(40.,inv,500.)[0] for inv in [0.,1.,3.,7.]]
        self.assertTrue(np.all(np.diff(gusts)<0))
        gusts = [sm.mixing_gust(40.,2.,depth)[0] for depth in [100.,300.,600.,900.]]
        self.assertTrue(np.all(np.diff(gusts)>0))

    def test_radii_order_growth_and_seam(self):
        for t in np.arange(12,133,3):
            radii = sm.quadrant_radii(t)
            active = radii[np.any(radii>0,axis=1)]
            self.assertTrue(np.all(np.diff(active,axis=0)<0))
        self.assertTrue(np.all(sm.quadrant_radii(96)[0]>sm.quadrant_radii(51)[0]))
        r = sm.quadrant_radii(51)[0]
        self.assertAlmostEqual(float(sm.angular_radius(r,-np.pi+1e-7)),float(sm.angular_radius(r,np.pi-1e-7)),places=4)
        self.assertGreater(np.ptp(r),100.)

    def test_deterministic_call_order_and_sting(self):
        a = sm.inst_gust(59)
        sm.inst_gust(78)
        b = sm.inst_gust(59)
        np.testing.assert_array_equal(a,b)
        f = sm.instant_fields(59)
        mask = f['sting']>.7
        self.assertTrue(mask.any())
        self.assertGreater(float(np.median(f['coupling'][mask])),.8)
        self.assertGreater(float(f['gust'][mask].max()),100.)
        self.assertLess(float(sm.sting_weight(70).max()),.001)
        p0 = np.unravel_index(sm.sting_weight(58).argmax(),sm.X.shape)
        p1 = np.unravel_index(sm.sting_weight(60).argmax(),sm.X.shape)
        self.assertGreater(sm.X[p1],sm.X[p0]); self.assertLess(sm.Y[p1],sm.Y[p0])

    def test_convective_width_and_motion(self):
        birth,x,y,life,width,peak,tilt = sm.convective_cells(sm.SEED)[0]
        self.assertGreaterEqual(width,5.); self.assertLessEqual(width,15.)
        a,_ = sm.convective_gust(44)
        b,_ = sm.convective_gust(47)
        self.assertGreater(float(a.max()),58.)
        self.assertGreater(float(b.max()),58.)
        # Gust-weighted center advances east, not a stationary smooth band.
        for field in (a,b): self.assertTrue(np.isfinite(field).all())
        self.assertGreater(float((sm.X*b).sum()/b.sum()),float((sm.X*a).sum()/a.sum()))

    def test_cache_and_peak_consistency(self):
        self.assertTrue(sm.cache_valid(sm.PEAK)); self.assertTrue(sm.cache_valid(sm.FRAMES))
        self.assertTrue(np.isfinite(sm.G).all()); self.assertTrue((sm.G>=5).all())
        self.assertTrue(((sm.T>=sm.START)&(sm.T<=sm.END)).all())
        with np.load(sm.FRAMES) as data:
            self.assertEqual(len(data['ft']),165)
            self.assertLessEqual(float((data['F'].astype(np.float32).max(axis=0)-sm.G).max()),.07)
        self.assertTrue(np.isnan(sm.sample(sm.G,-120,40)))

    def test_stations_outages_and_mean(self):
        rows = station_reports()
        self.assertEqual(len(rows),69)
        self.assertEqual(rows,station_reports())
        outages = [r for r in rows if r['outage']]
        self.assertGreater(len(outages),0)
        for r in rows:
            self.assertLessEqual(r['reported_gust_mph'],r['synthetic_full_gust_mph'])
            self.assertGreaterEqual(r['reported_gust_mph'],r['reported_mean_mph'])
            if r['outage']:
                self.assertFalse(r['outage_start_h']<=r['reported_peak_h']<=r['outage_end_h'])

    def test_ensemble_consistency(self):
        path = os.path.join(sm.HERE,'ensemble.npz')
        self.assertTrue(os.path.exists(path),'Run ensemble.py first')
        with np.load(path) as e:
            self.assertEqual(str(e['model_signature']),sm.signature())
            p=e['member_peak']
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(e['p58'],(p>=58).mean(axis=0)*100)
            np.testing.assert_allclose(e['p74'],(p>=74).mean(axis=0)*100)
            self.assertTrue((e['p74']<=e['p58']).all())
            np.testing.assert_allclose(e['mean_peak'],p.mean(axis=0))
            np.testing.assert_allclose(e['max_peak'],p.max(axis=0))
            self.assertTrue((e['p10']<=e['p90']).all())
            self.assertEqual(len(json.loads(str(e['members_json']))),int(e['n_members']))
        self.assertEqual(generate_members(3),generate_members(3))
        self.assertEqual(len({x.seed for x in generate_members(20)}),20)

if __name__=='__main__': unittest.main(verbosity=2)
