import csv
import hashlib
import json
import unittest

import numpy as np
from PIL import Image

from synthetic_cutlines import ROOT, bind_cutline_gauges, cutline_cd, load_model_api, synthetic_make_ib


class CutlineTests(unittest.TestCase):
    def setUp(self):
        # A triangular profile with threshold crossings at x=3 and x=7.
        self.image = np.tile(1 - np.abs(np.arange(11) - 5) / 4, (11, 1))

    def test_interpolated_and_exact_crossings(self):
        self.assertEqual(cutline_cd(self.image, 0, 5, 10, 5, .5), (4., 3., 7.))
        cd, left, right = cutline_cd(self.image, 0, 5, 10, 5, .6)
        np.testing.assert_allclose([cd, left, right], [3.2, 3.4, 6.6])

    def test_diagonal_and_reversed(self):
        cd, _, _ = cutline_cd(self.image, 0, 0, 10, 10, .5)
        self.assertAlmostEqual(cd, 4 * np.sqrt(2))
        self.assertEqual(cutline_cd(self.image, 10, 5, 0, 5, .5)[0], 4)

    def test_invalid_edges_and_geometry(self):
        for image, endpoints, threshold in (
            (np.zeros((11, 11)), (0, 5, 10, 5), .5),
            (np.tile([0., 1., 0., 1., 0.], (11, 1)), (0, 5, 4, 5), .5),
            (self.image, (-1, 5, 10, 5), .5),
            (self.image, (3, 5, 10, 5), .5),
            (self.image, (5, 5, 5, 5), .5),
            (np.tile([0., .5, .5, 1., 0.], (11, 1)), (0, 5, 4, 5), .5),
        ):
            with self.subTest(endpoints=endpoints, threshold=threshold):
                with self.assertRaises(ValueError):
                    cutline_cd(image, *endpoints, threshold)

    def test_residual_snapshot_and_invalid_gauge_id(self):
        gauges = [dict(gauge_id='G001', x0_px=0, y0_px=5, x1_px=10,
                       y1_px=5, threshold=.5, sample_step_px=.25, measured_cd_px=3.)]
        residuals = bind_cutline_gauges(gauges)
        gauges[0]['measured_cd_px'] = 100
        np.testing.assert_allclose(residuals(self.image), [1])
        with self.assertRaisesRegex(ValueError, 'G001'):
            residuals(np.zeros_like(self.image))

    def test_delivered_files_round_trip(self):
        folder = ROOT / 'data' / 'cutline_case'
        log = json.loads((folder / 'ideal_parameters.log').read_text())
        self.assertEqual(len(log['terms']), 50)
        self.assertEqual(hashlib.sha256((folder / 'input_image.png').read_bytes()).hexdigest(),
                         log['input_sha256'])
        image = np.array(Image.open(folder / 'input_image.png'), dtype=float) / 65535
        self.assertEqual(image.shape, (320, 320))
        with (folder / 'gauges.csv').open(newline='') as file:
            gauges = list(csv.DictReader(file))
        self.assertEqual(len(gauges), 4000)
        self.assertEqual(log['gauge_count'], 4000)
        self.assertEqual(len({g['gauge_id'] for g in gauges}), 4000)
        self.assertEqual(len({tuple(g[k] for k in ('x0_px', 'y0_px', 'x1_px', 'y1_px'))
                              for g in gauges}), 4000)
        for g in gauges:
            self.assertEqual(float(g['threshold']), log['threshold'])
            distance = np.hypot(float(g['edge1_x_px']) - float(g['edge0_x_px']),
                                float(g['edge1_y_px']) - float(g['edge0_y_px']))
            self.assertAlmostEqual(distance, float(g['measured_cd_px']), places=11)
        api = load_model_api()
        terms = [api['Term'](**{k: v for k, v in t.items() if k != 'term_id'}) for t in log['terms']]
        model = api['build'](terms, make_Ib=synthetic_make_ib)
        prediction = model(image)
        np.testing.assert_allclose(prediction, np.load(folder / 'ideal_response.npy'), atol=1e-14)
        objective = model.make_objective(image, bind_cutline_gauges(gauges))
        self.assertLess(objective(model.weights), 1e-10)
        self.assertGreater(objective(model.weights * 1.02), 1e-5)


if __name__ == '__main__':
    unittest.main()
