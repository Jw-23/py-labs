"""Run with: .venv/bin/python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import unittest

import numpy as np

namespace = {'__name__': __name__}
notebook = json.loads((Path(__file__).resolve().parents[1] / '黑盒优化.ipynb').read_text())
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and not cell['id'].startswith('synthetic-data-'):
        exec(compile(''.join(cell['source']), f"notebook:{cell['id']}", 'exec'), namespace)
Term, build = namespace['Term'], namespace['build']


class ResistTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.image = np.arange(1., 7.).reshape(2, 3)
        self.terms = [Term('positive', 0., 0, 1, 1., 3., 2.),
                      Term('negtive', 1., 0, 1, 1., 3., -0.5)]

        def make_ib(image, b, sign):
            self.calls.append(sign)
            return image + b

        self.model = build(self.terms, make_Ib=make_ib,
                           make_G=lambda s, p: np.ones((1, 1)))

    def test_weighted_sum_and_candidate(self):
        expected = 2 * self.image - .5 * (self.image + 1)
        np.testing.assert_allclose(self.model(self.image), expected)
        np.testing.assert_allclose(self.model(self.image, weights=[0, 1]), self.image + 1)
        np.testing.assert_array_equal(self.model.weights, [2, -.5])
        self.assertIn('negtive', self.calls)
        single = build(self.terms[:1], make_Ib=lambda a, b, sign: a,
                       make_G=lambda s, p: np.ones((1, 1)))
        np.testing.assert_allclose(single(self.image), 2 * self.image)

    def test_updates_and_copies(self):
        snapshot = self.model.terms
        weights = self.model.weights
        weights[:] = 9
        self.terms.clear()
        self.model.update_term(0, b=2., c=.25)
        self.assertEqual(snapshot[0].b, 0)
        self.assertEqual(self.model.terms[1], snapshot[1])
        fresh = build(self.model.terms, make_Ib=lambda a, b, sign: a + b,
                      make_G=lambda s, p: np.ones((1, 1)))
        np.testing.assert_allclose(self.model(self.image), fresh(self.image))
        self.model.set_weights([-1, 3])
        np.testing.assert_array_equal(self.model.weights, [-1, 3])
        self.assertEqual(self.model.terms[0].b, 2)

    def test_atomic_invalid_updates(self):
        before = self.model.terms
        for changes in ({'s': 0}, {'n': 0}, {'k': -1}, {'k': .5},
                        {'p': 0}, {'b': np.inf}, {'sign': 'bad'},
                        {'c': np.nan}, {'unknown': 1}, {'c': 3, 'n': 0}):
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    self.model.update_term(0, **changes)
                self.assertEqual(self.model.terms, before)
        for index in (-1, 2):
            with self.assertRaises(IndexError):
                self.model.update_term(index, c=0)
        for weights in ([1], [[1, 2]], [1, np.inf], [1j, 2]):
            with self.assertRaises(ValueError):
                self.model.set_weights(weights)
            self.assertEqual(self.model.terms, before)

    def test_objective_rms_cache_and_snapshot(self):
        image = self.image.copy()
        objective = self.model.make_objective(
            image, lambda prediction: prediction[0, :2] - [1, 0])
        self.assertEqual(len(self.calls), 2)
        # [1, 0] predicts [1, 2] -> residuals [0, 2].
        self.assertAlmostEqual(objective([1, 0]), np.sqrt(2))
        self.assertAlmostEqual(objective([0, 1]), np.sqrt(5))
        self.assertEqual(len(self.calls), 2)
        np.testing.assert_array_equal(self.model.weights, [2, -.5])
        image[:] = 100
        self.model.update_term(0, b=20, s=2, c=8)
        self.model.set_weights([0, 0])
        self.assertAlmostEqual(objective([1, 0]), np.sqrt(2))
        self.assertEqual(len(self.calls), 2)
        replacement = self.model.make_objective(
            self.image, lambda prediction: prediction[0, :2] - [1, 0])
        self.assertNotEqual(replacement([1, 0]), objective([1, 0]))
        with self.assertRaises(ValueError):
            objective([1])

    def test_bad_gauges_and_stable_rms(self):
        for residuals in ([], [[1]], [np.nan], [np.inf], [1j]):
            objective = self.model.make_objective(self.image, lambda p: residuals)
            with self.assertRaises(ValueError):
                objective([1, 0])
        def broken(prediction):
            raise RuntimeError('gauge failed')
        with self.assertRaisesRegex(RuntimeError, 'gauge failed'):
            self.model.make_objective(self.image, broken)([1, 0])
        for value in (0., 1e300):
            objective = self.model.make_objective(self.image, lambda p: [value, value])
            self.assertEqual(objective([1, 0]), value)

    def test_empty_and_image_validation(self):
        empty = build([], make_Ib=lambda a, b, sign: a)
        np.testing.assert_array_equal(empty(self.image), np.zeros_like(self.image))
        with self.assertRaises(ValueError):
            empty.make_objective(self.image, lambda a: [0.])
        for image in (np.ones(3), np.ones((2, 3), dtype=int), np.empty((0, 2))):
            with self.assertRaises(ValueError):
                self.model(image)
        self.model.update_term(0, k=1)
        with self.assertRaises(ValueError):
            self.model(np.ones((1, 3)))
        self.assertEqual(self.model(self.image).shape, self.image.shape)

    def test_direct_term_validation(self):
        for field in ('n', 's', 'p'):
            params = dict(image=self.image, sign='positive', b=0., k=0, n=1,
                          s=1., p=3., make_Ib=lambda a, b, sign: a,
                          make_G=namespace['guass_kernel'])
            params[field] = 0
            with self.assertRaises(ValueError):
                namespace['resist_term'](**params)
        kernel = namespace['guass_kernel'](1., 3.)
        self.assertAlmostEqual(kernel.sum(), 1.)
        self.assertEqual(kernel.shape, (7, 7))


if __name__ == '__main__':
    unittest.main()
