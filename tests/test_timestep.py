import unittest

from utils.timestep import is_zero_timestep


class FakeScalar:
    def __init__(self, value):
        self.value = value
        self.item_calls = 0

    def item(self):
        self.item_calls += 1
        return self.value


class IsZeroTimestepTest(unittest.TestCase):
    def test_python_scalars(self):
        self.assertTrue(is_zero_timestep(0))
        self.assertTrue(is_zero_timestep(0.0))
        self.assertFalse(is_zero_timestep(1))
        self.assertFalse(is_zero_timestep(0.5))

    def test_tensor_like_scalar_is_extracted_once(self):
        timestep = FakeScalar(0)

        self.assertTrue(is_zero_timestep(timestep))
        self.assertEqual(timestep.item_calls, 1)

    def test_nonzero_tensor_like_scalar(self):
        timestep = FakeScalar(999.0)

        self.assertFalse(is_zero_timestep(timestep))
        self.assertEqual(timestep.item_calls, 1)


if __name__ == "__main__":
    unittest.main()
