import unittest
import numpy as np

from post_process import find_centroid_and_orientation

class TestStudent(unittest.TestCase):

    def test_centroid_and_orientation(self):
        mask = np.zeros((300, 300), dtype=np.uint8)
        for i in range(100, 150):
            for j in range(100, 200):
                mask[i][j] = 255

        angle, centroid = find_centroid_and_orientation(mask)

        self.assertEqual(angle, 0. or 180.)
        self.assertEqual(centroid, (149.5, 124.5))

if __name__ == '__main__':
    unittest.main()