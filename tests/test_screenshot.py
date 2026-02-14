import sys
from unittest.mock import MagicMock
import unittest
import numpy as np

# Mock heavy dependencies before importing openrecall.screenshot
sys.modules["mss"] = MagicMock()
sys.modules["doctr"] = MagicMock()
sys.modules["doctr.models"] = MagicMock()
# Mock internal modules that import heavy stuff
sys.modules["openrecall.ocr"] = MagicMock()
sys.modules["openrecall.nlp"] = MagicMock()
sys.modules["openrecall.database"] = MagicMock()
# Mock config to avoid file system access or missing config
sys.modules["openrecall.config"] = MagicMock()
sys.modules["openrecall.config"].args = MagicMock()
sys.modules["openrecall.config"].args.primary_monitor_only = False

# Now we can safely import the functions we want to test
from openrecall.screenshot import is_similar, mean_structured_similarity_index

class TestScreenshot(unittest.TestCase):
    def test_ssim_identical(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ssim = mean_structured_similarity_index(img, img)
        self.assertAlmostEqual(ssim, 1.0, places=4)

    def test_ssim_different(self):
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        ssim = mean_structured_similarity_index(img1, img2)
        # SSIM should be very low for completely different images
        self.assertLess(ssim, 0.1)

    def test_is_similar_identical(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.assertTrue(is_similar(img, img))

    def test_is_similar_different(self):
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.assertFalse(is_similar(img1, img2))

    def test_is_similar_small_change(self):
        # 1 pixel change should result in high similarity
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        img2[50, 50] = 255
        ssim = mean_structured_similarity_index(img1, img2)
        # print(f"DEBUG: SSIM for small change: {ssim}")
        self.assertTrue(is_similar(img1, img2))

    def test_is_similar_large_image(self):
        # 4000x2000 image to trigger downsampling
        img1 = np.random.randint(0, 256, (2000, 4000, 3), dtype=np.uint8)
        # Identical
        self.assertTrue(is_similar(img1, img1))

        # Different
        img2 = np.random.randint(0, 256, (2000, 4000, 3), dtype=np.uint8)
        self.assertFalse(is_similar(img1, img2))

        # Small change (100x100 box) - should still be similar
        img3 = img1.copy()
        img3[500:600, 500:600] = 0
        self.assertTrue(is_similar(img1, img3))

if __name__ == '__main__':
    unittest.main()
