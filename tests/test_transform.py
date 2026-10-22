import unittest
import torch
from aspect_pad import AspectPad

class TestAspectPad(unittest.TestCase):
    def test_initialization(self):
        """Test that the target size correctly handles both ints and tuples."""
        padder_square = AspectPad(512)
        self.assertEqual(padder_square.target_size, (512, 512))

        padder_rect = AspectPad((256, 512))
        self.assertEqual(padder_rect.target_size, (256, 512))

    def test_invalid_target_size(self):
        for size in (0, -1, True, (0, 10), (10, -1), (10,), (10, 10, 10), 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                AspectPad(size)

    def test_single_image_tensor(self):
        """Test processing a single 3D tensor [C, H, W]."""
        tensor = torch.ones(3, 1080, 1920)
        padder = AspectPad(512)
        result = padder(tensor)
        
        self.assertEqual(result.shape, (3, 512, 512))
        self.assertIsInstance(result, torch.Tensor)

    def test_batch_image_tensor(self):
        """Test processing a 4D batch of tensors [B, C, H, W]."""
        batch_tensor = torch.ones(16, 3, 1080, 1920)
        padder = AspectPad((512, 512))
        result = padder(batch_tensor)
        
        self.assertEqual(result.shape, (16, 3, 512, 512))
        self.assertIsInstance(result, torch.Tensor)

    def test_invalid_input_type(self):
        """Test that the package explicitly rejects non-tensors (like PIL images)."""
        padder = AspectPad(512)
        with self.assertRaises(TypeError):
            padder("this is a string, not a tensor")
            
    def test_invalid_tensor_dimensions(self):
        """Test that the package rejects improperly formatted tensors."""
        padder = AspectPad(512)
        tensor_2d = torch.ones(100, 100)
        with self.assertRaises(ValueError):
            padder(tensor_2d)

    def test_empty_image(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            AspectPad(8)(torch.empty(3, 0, 5))

    def test_empty_batch_or_channels(self):
        for shape in ((0, 3, 4, 4), (1, 0, 4, 4)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "positive"):
                AspectPad(8)(torch.empty(shape))

    def test_centered_padding_and_fill(self):
        image = torch.ones(1, 2, 4)
        result = AspectPad((4, 4), fill=7)(image)
        self.assertEqual(result.shape, (1, 4, 4))
        self.assertTrue(torch.equal(result[:, 1:3, :], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(result[:, 0, :], torch.full((1, 4), 7.0)))
        self.assertTrue(torch.equal(result[:, 3, :], torch.full((1, 4), 7.0)))

    def test_uint8_dtype_and_batch(self):
        images = torch.full((2, 3, 2, 4), 12, dtype=torch.uint8)
        result = AspectPad(4, fill=128)(images)
        self.assertEqual(result.dtype, torch.uint8)
        self.assertEqual(result.shape, (2, 3, 4, 4))
        self.assertTrue(torch.equal(result[:, :, 1:3, :], images))
        self.assertTrue(torch.all(result[:, :, 0, :] == 128))

    def test_extreme_aspect_ratio(self):
        result = AspectPad(3)(torch.ones(1, 1, 1000))
        self.assertEqual(result.shape, (1, 3, 3))

    def test_limiting_axis_reaches_target_without_rounding_loss(self):
        for h, w, target_h, target_w, expected_content in (
            (243, 179, 116, 15, (20, 15)),
            (46, 181, 13, 189, (13, 51)),
        ):
            with self.subTest(input_hw=(h, w), target_hw=(target_h, target_w)):
                output = AspectPad((target_h, target_w), fill=-17)(torch.ones(1, h, w))
                rows, columns = torch.where(output[0] != -17)
                self.assertEqual((rows.max().item() - rows.min().item() + 1,
                                  columns.max().item() - columns.min().item() + 1), expected_content)

    def test_float64_precision(self):
        image = torch.tensor([[[1.0, 1.0 + 1e-9]]], dtype=torch.float64)
        result = AspectPad((2, 4))(image)
        self.assertEqual(result.dtype, torch.float64)
        self.assertGreater(result[0, 0, 1].item(), 1.0)

if __name__ == '__main__':
    unittest.main()
