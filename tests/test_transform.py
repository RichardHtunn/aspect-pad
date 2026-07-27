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

if __name__ == '__main__':
    unittest.main()