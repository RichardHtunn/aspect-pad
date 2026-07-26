import unittest
from PIL import Image
from aspect_pad import AspectPad

class TestAspectPad(unittest.TestCase):
    
    def test_wide_image_padding(self):
        """Test if a wide rectangle (800x400) properly pads into a 512x512 square."""
        raw_image = Image.new('RGB', (800, 400))
        padder = AspectPad(target_size=512)
        processed_image = padder(raw_image)
        
        self.assertEqual(processed_image.size, (512, 512))

    def test_tall_image_padding(self):
        """Test if a tall rectangle (400x800) properly pads into a 224x224 square."""
        raw_image = Image.new('RGB', (400, 800))
        padder = AspectPad(target_size=224)
        processed_image = padder(raw_image)
        
        self.assertEqual(processed_image.size, (224, 224))
        
    def test_custom_tuple_target(self):
        """Test if the class accepts a custom rectangular target like (1920, 1080)."""
        raw_image = Image.new('RGB', (500, 500))
        padder = AspectPad(target_size=(1920, 1080))
        processed_image = padder(raw_image)
        
        self.assertEqual(processed_image.size, (1920, 1080))

if __name__ == '__main__':
    unittest.main()