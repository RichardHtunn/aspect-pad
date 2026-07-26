from PIL import Image, ImageOps
from typing import Union, Tuple

class AspectPad:
    """
    A PyTorch-compatible transform that resizes an image while maintaining its aspect ratio,
    and mathematically pads it to perfectly fit a target square or rectangular size.
    """
    def __init__(self, target_size: Union[int, Tuple[int, int]], fill: Union[int, Tuple[int, int, int]] = 0):
        if isinstance(target_size, int):
            self.target_w = target_size
            self.target_h = target_size
        else:
            self.target_w, self.target_h = target_size
            
        self.fill = fill 

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size

        scale = min(self.target_w / w, self.target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        delta_w = self.target_w - new_w
        delta_h = self.target_h - new_h
        
        # divide the blank space perfectly in half for top/bottom and left/right
        padding = (
            delta_w // 2,             # left
            delta_h // 2,             # top
            delta_w - (delta_w // 2), # right
            delta_h - (delta_h // 2)  # bottom
        )
        
        return ImageOps.expand(img, padding, fill=self.fill)
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_size={(self.target_w, self.target_h)}, fill={self.fill})"