import torch
import torch.nn.functional as F

class AspectPad:
    """
    A PyTorch native transform that scales a Tensor image while preserving 
    its aspect ratio, padding the remaining space to fit a target size.
    """
    def __init__(self, target_size, fill=0):
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        self.fill = fill

    def __call__(self, tensor_img):
        if not isinstance(tensor_img, torch.Tensor):
            raise TypeError(f"AspectPad expects a torch.Tensor. Please use ToTensor() first. Got {type(tensor_img)}.")

        original_dim = tensor_img.dim()
        if original_dim == 3:
            tensor_img = tensor_img.unsqueeze(0) 
        elif original_dim != 4:
            raise ValueError(f"Expected a 3D or 4D tensor, but got {original_dim}D.")

        _, _, h, w = tensor_img.shape
        target_h, target_w = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        original_dtype = tensor_img.dtype
        tensor_img = tensor_img.to(torch.float32)
        
        resized = F.interpolate(tensor_img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized = resized.to(original_dtype)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=self.fill)

        if original_dim == 3:
            padded = padded.squeeze(0)

        return padded