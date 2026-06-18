import cv2
import numpy as np
import random
from PIL import Image, ImageFilter

class DeepfakeAugmentation:
    """
    Data augmentation techniques specifically for deepfake detection.
    Simulates real-world "in-the-wild" distortions.
    """
    
    @staticmethod
    def apply_compression(image: np.ndarray, quality: int = 20) -> np.ndarray:
        """Simulates JPEG/Video compression artifacts."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    @staticmethod
    def apply_blur(image: np.ndarray, radius: float = 2.0) -> np.ndarray:
        """Simulates camera defocus or motion blur."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def add_noise(image: np.ndarray, level: float = 0.1) -> np.ndarray:
        """Adds Gaussian noise to simulate sensor grain."""
        noise = np.random.normal(0, 255 * level, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Applies a random sequence of forensic augmentations."""
        if random.random() > 0.5:
            image = self.apply_compression(image, random.randint(10, 50))
        if random.random() > 0.5:
            image = self.apply_blur(image, random.uniform(0.5, 3.0))
        if random.random() > 0.3:
            image = self.add_noise(image, random.uniform(0.01, 0.1))
        return image
