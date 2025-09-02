import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

class RetinalImagePreprocessor:
    """
    Advanced preprocessing for retinal images based on medical image analysis best practices
    Implements techniques from research papers achieving high accuracy in DR classification
    """

    def __init__(self, image_size=224):
        self.image_size = image_size

    def remove_black_background(self, image):
        """Remove black background and crop to retina region"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Create mask for non-black pixels
        mask = np.any(img_array > 30, axis=2) if len(img_array.shape) == 3 else img_array > 30

        # Find bounding box of non-black region
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return image  # Return original if no non-black pixels found

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add small padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(img_array.shape[0], y_max + padding)
        x_max = min(img_array.shape[1], x_max + padding)

        # Crop to bounding box
        cropped = img_array[y_min:y_max, x_min:x_max]

        return Image.fromarray(cropped) if isinstance(image, Image.Image) else cropped

    def ben_graham_preprocessing(self, image, sigma=30):
        """
        Ben Graham's preprocessing method for retinal images
        Subtracts local average color and clips intensities
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32)
        else:
            img_array = image.astype(np.float32)

        # Apply Gaussian blur to get local average
        blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)

        # Subtract local average and add 50% gray
        processed = img_array - blurred + 128

        # Clip values to valid range
        processed = np.clip(processed, 0, 255)

        return Image.fromarray(processed.astype(np.uint8)) if isinstance(image, Image.Image) else processed.astype(np.uint8)

    def clahe_enhancement(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced) if isinstance(image, Image.Image) else enhanced

    def normalize_illumination(self, image):
        """Normalize illumination using morphological operations"""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))

        # Apply morphological opening (erosion + dilation)
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Normalize each channel
        normalized = np.zeros_like(img_array)
        for i in range(3):
            channel = img_array[:, :, i].astype(np.float32)
            # Avoid division by zero
            background_safe = np.where(background == 0, 1, background).astype(np.float32)
            normalized[:, :, i] = np.clip((channel / background_safe) * 128, 0, 255)

        return Image.fromarray(normalized.astype(np.uint8)) if isinstance(image, Image.Image) else normalized.astype(np.uint8)

    def preprocess_pipeline(self, image, apply_ben_graham=True, apply_clahe=True, 
                          apply_illumination_norm=False):
        """Complete preprocessing pipeline"""

        # Step 1: Remove black background and crop
        processed = self.remove_black_background(image)

        # Step 2: Apply Ben Graham preprocessing if specified
        if apply_ben_graham:
            processed = self.ben_graham_preprocessing(processed)

        # Step 3: Apply CLAHE enhancement if specified
        if apply_clahe:
            processed = self.clahe_enhancement(processed)

        # Step 4: Apply illumination normalization if specified
        if apply_illumination_norm:
            processed = self.normalize_illumination(processed)

        # Step 5: Resize to target size
        if isinstance(processed, Image.Image):
            processed = processed.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        else:
            processed = cv2.resize(processed, (self.image_size, self.image_size))
            processed = Image.fromarray(processed)

        return processed
