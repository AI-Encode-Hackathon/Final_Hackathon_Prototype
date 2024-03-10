from PIL import Image
import numpy as np

def create_image_embedding(file):
    image = Image.open(file)
    image = image.resize((64, 64))
    image_array = np.array(image)

    # Ensure that the image array has a consistent size
    flattened_image = image_array.flatten()
    flattened_image_size = np.prod((64, 64)) * image_array.shape[-1]

    if len(flattened_image) != flattened_image_size:
        raise ValueError(f"Flattened image size mismatch: expected {flattened_image_size}, got {len(flattened_image)}")

    return flattened_image
