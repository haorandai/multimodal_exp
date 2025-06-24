import os
from PIL import Image

def create_dummy_image(output_path: str, size: tuple = (224, 224), color: tuple = (128, 128, 128)) -> str:
    """
    Create a dummy image with specified size and color.
    
    Args:
        output_path (str): Full path where the image will be saved
        size (tuple): Image dimensions as (width, height). Default: (224, 224)
        color (tuple): RGB color values as (R, G, B). Default: (128, 128, 128) - gray
    
    Returns:
        str: The basename of the saved image file
    """
    image = Image.new('RGB', size, color)
    
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    image.save(output_path)
    
    return os.path.basename(output_path)


if __name__ == "__main__":
    output_path = "images/dummy_image.jpg"
    create_dummy_image(output_path)