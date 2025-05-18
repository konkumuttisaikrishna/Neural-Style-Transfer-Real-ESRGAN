import os
import subprocess

def upscale_image(input_path, output_path, scale=4):
    """
    Upscale an image using Real-ESRGAN.

    Args:
        input_path (str): Path to input image.
        output_path (str): Path to save upscaled image.
        scale (int): Upscaling factor (typically 2, 4, or 8).
    """
    # Make sure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", input_path,
        "--outscale", str(scale),
        "--output", output_path
    ]

    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Real-ESRGAN failed: {e}")
        return None
