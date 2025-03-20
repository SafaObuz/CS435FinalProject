from PIL import Image, ImageFilter
import os
import argparse
import numpy as np


def add_gaussian_noise(image, mean=0, std=10):
    """
    Add Gaussian noise to an image.
    image: PIL Image.
    mean: Mean of the Gaussian noise.
    std: Standard deviation of the Gaussian noise.
    """
    np_image = np.array(image)
    noise = np.random.normal(mean, std, np_image.shape).astype(np.uint8)
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def add_jpeg_artifacts(image, quality=50):
    """
    Simulate JPEG compression artifacts.
    image: PIL Image.
    quality: JPEG compression quality (lower means more artifacts).
    """
    temp_path = "temp.jpg"
    image.save(temp_path, "JPEG", quality=quality)
    return Image.open(temp_path)


def downscale_images(hr_folder, lr_folder, scale_factor=2, blur_radius=1.0, noise_std=10, jpeg_quality=50):
    """
    Downscale HR images to generate degraded LR images with blur, noise, and compression artifacts.
    hr_folder: Path to the folder containing high-resolution images.
    lr_folder: Path to the folder where low-resolution images will be saved.
    scale_factor: Factor by which the image will be downscaled (e.g., 2 for 2x downscaling).
    blur_radius: Radius for Gaussian blur.
    noise_std: Standard deviation of Gaussian noise.
    jpeg_quality: JPEG compression quality (lower means more artifacts).
    """
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)
    
    hr_images = [f for f in os.listdir(hr_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
    
    for hr_image in hr_images:
        hr_image_path = os.path.join(hr_folder, hr_image)
        lr_image_path = os.path.join(lr_folder, hr_image)
        
        # Open the high-resolution image
        hr_img = Image.open(hr_image_path)
        
        # Apply Gaussian blur
        blurred_img = hr_img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Add Gaussian noise
        noisy_img = add_gaussian_noise(blurred_img, std=noise_std)
        
        # Downscale the image
        lr_img = noisy_img.resize((noisy_img.width // scale_factor, noisy_img.height // scale_factor), Image.BICUBIC)
        
        # Apply additional blur
        blurry_img = lr_img.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Add JPEG artifacts
        final_img = add_jpeg_artifacts(blurry_img, quality=jpeg_quality)
        
        # Save the final low-resolution image
        final_img.save(lr_image_path)
        
    print(f"Generated degraded LR images saved in {lr_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Downscale high-resolution images with blur, noise, and compression artifacts to generate low-quality images."
    )
    parser.add_argument("hr_folder", type=str, help="Path to the folder containing high-resolution images")
    parser.add_argument("lr_folder", type=str, help="Path to the folder where low-resolution images will be saved")
    parser.add_argument("--scale_factor", type=int, default=2, help="Downscaling factor (default: 2)")
    parser.add_argument("--blur_radius", type=float, default=1.0, help="Gaussian blur radius (default: 1.0)")
    parser.add_argument("--noise_std", type=int, default=10, help="Standard deviation of Gaussian noise (default: 10)")
    parser.add_argument("--jpeg_quality", type=int, default=50, help="JPEG compression quality (default: 50)")
    
    args = parser.parse_args()
    downscale_images(args.hr_folder, args.lr_folder, args.scale_factor, args.blur_radius, args.noise_std, args.jpeg_quality)


if __name__ == "__main__":
    main()