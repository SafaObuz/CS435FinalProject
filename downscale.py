from PIL import Image
import os
import argparse

def downscale_images(hr_folder, lr_folder, scale_factor=2):
    """
    Downscale HR images to generate LR images.
    hr_folder: Path to the folder containing high-resolution images.
    lr_folder: Path to the folder where low-resolution images will be saved.
    scale_factor: Factor by which the image will be downscaled (e.g., 2 for 2x downscaling).
    """
    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)
    
    hr_images = [f for f in os.listdir(hr_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for hr_image in hr_images:
        hr_image_path = os.path.join(hr_folder, hr_image)
        lr_image_path = os.path.join(lr_folder, hr_image)
        
        # Open the high-resolution image
        hr_img = Image.open(hr_image_path)
        
        # Downscale the image
        lr_img = hr_img.resize((hr_img.width // scale_factor, hr_img.height // scale_factor), Image.BICUBIC)
        
        # Save the low-resolution image
        lr_img.save(lr_image_path)
        
    print(f"Generated LR images saved in {lr_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Downscale high-resolution images to generate low-resolution images."
    )
    parser.add_argument("hr_folder", type=str, help="Path to the folder containing high-resolution images")
    parser.add_argument("lr_folder", type=str, help="Path to the folder where low-resolution images will be saved")
    parser.add_argument("--scale_factor", type=int, default=2, help="Downscaling factor (default: 2)")
    
    args = parser.parse_args()
    downscale_images(args.hr_folder, args.lr_folder, args.scale_factor)

if __name__ == '__main__':
    main()
