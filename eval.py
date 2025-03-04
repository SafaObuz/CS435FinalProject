import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import math
import os
import argparse
from DAT import DAT  # Import your model

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and upscale images using the DAT model")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset (low-resolution images)')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./upscale', help='Directory to save upscaled images')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of images for testing')
    parser.add_argument('--resolution', type=str, choices=['1080p', '2k', '4k'], default='2k', help='Choose output resolution (1080p, 2k, or 4k)')
    return parser.parse_args()

# PSNR Calculation Function (Optional for evaluation)
def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * math.log10(1.0 / math.sqrt(mse))

class DIV2KEvalDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, transform=None, limit=None):
        self.lr_dir = lr_dir
        self.transform = transform
        self.lr_images = sorted(os.listdir(lr_dir))

        # Apply limit if specified
        if limit:
            self.lr_images = self.lr_images[:limit]

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        
        try:
            lr_image = Image.open(lr_image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {lr_image_path}: {e}")
            return None  # Skip this image

        if self.transform:
            lr_image = self.transform(lr_image)

        return lr_image, self.lr_images[idx]  # Return filename for saving


def main():
    # Parse arguments
    torch.cuda.empty_cache()
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set output resolution
    if args.resolution == '1080p':
        output_resolution = (1920, 1080)
    elif args.resolution == '2k':
        output_resolution = (2048, 1080)
    elif args.resolution == '4k':
        output_resolution = (3840, 2160)

    # Image transformation (ensure consistent input format)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load test data
    dataset = DIV2KEvalDataset(args.dataset_path, transform, limit=args.limit)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAT(upsample_steps=3).to(device)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.eval()

    # Evaluation and saving upscaled images
    with torch.no_grad():
        total_psnr = 0.0
        total_images = 0
        
        for lr_images, filenames in test_loader:
            if lr_images is None:  # Skip None images due to errors
                continue

            lr_images = lr_images.to(device)

            with torch.cuda.amp.autocast():
                sr_images = model(lr_images)
            sr_images = sr_images.clamp(-1, 1)  # Ensure valid pixel values

            # Compute PSNR for each image
            for i in range(len(filenames)):
                img = sr_images[i].cpu().permute(1, 2, 0).numpy()  # Convert to NumPy
                img = ((img + 1) / 2 * 255).astype('uint8')  # Denormalize and scale

                # Convert to PIL Image for resizing
                img_pil = Image.fromarray(img)

                # Resize the image to desired output resolution
                img_pil = img_pil.resize(output_resolution, Image.BICUBIC)

                # Ensure filename has extension (if not, add it)
                if not filenames[i].lower().endswith(('.png', '.jpg', '.jpeg')):
                    filenames[i] += '.png'

                img_pil.save(os.path.join(args.save_dir, filenames[i]))  # Save image

                # Optionally calculate PSNR for evaluation
                # Here you can provide target HR images if available
                # Example: total_psnr += psnr(sr_images[i], target_hr_image)  # Assume you have target HR images
                total_images += 1

        # Print PSNR if target images are available
        # Example: print(f"Average PSNR: {total_psnr / total_images} dB")

    print(f"Upscaled images saved to {args.save_dir}")

if __name__ == '__main__':
    main()
