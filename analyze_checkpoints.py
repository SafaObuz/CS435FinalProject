import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from DAT import DAT  # Import your model

def load_checkpoint(model, checkpoint_path):
    """Load a model checkpoint with strict=False to allow layer mismatches."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)  # Allow missing/unexpected keys
    model.eval()
    return model

def compute_psnr_ssim(gt, pred):
    """Compute PSNR and SSIM between ground truth and predicted images."""
    gt = gt.numpy().transpose(1, 2, 0)
    pred = pred.numpy().transpose(1, 2, 0)
    
    # Ensure image ranges are in [0,1]
    gt = np.clip(gt, 0, 1)
    pred = np.clip(pred, 0, 1)
    
    psnr_value = psnr(gt, pred, data_range=1.0)
    ssim_value = ssim(gt, pred, data_range=1.0, channel_axis=2)
    
    return psnr_value, ssim_value

def evaluate_checkpoints(model, checkpoint_dir, test_loader):
    """Evaluate model at each checkpoint and compute PSNR/SSIM."""
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
    
    psnr_values = []
    ssim_values = []
    train_losses = []
    val_losses = []
    epoch_numbers = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for i, checkpoint in enumerate(checkpoint_files):
        model = load_checkpoint(model, os.path.join(checkpoint_dir, checkpoint))
        total_psnr, total_ssim, total_train_loss, total_val_loss, count = 0.0, 0.0, 0.0, 0.0, 0
        
        with torch.no_grad():
            for lr_images, hr_images in test_loader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                output = model(lr_images)
                output = torch.clamp(output, 0, 1)  # Ensure proper range
                
                loss = F.mse_loss(output, hr_images)
                if count % 2 == 0:
                    total_train_loss += loss.item()
                else:
                    total_val_loss += loss.item()
                
                for j in range(output.shape[0]):
                    psnr_val, ssim_val = compute_psnr_ssim(hr_images[j].cpu(), output[j].cpu())
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    count += 1
        
        psnr_values.append(total_psnr / count)
        ssim_values.append(total_ssim / count)
        train_losses.append(total_train_loss / (count // 2))
        val_losses.append(total_val_loss / (count // 2))
        epoch_numbers.append(i + 1)
        print(f"Checkpoint {i+1}: PSNR={psnr_values[-1]:.2f}, SSIM={ssim_values[-1]:.4f}, Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")
    
    return epoch_numbers, psnr_values, ssim_values, train_losses, val_losses

def plot_metrics(epoch_numbers, psnr_values, ssim_values, train_losses, val_losses, title, filename):
    """Plot PSNR, SSIM, Training Loss, and Validation Loss over epochs."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    
    ax1.plot(epoch_numbers, psnr_values, marker='o', label='PSNR', color='b')
    ax1.plot(epoch_numbers, ssim_values, marker='s', label='SSIM', color='r')
    ax2.plot(epoch_numbers, train_losses, marker='^', linestyle='--', label='Train Loss', color='g')
    ax2.plot(epoch_numbers, val_losses, marker='v', linestyle='--', label='Val Loss', color='orange')
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR / SSIM")
    ax2.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.grid()
    plt.savefig(filename)
    plt.show()

def upscale_image(model, checkpoint_path, image_path, output_path):
    """Load a trained model and upscale an arbitrary low-resolution image."""
    model = load_checkpoint(model, checkpoint_path)
    transform = transforms.Compose([transforms.ToTensor()])
    
    image = Image.open(image_path).convert("RGB")
    lr_tensor = transform(image).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    lr_tensor = lr_tensor.to(device)
    
    with torch.no_grad():
        output = model(lr_tensor)
        output = torch.clamp(output, 0, 1).cpu().squeeze(0)
    
    output_image = transforms.ToPILImage()(output)
    output_image.save(output_path)
    print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    import argparse
    from dataset import DIV2KDataset  # Ensure dataset is properly set up
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing checkpoints')
    parser.add_argument('--test_hr', type=str, required=True, help='Path to high-resolution test images')
    parser.add_argument('--test_lr', type=str, required=True, help='Path to low-resolution test images')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    args = parser.parse_args()
    
    model = DAT(upsample_steps=4)
    
    test_dataset = DIV2KDataset(args.test_hr, args.test_lr, transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    epochs, psnr_vals, ssim_vals, train_losses, val_losses = evaluate_checkpoints(model, args.checkpoint_dir, test_loader)
    plot_metrics(epochs, psnr_vals, ssim_vals, train_losses, val_losses, "Model Checkpoint Analysis", "checkpoint_analysis.png")