import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import argparse
import os
from dataset import DIV2KDataset
from DAT import DAT # Import the model from DAT.py

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train the Dual Aggregation Transformer for Image Super-Resolution")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Path to the DIV2K training dataset')
    parser.add_argument('--save_model', type=str, default='./checkpoints/model_epoch.pth', help='Path to save the trained model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')  # Add num_workers argument
    parser.add_argument('--num_upscale', type=int, default=2, help='Number of times to upscale images during training')
    return parser.parse_args()

def get_data_loaders(hr_dir, lr_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Resize to a fixed size before any other operations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = DIV2KDataset(hr_dir, lr_dir, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # Add num_workers
    
    return train_loader

def main():
    # Parse arguments
    args = parse_args()

    # Get training data loaders
    train_loader = get_data_loaders(args.dataset_path + '/DIV2K_train_HR', args.dataset_path + '/DIV2K_train_LR', args.batch_size, args.num_workers)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Initialization with dynamic upscaling parameter
    model = DAT(upsample_steps=args.num_upscale).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning Rate Scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss Function (MSE loss, but you can also try perceptual loss for better image quality)
    def loss_fn(pred, target):
        # Resize the target to match the predicted output size
        target = F.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
        return F.mse_loss(pred, target)

    print("Start Training.......................")
    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # Forward pass
            pred_images = model(lr_images)
            
            # Compute loss
            loss = loss_fn(pred_images, hr_images)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print loss
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss / len(train_loader)}")
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f'{args.save_model}_epoch_{epoch+1}.pth')

        # Step the scheduler
        scheduler.step()

if __name__ == '__main__':
    main()
