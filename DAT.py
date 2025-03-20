import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAggregationTransformerBlock(nn.Module):
    def __init__(self, in_channels, heads=1, embed_size=64):
        super(DualAggregationTransformerBlock, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        
        # Spatial Window Self-Attention (SW-SA)
        self.spatial_attention = SpatialWindowSelfAttention(in_channels, heads)
        
        # Channel-Wise Self-Attention (CW-SA)
        self.channel_attention = ChannelWiseSelfAttention(in_channels, heads)
        
        # Convolution branch for locality
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Adaptive Interaction Module (AIM)
        self.aim = AdaptiveInteractionModule(in_channels)
        
        # Final convolution to refine features
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        # Apply SW-SA and CW-SA
        spatial_features = self.spatial_attention(x)
        channel_features = self.channel_attention(x)
        
        # Apply convolution for locality
        conv_features = self.conv(x)
        
        # Adaptive Interaction Module (AIM)
        aim_features = self.aim(spatial_features, conv_features)
        
        # Combine spatial and channel features, followed by final convolution
        out = self.final_conv(aim_features)
        
        return out


class SpatialWindowSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(SpatialWindowSelfAttention, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.embed_size = in_channels // heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value matrices
        q = self.query(x).view(batch_size, self.heads, self.embed_size, height * width)
        k = self.key(x).view(batch_size, self.heads, self.embed_size, height * width)
        v = self.value(x).view(batch_size, self.heads, self.embed_size, height * width)
        
        # Ensure the attention computation matches the shapes
        q = q.view(batch_size * self.heads, self.embed_size, height * width)
        k = k.view(batch_size * self.heads, self.embed_size, height * width)
        v = v.view(batch_size * self.heads, self.embed_size, height * width)
        
        # Calculate attention
        attention = torch.matmul(q.transpose(1, 2), k)  # Attention shape: [batch_size * heads, height * width, height * width]
        attention = F.softmax(attention / self.embed_size ** 0.5, dim=-1)  # Normalize the attention matrix
        
        # Calculate output: (batch_size * heads) x (embed_size) x (height * width)
        out = torch.matmul(attention, v.transpose(1, 2))  # out shape: [batch_size * heads, height * width, embed_size]
        
        # Reshape back to [batch_size, heads * embed_size, height, width]
        out = out.transpose(1, 2).view(batch_size, self.heads * self.embed_size, height, width)
        return out


class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(ChannelWiseSelfAttention, self).__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.embed_size = in_channels // heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value matrices
        q = self.query(x).view(batch_size, self.heads, self.embed_size, height * width)
        k = self.key(x).view(batch_size, self.heads, self.embed_size, height * width)
        v = self.value(x).view(batch_size, self.heads, self.embed_size, height * width)
        
        # Ensure the attention computation matches the shapes
        q = q.view(batch_size * self.heads, self.embed_size, height * width)
        k = k.view(batch_size * self.heads, self.embed_size, height * width)
        v = v.view(batch_size * self.heads, self.embed_size, height * width)
        
        # Calculate attention: (batch_size * heads) x (embed_size) x (height * width)
        attention = torch.matmul(q.transpose(1, 2), k)  # Attention shape: [batch_size * heads, height * width, height * width]
        attention = F.softmax(attention / self.embed_size ** 0.5, dim=-1)  # Normalize the attention matrix
        
        # Calculate output: (batch_size * heads) x (embed_size) x (height * width)
        out = torch.matmul(attention, v.transpose(1, 2))  # out shape: [batch_size * heads, height * width, embed_size]
        
        # Reshape back to [batch_size, heads * embed_size, height, width]
        out = out.transpose(1, 2).view(batch_size, self.heads * self.embed_size, height, width)
        return out


class AdaptiveInteractionModule(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveInteractionModule, self).__init__()
        
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Spatial-Interaction (S-I)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        
        # Channel-Interaction (C-I)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
    
    def forward(self, spatial_features, conv_features):
        # Apply spatial and channel interaction
        spatial_interacted = self.spatial_interaction(spatial_features)
        channel_interacted = self.channel_interaction(conv_features)
        
        # Element-wise multiplication of interactions
        spatial_output = spatial_interacted * channel_interacted
        channel_output = channel_interacted * spatial_interacted
        
        return spatial_output + channel_output


class ImageReconstructionModule(nn.Module):
    def __init__(self, in_channels, out_channels=3, upsample_steps=1):
        super(ImageReconstructionModule, self).__init__()
        self.upsample_steps = upsample_steps
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Upsample for the specified number of steps
        for _ in range(self.upsample_steps):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.upconv(x)


class DAT(nn.Module):
    def __init__(self, in_channels=3, num_features=64, heads=1, num_blocks=4, upsample_steps=1):
        super(DAT, self).__init__()
        
        self.shallow_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Deep feature extraction module
        self.deep_blocks = nn.ModuleList([DualAggregationTransformerBlock(num_features, heads=heads) for _ in range(num_blocks)])
        
        # Image reconstruction module with the upsample_steps argument
        self.reconstruction = ImageReconstructionModule(num_features, upsample_steps=upsample_steps)
    
    def forward(self, x):
        # Shallow feature extraction
        shallow_features = self.shallow_conv(x)
        
        # Deep feature extraction
        deep_features = shallow_features
        for block in self.deep_blocks:
            deep_features = block(deep_features)
        
        # Image reconstruction
        reconstructed_image = self.reconstruction(deep_features)
        
        return reconstructed_image


if __name__ == "__main__":
    # Initialize model with the desired number of upsample steps (e.g., 3 upsample steps)
    model = DAT(upsample_steps=3)
    
    input_image = torch.randn(1, 3, 64, 64)  # Example input image (batch_size, channels, height, width)
    output_image = model(input_image)
    
    print(output_image.shape)  # Output image shape after upscaling