import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialWindowSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(SpatialWindowSelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = in_channels // heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, height, width = x.size()
        q = self.query(x).view(batch_size, self.heads, self.embed_size, height * width)
        k = self.key(x).view(batch_size, self.heads, self.embed_size, height * width)
        v = self.value(x).view(batch_size, self.heads, self.embed_size, height * width)
        
        attention = torch.matmul(q.transpose(2, 3), k) / (self.embed_size ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v.transpose(2, 3))
        
        out = out.transpose(2, 3).view(batch_size, self.heads * self.embed_size, height, width)
        return out

class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(ChannelWiseSelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = in_channels // heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        q = self.query(x).view(batch_size, self.heads, self.embed_size, height * width)
        k = self.key(x).view(batch_size, self.heads, self.embed_size, height * width)
        v = self.value(x).view(batch_size, self.heads, self.embed_size, height * width)
        
        attention = torch.matmul(q, k.transpose(2, 3)) / (self.embed_size ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        
        out = out.view(batch_size, self.heads * self.embed_size, height, width)
        return out

class DualAggregationTransformerBlock(nn.Module):
    def __init__(self, in_channels, heads=1, use_spatial=True):
        super(DualAggregationTransformerBlock, self).__init__()
        self.use_spatial = use_spatial
        
        if self.use_spatial:
            self.attention = SpatialWindowSelfAttention(in_channels, heads)
        else:
            self.attention = ChannelWiseSelfAttention(in_channels, heads)
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        attention_features = self.attention(x)
        conv_features = self.conv(x)
        out = self.final_conv(attention_features + conv_features)
        return out

class ImageReconstructionModule(nn.Module):
    def __init__(self, in_channels, out_channels=3, upsample_steps=1):
        super(ImageReconstructionModule, self).__init__()
        self.upsample_steps = upsample_steps
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        for _ in range(self.upsample_steps):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.upconv(x)

class DAT(nn.Module):
    def __init__(self, in_channels=3, num_features=64, heads=1, num_blocks=4, upsample_steps=1):
        super(DAT, self).__init__()
        
        self.shallow_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        self.deep_blocks = nn.ModuleList([
            DualAggregationTransformerBlock(num_features, heads=heads, use_spatial=(i % 2 == 0))
            for i in range(num_blocks)
        ])
        
        self.reconstruction = ImageReconstructionModule(num_features, upsample_steps=upsample_steps)
    
    def forward(self, x):
        shallow_features = self.shallow_conv(x)
        deep_features = shallow_features
        for block in self.deep_blocks:
            deep_features = block(deep_features)
        return self.reconstruction(deep_features)

if __name__ == "__main__":
    model = DAT(upsample_steps=3)
    input_image = torch.randn(1, 3, 64, 64)
    output_image = model(input_image)
    print(output_image.shape)
