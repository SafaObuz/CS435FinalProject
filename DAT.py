import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAggregationTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_heads=8, num_layers=4, embed_size=256):
        super(DualAggregationTransformer, self).__init__()
        
        self.conv_in = nn.Conv2d(in_channels, embed_size, kernel_size=3, stride=1, padding=1)
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads),
            num_layers=num_layers
        )
        
        # Local Aggregation (Convolutional part)
        self.local_aggregation = nn.Conv2d(embed_size, embed_size, kernel_size=3, stride=1, padding=1)
        
        # Global Aggregation (Transformer part)
        self.global_aggregation = nn.Conv2d(embed_size, embed_size, kernel_size=3, stride=1, padding=1)
        
        # Output Convolution Layer
        self.conv_out = nn.Conv2d(embed_size, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Initial convolution to get feature map
        x = self.conv_in(x)
        
        # Dual Aggregation: Local and Global
        local_features = self.local_aggregation(x)
        global_features = self.global_aggregation(x)
        
        # Reshape for Transformer Encoder (flatten spatial dimensions to sequence)
        batch_size, channels, height, width = x.size()
        x_reshaped = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [seq_len, batch_size, d_model]

        # Transformer Encoder
        x = self.transformer_encoder(x_reshaped)

        # Reshape back to image dimensions
        x = x.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        # Combine local and global features (add them)
        x = x + local_features + global_features
        
        # Output layer
        x = self.conv_out(x)

        # Apply a final activation function (optional)
        x = torch.tanh(x)  # Example: output values in the range [-1, 1]
        
        return x
