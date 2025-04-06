from torch import *

def Model(self):
    # Main CNN layer
    # 
    return



class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=2):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out += residual
        
        # Final activation
        out = torch.relu(out)
        
        return out