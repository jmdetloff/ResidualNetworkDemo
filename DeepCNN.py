import torch.nn as nn

def ConvolutionalSeries(num_filters, block_count, use_skip_connections):
    blocks = []
    for _ in range(block_count):
        blocks.append(ConvolutionalBlock(num_filters, use_skip_connections))
    return nn.Sequential(*blocks)

class ConvolutionalBlock(nn.Module):
    def __init__(self, num_filters, use_skip_connection):
        super(ConvolutionalBlock, self).__init__()

        self.use_skip_connection = use_skip_connection

        reduced_channels = num_filters//2
        
        self.conv1 = nn.Conv2d(num_filters, reduced_channels, 1, padding=0)
        self.batch1 = nn.BatchNorm2d(reduced_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(reduced_channels)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(reduced_channels, num_filters, 1, padding=0)
        self.batch3 = nn.BatchNorm2d(num_filters)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.batch1(x_out)
        x_out = self.relu1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.batch2(x_out)
        x_out = self.relu2(x_out)
        x_out = self.conv3(x_out)
        x_out = self.batch3(x_out)

        if self.use_skip_connection:
            x_out = x_out + x
        
        return self.relu3(x_out)

class DeepCNN(nn.Module):
    def __init__(self, res_block_count, use_skip_connections):
        super(DeepCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride = 1, padding=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
    
        self.res_series = ConvolutionalSeries(16, res_block_count, use_skip_connections)
        
        self.poolN = nn.MaxPool2d(kernel_size=2)
        
        self.fc = nn.Linear(16*14*14, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        out = self.res_series(out)
                
        out = self.poolN(out)
        
        out = out.view(out.size(0), -1)

        return self.fc(out)
