import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BioFeatureMLP(nn.Module):
    def __init__(self, input_dim, output_dim=256, hidden_dims=[128, 64], dropout_rate=0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.ReLU())  # 使用tanh激活输出在-1到1之间
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class ASPP_1D(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(ASPP_1D, self).__init__()

        # 保留原有的膨胀卷积块
        self.atrous_block1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
        self.atrous_block2 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, padding=2, dilation=2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
        self.atrous_block4 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, padding=4, dilation=4),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
        # 更大膨胀率的卷积块
        self.atrous_block6 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, padding=6, dilation=6),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
        # 修复：修改全局平均池化路径，移除BatchNorm
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.ReLU()  # 移除BatchNorm层
        )

        # 保持输出通道数调整
        self.conv_1x1_output = nn.Conv1d(out_channel * 5, out_channel, 1, 1)
        self.batch_norm_output = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 获取输入特征的形状用于后续的上采样
        x_shape = x.shape[2]
        
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block6 = self.atrous_block6(x)
        
        # 对全局平均池化的结果进行上采样
        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(global_avg, size=x_shape, mode='nearest')

        # 连接所有特征路径的输出
        x = torch.cat([atrous_block1, atrous_block2, atrous_block4, atrous_block6, global_avg], dim=1)
        
        # 通过1x1卷积调整通道数并应用批归一化
        x = self.conv_1x1_output(x)
        x = self.batch_norm_output(x)
        
        # 应用Dropout
        x = self.dropout(x)
        
        return x


class cnn_module(nn.Module):
    def __init__(self, input_channels=4, seq_length=23,bio_feature_dim=11,num_classes=1):

        super().__init__()

        #添加可学习的位置编码
        self.position_encoding = nn.Parameter(torch.zeros(1, input_channels, seq_length))
        nn.init.uniform_(self.position_encoding, -0.02, 0.02)

        
        self.permute = lambda x: x.permute(0, 2, 1)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=4, padding='same'),    
            nn.ReLU(),  
            nn.Conv1d(64, 64, kernel_size=4, padding='same'),   
            nn.BatchNorm1d(64),
            nn.ReLU(),                      
            #nn.MaxPool1d(2),
            #nn.Flatten(),
            nn.Dropout(0.3), 
        )
        
        self.aspp = ASPP_1D(in_channel=64, out_channel=32)
        self.skip_conv = nn.Conv1d(32, 32, 1)

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, seq_length)  
            conv_output = self.conv_layers(dummy_input)
            aspp_output = self.aspp(conv_output)
            skip_output = self.skip_conv(aspp_output)
            fused_output = aspp_output + skip_output
            flattened_output = torch.flatten(fused_output, start_dim=1)
            fc_input_dim = flattened_output.shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 512),  
            nn.BatchNorm1d(512),             
            nn.ReLU(),
            nn.Dropout(0.3),        
            nn.Linear(512, 256),   
            nn.BatchNorm1d(256),   
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, bio_features=None):

        x = self.permute(x)      
        #位置编码
        x = x + self.position_encoding

        x = self.conv_layers(x)   
        x = self.aspp(x)
        skip = self.skip_conv(x)
        x = x + skip 
        x = torch.flatten(x, start_dim=1)

        return self.fc_layers(x)  

