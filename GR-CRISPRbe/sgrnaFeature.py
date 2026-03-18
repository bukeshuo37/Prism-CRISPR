import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(BiLSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self, seq, offset, length):
        batch_size = length[0]
        device = seq.device
        #emb = self.dropout(self.embedding(seq, offset))
        emb = self.embedding(seq, offset)
        if self.training:
            emb = self.dropout(emb)
        emb = self.bn(emb)
        emb_v = emb.view(batch_size, 20, -1)  # 20bp sequence length
        emb_vt = emb_v.transpose(1, 0)
        out, (hidden, _) = self.rnn(emb_vt)
        out = out.permute(1, 0, 2)
        if self.training:
            query = self.dropout(out)
        else:
            query = out
        #query = self.dropout(out)
        attn_output, alpha_n = self.attention_net(out, query)
        #logit = torch.nn.functional.leaky_relu(self.fc(attn_output))
        logit = torch.sigmoid(self.fc(attn_output))
        return logit


class ASPP_1D(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(ASPP_1D, self).__init__()

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
        
        self.atrous_block6 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, 1, padding=6, dilation=6),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channel, out_channel, 1, 1),
            nn.ReLU()  # 移除BatchNorm层
        )

        self.conv_1x1_output = nn.Conv1d(out_channel * 5, out_channel, 1, 1)
        self.batch_norm_output = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x_shape = x.shape[2]
        
        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block6 = self.atrous_block6(x)
        
        global_avg = self.global_avg_pool(x)
        global_avg = F.interpolate(global_avg, size=x_shape, mode='nearest')

        x = torch.cat([atrous_block1, atrous_block2, atrous_block4, atrous_block6, global_avg], dim=1)
        
        x = self.conv_1x1_output(x)
        x = self.batch_norm_output(x)
        
        x = self.dropout(x)
        
        return x


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=None):
        super(Attention, self).__init__()
        if attn_dim is None:
            attn_dim = hidden_dim
        
        # 加性注意力：先投影到attn_dim维度，再投影到1
        self.attn_proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)  # 通常不加偏置
        self.tanh = nn.Tanh()  # 或者用 nn.Tanh()
    
    def forward(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        
        # 计算注意力分数
        energy = self.tanh(self.attn_proj(lstm_output))  # [batch, seq_len, attn_dim]
        attention_scores = self.v(energy)  # [batch, seq_len, 1]
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # 序列维度归一化
        
        # 计算上下文向量
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector, attention_weights.squeeze(-1)

        
class StackcCNN(nn.Module):
    def __init__(self, input_channels=4, seq_length=23,num_classes=1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels*2, out_channels=64, kernel_size=3, padding='same'),    
            nn.ReLU(),  
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),   
            nn.BatchNorm1d(64),
            nn.ReLU(),    
            nn.Conv1d(64, 32, kernel_size=3, padding='same'),   
            nn.BatchNorm1d(32),
            nn.ReLU(),                    
            nn.Flatten(),
            nn.Dropout(0.3), 
        )
        self.prodict_layers= nn.Sequential(
            nn.Linear(736, 256),  
            nn.BatchNorm1d(256),             
            nn.ReLU(),
            nn.Dropout(0.3),        
            nn.Linear(256, 128),   
            nn.BatchNorm1d(128),   
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x1,x2):
        x = torch.cat([x1, x2], dim=2)  
        x = x.permute(0, 2, 1) # 一维CNN需要扭转维度     
        x = self.conv_layers(x)   
        x = self.prodict_layers(x)   
        return x




import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """注意力模块：让模型学会关注重要的通道"""
    def __init__(self, channel, reduction=8): # One-Hot通道少，reduction改小一点
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    """多尺度卷积块：同时看局部(k=3)和全局(k=5/1)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 分支1：1x1 卷积 (降维/整合)
        self.branch1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        
        # 分支2：3x3 卷积 (看局部)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        )
        
        # 分支3：5x5 卷积 (看稍远一点)
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=5, padding=2)
        )
        
        # 分支4：池化分支 (保留最强特征)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


import torch
import torch.nn as nn

class StackcCNN_OneHot_Pro(nn.Module):
    def __init__(self, input_channels=4, seq_length=23):
        super().__init__()
        
        # 1. 输入层调整
        # Target(4) + Outcome(4) + Diff(4) = 12通道
        self.real_input_channels = input_channels * 3 
        

        self.stem = nn.Sequential(
            nn.Conv1d(self.real_input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # 3. 核心特征提取 (减少通道，移除池化)
        
        # Layer 1: 32 -> 48 (原 64->128)
        self.layer1 = InceptionBlock(32, 48) 
        self.se1 = SEBlock(48)


        self.layer2 = InceptionBlock(48, 64)
        self.se2 = SEBlock(64)
 

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1472, 256), # 输入维度变为 1472，隐层减小到 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 64),   # 隐层减小到 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2):
        diff = x2 - x1 
        x = torch.cat([x1, x2, diff], dim=2) 
        x = x.permute(0, 2, 1) 

        # 初始特征
        x = self.stem(x)
        
        # 第一层 Inception (无池化，长度保持 23)
        x = self.layer1(x)
        x = self.se1(x)
        
        # 第二层 Inception (无池化，长度保持 23)
        x = self.layer2(x)
        x = self.se2(x)
        
        # 预测
        x = self.regressor(x)
        return x


class cnn_module(nn.Module):
    def __init__(self, input_channels=4, seq_length=23,num_classes=1):

        super().__init__()

        self.position_encoding = nn.Parameter(torch.zeros(1, input_channels, seq_length * 2))
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

        self.bilstm = nn.LSTM(
            input_size=32, 
            hidden_size=64, 
            num_layers=2,  
            bidirectional=True, 
            batch_first=True,  
            dropout=0.3  
        )

        self.attention = Attention(hidden_dim=64 * 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(64*2, 256),  
            nn.BatchNorm1d(256),             
            nn.ReLU(),
            nn.Dropout(0.3),        
            nn.Linear(256, 128),   
            nn.BatchNorm1d(128),   
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x1,x2):

        x = torch.cat([x1, x2], dim=1)

        x = x.permute(0, 2, 1)      

        x = x + self.position_encoding

        x = self.conv_layers(x)   
        x = self.aspp(x)
        skip = self.skip_conv(x)
        x = x + skip 

        x = x.permute(0, 2, 1)
        lstm_output, _ = self.bilstm(x)
        context_vector, attention_weights = self.attention(lstm_output)

        return self.fc_layers(context_vector)  

