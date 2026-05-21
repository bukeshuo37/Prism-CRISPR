import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import torch
from torch.utils.data import DataLoader, Dataset

import encoding  # 导入编码模块
import sgrnaFeature  # 导入模型模块

# 定义数据集类
class agrnaDataset(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        # 直接提取sgrna序列，不再拼接pam列
        self.sgrnas = self.data['sgrna'].values
        # 对sgrna序列进行编码
        self.encodings = [encoding.sgrna_encoding(sgrna) for sgrna in self.sgrnas]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 获取编码后的序列和对应的sgrna
        encoding = torch.tensor(self.encodings[idx], dtype=torch.float32)
        sgrna = self.sgrnas[idx]
        return encoding, sgrna

# 测试函数
def evaluate_model(loader, model, device):
    model.eval()  # 设置模型为评估模式
    all_sgrnas = []
    all_preds = []
    
    with torch.no_grad():
        for encodings, sgrnas in loader:
            encodings = encodings.to(device)
            
            outputs = model(encodings)
            
            # 保存结果
            all_sgrnas.extend(sgrnas)
            all_preds.extend(outputs.cpu().numpy().flatten())
    
    return all_sgrnas, all_preds

# 保存结果函数
def save_results(original_data, model_predictions, output_file):
    # 创建结果数据框，包含原有文件的所有列
    results = original_data.copy()
    # 添加预测值列，固定命名为 pred_eff
    for model_name, preds in model_predictions.items():
        results['pred_eff'] = preds
    # 保存为CSV文件
    results.to_csv(output_file, index=False)
    
    print(f'Results saved to {output_file}')

# 计算相关系数
def calculate_correlations(true_values, pred_values, model_name):
    spearman_corr, spearman_p = spearmanr(true_values, pred_values)
    pearson_corr, pearson_p = pearsonr(true_values, pred_values)
    
    print(f'\nCorrelation results for {model_name}:')
    print(f'Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})')
    print(f'Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})')
    
    return {'spearman': spearman_corr, 'spearman_p': spearman_p,
            'pearson': pearson_corr, 'pearson_p': pearson_p}

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建结果文件夹（如果需要）
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Predict sgRNA activity using specified model')
    parser.add_argument('model_type', type=str, 
                        choices=['eSpCas9', 'SpCas9-HF1', 'WT-SpCas9'],
                        help='Model type: eSpCas9, SpCas9-HF1, or WT-SpCas9')
    parser.add_argument('input_file', type=str, help='Input CSV file path')
    parser.add_argument('output_file', type=str, help='Output CSV file path')
    
    args = parser.parse_args()
    
    # 根据模型类型确定模型路径
    model_path_map = {
        'eSpCas9': 'eSpCas9_best_model.pth',
        'SpCas9-HF1': 'SpCas9-HF1_best_model.pth',
        'WT-SpCas9': 'WT-SpCas9_best_model.pth'
    }
    
    model_path = model_path_map[args.model_type]
    input_file = args.input_file
    output_file = args.output_file
    
    print(f'\nModel type: {args.model_type}')
    print(f'Model path: {model_path}')
    print(f'Input file: {input_file}')
    print(f'Output file: {output_file}')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f'Error: Input file {input_file} not found.')
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f'Error: Model file {model_path} not found.')
        return
    
    # 读取原始数据，用于保存结果时包含所有列
    original_data = pd.read_csv(input_file)
    
    # 创建数据集和数据加载器
    test_dataset = agrnaDataset(input_file)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    input_channels = 4  # sgrna_encoding的输出通道数
    seq_length = 23  # sgrna_encoding的序列长度
    model = sgrnaFeature.cnn_module(input_channels, seq_length).to(device)
    
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    print(f'Model loaded from {model_path}')
    
    # 评估测试集
    test_sgrnas, test_preds = evaluate_model(test_loader, model, device)
    
    # 存储预测值
    model_predictions = {args.model_type: test_preds}
    
    # 计算相关系数（如果存在真实值列）
    if 'true_eff' in original_data.columns:
        true_values = original_data['true_eff'].values
        calculate_correlations(true_values, test_preds, args.model_type)
    else:
        print('\nInfo: No true_eff column found in input file, skipping correlation calculation.')
    
    # 保存预测结果
    save_results(original_data, model_predictions, output_file)

if __name__ == '__main__':
    main()