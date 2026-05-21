import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr, pearsonr
import encoding  # 导入编码模块
import sgrnaFeature  # 导入模型模块
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)


def get_column_name(df, possible_names):

    for name in possible_names:
        if name in df.columns:
            return name
    return None


def preprocess_data(df, file_name=None):
    original_len = len(df)
    
    # 检查是否有target列
    if 'target' not in df.columns:
        print("Error: 'target' column not found in data")
        return df
    
    # 检查是否有outcome列
    if 'outcome' not in df.columns:
        print("Error: 'outcome' column not found in data")
        return df
    
    # 过滤长度不符合要求的行
    filtered_rows = df[(df['target'].str.len() != 23) | (df['outcome'].str.len() != 23)]
    df = df[(df['target'].str.len() == 23) & (df['outcome'].str.len() == 23)]
    
    filtered_len = len(df)
    removed_rows = original_len - filtered_len
    if removed_rows > 0:
        print(f"预处理: 过滤掉{removed_rows}行不符合要求的数据 (原始{original_len}行, 剩余{filtered_len}行)")
        
        if file_name and not filtered_rows.empty:
            filtered_dir = 'filtered_data'
            os.makedirs(filtered_dir, exist_ok=True)
            
            filtered_file_name = f"filtered_{file_name.replace('.csv', '.txt')}"
            filtered_file_path = os.path.join(filtered_dir, filtered_file_name)
            
            with open(filtered_file_path, 'w') as f:
                f.write(','.join(filtered_rows.columns) + '\n')
                for _, row in filtered_rows.iterrows():
                    f.write(','.join([str(val) if pd.notna(val) else '' for val in row]) + '\n')
            
            print(f"已将被过滤的{len(filtered_rows)}行数据保存到: {filtered_file_path}")
    
    return df



class agrnaDataset(Dataset):
    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        file_name = os.path.basename(csv_file)
        self.data = preprocess_data(self.data, file_name)

        self.targets = self.data['target'].values
        self.outcomes = self.data['outcome'].values
        self.target_encodings = [encoding.sgrna_encoding(target) for target in self.targets]
        self.outcome_encodings = [encoding.sgrna_encoding(outcome) for outcome in self.outcomes]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):

        target_encoding = torch.tensor(self.target_encodings[idx], dtype=torch.float32)
        outcome_encoding = torch.tensor(self.outcome_encodings[idx], dtype=torch.float32)
        target = self.targets[idx]
        outcome = self.outcomes[idx]
        return target_encoding, outcome_encoding, target, outcome



def evaluate_model(loader, model, device):
    model.eval()  
    all_targets = []
    all_outcomes = []
    all_preds = []
    
    with torch.no_grad():
        for target_encodings, outcome_encodings, targets, outcomes in loader:
            target_encodings, outcome_encodings = target_encodings.to(device), outcome_encodings.to(device)
            
            outputs = model(target_encodings, outcome_encodings)
            
            all_targets.extend(targets)
            all_outcomes.extend(outcomes)
            all_preds.extend(outputs.cpu().numpy().flatten())
    
    return all_targets, all_outcomes, all_preds



def save_results(original_data, predictions_dict, output_file):
    # 复制原始数据
    results = original_data.copy()
    
    # 添加预测列
    for model_name, preds in predictions_dict.items():
        results[f'pre_pro_{model_name}'] = preds
    
    results.to_csv(output_file, index=False)
    
    print(f'Results saved to {output_file}')


def calculate_correlation(true_values, predicted_values):
    """计算真实值和预测值之间的Spearman和Pearson相关系数"""
    spearman_corr, spearman_p = spearmanr(true_values, predicted_values)
    pearson_corr, pearson_p = pearsonr(true_values, predicted_values)
    
    print(f"\n相关系数计算结果:")
    print(f"Spearman相关系数: {spearman_corr:.4f} (p值: {spearman_p:.4e})")
    print(f"Pearson相关系数: {pearson_corr:.4f} (p值: {pearson_p:.4e})")
    
    return spearman_corr, pearson_corr


# 模型名称映射
MODEL_MAP = {
    'ABE7': 'ABE7_best_model.pth',
    'ABE8e': 'ABE8e_best_model.pth',
    'ABEmax': 'ABEmax_best_model.pth',
    'BE4': 'BE4_best_model.pth',
    'CBE4max': 'CBE4max_best_model.pth',
    'Target-AID': 'Target-AID_best_model.pth'
}


def predict_single_model(model_name, input_file, output_file):
    """
    使用指定的模型对指定的文件进行预测并保存结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取模型文件路径
    if model_name not in MODEL_MAP:
        print(f"Error: Unknown model '{model_name}'. Available models: {list(MODEL_MAP.keys())}")
        return
    
    model_file = MODEL_MAP[model_name]
    model_path = os.path.join(os.getcwd(), model_file)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return
    
    print(f'\nProcessing {input_file} with model {model_file}')
    
    # 读取原始数据
    original_data = pd.read_csv(input_file)
    file_name = os.path.basename(input_file)
    original_data = preprocess_data(original_data, file_name)
    
    # 准备数据集和加载器
    test_dataset = agrnaDataset(input_file)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 加载模型
    input_channels = 4  
    seq_length = 23  
    model = sgrnaFeature.StackcCNN_OneHot_Pro(input_channels, seq_length).to(device)
    model.load_state_dict(torch.load(model_path))
    
    print(f'  Using model {model_file}')
    
    # 进行预测
    test_targets, test_outcomes, test_preds = evaluate_model(test_loader, model, device)
    
    # 保存结果
    predictions_dict = {model_name: test_preds}
    save_results(original_data, predictions_dict, output_file)
    
    # 计算相关系数（如果存在pro列）
    if 'pro' in original_data.columns:
        true_values = original_data['pro'].values
        calculate_correlation(true_values, test_preds)
    else:
        print("\nWarning: 'pro' column not found, skipping correlation calculation.")


def predict_models():
    """
    调用models下的已经训练好的模型，对test文件夹下的测试集进行预测，将结果保存在test_results文件夹下的csv文件中
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    
    test_dir = 'test'
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory '{test_dir}' not found.")
        return
    
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    # 模型列表
    model_names = ['ABE7_train', 'ABE8e_pro_train', 'ABEmax_pro_train']
    
    for test_file in test_files:
        file_name = os.path.splitext(test_file)[0]
        print(f'\nProcessing {test_file}')
        
        # 读取原始数据
        original_data = pd.read_csv(os.path.join(test_dir, test_file))
        original_data = preprocess_data(original_data, test_file)
        
        # 准备数据集和加载器
        test_dataset = agrnaDataset(os.path.join(test_dir, test_file))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # 存储所有模型的预测结果
        predictions_dict = {}
        
        for model_name in model_names:
            model_file = f'{model_name}_best_model.pth'
            model_path = os.path.join('models', model_file)
            
            # 从模型名称中移除_train后缀，用于结果列名
            result_model_name = model_name.replace('_train', '')
            
            if not os.path.exists(model_path):
                print(f'Warning: Model file not found for {model_name}')
                continue
            
            print(f'  Using model {model_file}')
            
            input_channels = 4  
            seq_length = 23  
            model = sgrnaFeature.StackcCNN_OneHot_Pro(input_channels, seq_length).to(device)
            
            model.load_state_dict(torch.load(model_path))
            
            test_targets, test_outcomes, test_preds = evaluate_model(test_loader, model, device)
            predictions_dict[result_model_name] = test_preds
        
        test_output_file = os.path.join('test_results', f'{file_name}_results.csv')
        save_results(original_data, predictions_dict, test_output_file)





def main():
    # 检查命令行参数
    if len(sys.argv) == 4:
        # 使用命令行参数进行预测
        model_name = sys.argv[1]
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        
        print(f"Starting prediction with model: {model_name}")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        predict_single_model(model_name, input_file, output_file)
        print("\nPrediction completed!")
    else:
        # 默认行为：调用所有模型对test文件夹下的测试集进行预测
        print("Starting model prediction...")
        predict_models()
        print("\nAll tasks completed!")


if __name__ == '__main__':
    main()


# 命令行使用示例:
# python predict.py ABE7 ABE7_test.csv ABE7_results.csv
# python predict.py ABE8e ABE8e_test.csv ABE8e_results.csv
# python predict.py ABEmax ABEmax_test.csv ABEmax_results.csv
# python predict.py BE4 BE4_test.csv BE4_results.csv
# python predict.py CBE4max CBE4max_test.csv CBE4max_results.csv
# python predict.py Target-AID Target-AID_test.csv Target-AID_results.csv