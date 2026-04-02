import numpy as np
import pandas as pd

def preprocess_bio_features(bio_features):
    """预处理生物特征，包括标准化和缺失值处理"""
    # 将列表转换为numpy数组
    bio_features = np.array(bio_features, dtype=np.float32)
    
    # 处理缺失值（将NaN替换为0）
    bio_features = np.nan_to_num(bio_features)
    
    # 标准化处理
    mean = np.mean(bio_features, axis=0)
    std = np.std(bio_features, axis=0)
    # 避免除以0
    std[std == 0] = 1e-8
    bio_features = (bio_features - mean) / std
    
    return bio_features

def sgrna_encoding(seq, max_len=23):
    seq = seq[:max_len] if len(seq) > max_len else seq.ljust(max_len, '-')
    base_dict = {'A':0, 'T':1, 'C':2, 'G':3}
    encoding = np.zeros((max_len, 4))
    for i in range(max_len):
        if seq[i] in base_dict:
            encoding[i][base_dict[seq[i]]] = 1
    return encoding

def position_code(seq, max_len=23):
    """生成位置编码，根据碱基类型和位置生成不同的数值"""
    seq = seq[:max_len] if len(seq) > max_len else seq.ljust(max_len, '-')
    position_enc = np.zeros((max_len, 1))
    for i in range(max_len):
        if seq[i] == '-':
            continue
        if seq[i] in 'Aa':
            position_enc[i, 0] = i + 1
        elif seq[i] in "Cc":
            position_enc[i, 0] = i + 24
        elif seq[i] in "Gg":
            position_enc[i, 0] = i + 47
        elif seq[i] in "Tt":
            position_enc[i, 0] = i + 70
    
    return position_enc

def sgrna_with_position_encoding(seq, max_len=23):
    """生成结合one-hot编码和位置编码的特征"""
    one_hot_enc = sgrna_encoding(seq, max_len)
    pos_enc = position_code(seq, max_len)
    combined_enc = np.concatenate((one_hot_enc, pos_enc), axis=1)
    return combined_enc


def double_sgrna_encoding(seq1, seq2,max_len=23):
    enc1 = sgrna_encoding(seq1)  
    enc2 = sgrna_encoding(seq2) 
    encoding = []
    for i in range(max_len):
        encoding.append(np.concatenate([enc1[i], enc2[i]]))
    return np.array(encoding)


def off_target_sgrna_encoding(on_seq, off_seq,max_len=23):

    on_seq = on_seq.replace('_', '-')
    off_seq = off_seq.replace('_', '-')

    base_dict = {'A':0, 'T':1, 'C':2, 'G':3}

    encoding = []
    
    for i in range(max_len):

        on_base = on_seq[i].upper() if i < len(on_seq) else '-'
        off_base = off_seq[i].upper() if i < len(off_seq) else '-'  
        
        channel = [0]*4
        if on_base in base_dict:
            channel[base_dict[on_base]] = 1
        if off_base in base_dict and off_base != on_base:
            channel[base_dict[off_base]] = 1
            
        mut_feature = [0]*3
        has_gap = (on_base == '-' or off_base == '-')
        same_base = (on_base.upper() == off_base.upper())
        
        if has_gap:
            mut_feature[0] = 1
            mut_feature[1] = int(on_base in base_dict) 
            mut_feature[2] = int(off_base in base_dict) 
        elif not same_base:

            on_channel = base_dict.get(on_base, -1)
            off_channel = base_dict.get(off_base, -1)

            mut_feature[1] = 1 if on_channel != -1 and channel[on_channel] == 1 else 0
            mut_feature[2] = 1 if off_channel != -1 and channel[off_channel] == 1 else 0
            
            if mut_feature[1] and mut_feature[2]:
                on_idx = base_dict.get(on_base, 4)
                off_idx = base_dict.get(off_base, 4)
                mut_feature[1], mut_feature[2] = (1, 0) if on_idx < off_idx else (0, 1)
        
        encoding.append(channel + mut_feature)
    
    return np.array(encoding)


if __name__ == "__main__":

    seq1 = "ANTCG"  
    seq2 = "ANTCG"  

    seq3 = "AACGATCc"  
    seq4 = "A-CGATCG"  

    seq5 = "ATCGATCT-"  
    seq6 = "ATCGATCTT"  

    
    features_1 = double_sgrna_encoding(seq1,seq2,max_len=20)
    features_2 = off_target_sgrna_encoding(seq3,seq4)
    features_3 = off_target_sgrna_encoding(seq5,seq6)

    print("\n特征矩阵形状:", features_1.shape)
    print(features_1[:])  
    print("\n特征矩阵形状:", features_2.shape)
    print(features_2[:])  
    print("\n特征矩阵形状:", features_3.shape)
    print(features_3[:])  
