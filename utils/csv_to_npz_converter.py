#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import gc

def process_csv_to_npz(csv_files, output_file):
    """
    将多个大型CSV文件转换为原始NPZ格式，每个文件代表一种币种的所有时间点数据
    
    参数:
        csv_files (list): CSV文件路径列表，每个文件对应一种币种
        output_file (str): 输出NPZ文件路径
    """
    # 硬编码参数
    feature_num = 43
    chunksize = 10000
    
    # 币种数量等于文件数量
    stock_num = len(csv_files)
    
    print(f"处理{len(csv_files)}个CSV文件，每个文件代表一种币种...")
    
    # 定义要使用的43个特征
    selected_features = [
        'spread',  # 核心市场状态
        'buys',
        'sells',
        # 前10档买单深度
        'bids_distance_0', 'bids_distance_1', 'bids_distance_2', 'bids_distance_3', 'bids_distance_4',
        'bids_distance_5', 'bids_distance_6', 'bids_distance_7', 'bids_distance_8', 'bids_distance_9',
        'bids_notional_0', 'bids_notional_1', 'bids_notional_2', 'bids_notional_3', 'bids_notional_4',
        'bids_notional_5', 'bids_notional_6', 'bids_notional_7', 'bids_notional_8', 'bids_notional_9',
        # 前10档卖单深度
        'asks_distance_0', 'asks_distance_1', 'asks_distance_2', 'asks_distance_3', 'asks_distance_4',
        'asks_distance_5', 'asks_distance_6', 'asks_distance_7', 'asks_distance_8', 'asks_distance_9',
        'asks_notional_0', 'asks_notional_1', 'asks_notional_2', 'asks_notional_3', 'asks_notional_4',
        'asks_notional_5', 'asks_notional_6', 'asks_notional_7', 'asks_notional_8', 'asks_notional_9'
    ]
    
    # 确认特征数量与预期一致
    assert len(selected_features) == feature_num, f"选定的特征数量({len(selected_features)})与预期的特征数量({feature_num})不一致"
    
    print(f"将使用以下{feature_num}个特征:")
    print(", ".join(selected_features))
    
    # 检查文件是否存在
    for file in csv_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"文件不存在: {file}")
    
    # 计算每个文件的行数
    file_rows = []
    for file in csv_files:
        # 使用wc -l来快速计算行数而不读取整个文件
        wc_output = os.popen(f"wc -l {file}").read()
        try:
            rows = int(wc_output.split()[0])
            file_rows.append(rows)
            print(f"文件 {file} 有 {rows} 行")
        except:
            print(f"无法计算 {file} 的行数，将继续处理")
            file_rows.append(0)
    
    # 使用最小的行数作为时间点数量，确保所有币种数据对齐
    total_rows = min(file_rows) if file_rows else 0
    if total_rows == 0:
        raise ValueError("无法确定数据行数，请检查文件")
    
    print(f"使用最小行数作为时间点数量: {total_rows}")
    
    # 初始化数据数组
    input_data = np.zeros((total_rows, stock_num, feature_num), dtype=np.float32)
    buy_price = np.zeros((total_rows, stock_num), dtype=np.float32)
    sell_price = np.zeros((total_rows, stock_num), dtype=np.float32)
    date_list = []  # 只保存第一个币种的时间戳
    
    # 处理每个CSV文件（每个文件代表一种币种）
    for stock_idx, file in enumerate(csv_files):
        print(f"处理币种 {stock_idx+1}/{stock_num}: {file}")
        
        # 用于追踪当前处理的时间点
        current_time = 0
        
        # 使用分块读取大文件
        for chunk in tqdm(pd.read_csv(file, chunksize=chunksize), desc=f"处理 {os.path.basename(file)}"):
            # 预处理数据
            # 移除不需要的列，通常第一列是索引
            if chunk.columns[0] == 'Unnamed: 0' or chunk.columns[0] == '':
                chunk = chunk.iloc[:, 1:]
            
            # 提取时间列
            time_col = None
            for col in chunk.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    time_col = col
                    break
            
            if time_col is None:
                print("警告：找不到时间列，使用行索引作为时间")
            
            # 确保所有需要的特征列都存在
            missing_features = [feat for feat in selected_features if feat not in chunk.columns]
            if missing_features:
                raise ValueError(f"CSV文件{file}中缺少以下特征列: {', '.join(missing_features)}")
            
            # 找到midpoint列用于计算买卖价格
            midpoint_col = None
            for col in chunk.columns:
                if 'midpoint' in col.lower() or 'mid' in col.lower() or 'price' in col.lower():
                    midpoint_col = col
                    break
            
            if midpoint_col is None:
                print("警告：找不到中间价列，假设第2列是中间价")
                if len(chunk.columns) > 1:
                    midpoint_col = chunk.columns[1]
                else:
                    raise ValueError("CSV格式不符合要求，无法找到中间价列")
            
            # 计算买卖价格
            chunk_buy = chunk[midpoint_col] + chunk['spread']/2
            chunk_sell = chunk[midpoint_col] - chunk['spread']/2
            
            # 填充数据 - 每行是一个时间点的一种币种数据
            rows_in_chunk = len(chunk)
            
            for i in range(rows_in_chunk):
                if current_time >= total_rows:
                    break
                
                # 只在处理第一个币种时保存时间戳
                if stock_idx == 0 and time_col is not None:
                    try:
                        date_list.append(str(chunk[time_col].iloc[i]))
                    except:
                        date_list.append(f"time_{current_time}")
                
                # 填充特征数据
                for j, feature_name in enumerate(selected_features):
                    input_data[current_time, stock_idx, j] = chunk[feature_name].iloc[i]
                
                # 填充买卖价格
                buy_price[current_time, stock_idx] = chunk_buy.iloc[i]
                sell_price[current_time, stock_idx] = chunk_sell.iloc[i]
                
                current_time += 1
                
                if current_time >= total_rows:
                    break
            
            # 清理内存
            del chunk
            gc.collect()
            
            # 如果已经处理完所有时间点，则退出
            if current_time >= total_rows:
                break
    
    # 确保date_list长度与时间点数量一致
    if len(date_list) < total_rows:
        print(f"警告：时间戳数量({len(date_list)})少于时间点数量({total_rows})，将添加默认时间戳")
        date_list.extend([f"time_{i}" for i in range(len(date_list), total_rows)])
    elif len(date_list) > total_rows:
        print(f"警告：时间戳数量({len(date_list)})多于时间点数量({total_rows})，将截断")
        date_list = date_list[:total_rows]
    
    # 保存为NPZ文件
    print(f"保存数据到 {output_file}")
    print(f"input_data shape: {input_data.shape}")
    print(f"buy_price shape: {buy_price.shape}")
    print(f"sell_price shape: {sell_price.shape}")
    print(f"时间点数量: {len(date_list)}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    np.savez(output_file, 
             input_data=input_data, 
             buy_price=buy_price, 
             sell_price=sell_price,
             date_list=np.array(date_list, dtype=object))
    
    print("转换完成！")
    print("注意：滑动窗口操作请使用sliding_window.ipynb进行处理")

def main():
    # 硬编码输入输出路径
    csv_files = [
        "./data/ADA_1sec.csv",  # 第一种币
        "./data/BTC_1sec.csv",  # 第二种币
        "./data/ETH_1sec.csv",  # 第三种币
    ]
    output_file = "./data/coin.npz"  # 修改为相对于当前工作目录的路径
    
    print("开始转换CSV文件到NPZ格式...")
    print(f"输入文件: {csv_files}")
    print(f"输出文件: {output_file}")
    
    process_csv_to_npz(csv_files, output_file)

if __name__ == "__main__":
    main() 