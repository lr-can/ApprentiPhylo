import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 分类器和标记映射
CLASSIFIERS = [
    'AACnnClassifier',
    'DenseMsaClassifier',
    'DenseSiteClassifier',
    'LogisticRegressionClassifier',
]

CLASSIFIER_MARKERS = {
    'AACnnClassifier': 's',      # 正方形
    'DenseSiteClassifier': '^',  # 上三角
    'DenseMsaClassifier': 'o',   # 圆圈
    'LogisticRegressionClassifier': 'x'  # 叉号
}

# 数据根目录
GROUP_ROOT = 'viridiplantae_group_results'
RUNS_ROOT = 'runs_viridiplantae'

def extract_suffix(sim_name):
    """提取模拟名称的后缀"""
    m = re.search(r'(ext|root)_[0-9.]+$', sim_name)
    return m.group(0) if m else sim_name

def collect_all_data():
    """收集所有数据到DataFrame"""
    records = []
    for group in os.listdir(GROUP_ROOT):
        group_path = os.path.join(GROUP_ROOT, group)
        if not os.path.isdir(group_path):
            continue
        for root, dirs, files in os.walk(group_path):
            for file in files:
                if file.startswith('distance_results') and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    distance = None
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.startswith('OVERALL_AVERAGE'):
                                distance = float(line.strip().split(',')[-1])
                                break
                    if distance is None:
                        continue
                    sim_name = os.path.basename(os.path.dirname(file_path))
                    sim_folder = os.path.join(RUNS_ROOT, sim_name)
                    if not os.path.isdir(sim_folder):
                        continue
                    for clf in CLASSIFIERS:
                        clf_folder = os.path.join(sim_folder, clf)
                        summary_path = os.path.join(clf_folder, 'summary.json')
                        if os.path.isfile(summary_path):
                            try:
                                with open(summary_path, 'r') as f:
                                    data = json.load(f)
                                if clf == 'LogisticRegressionClassifier':
                                    accs = data.get('fold_accuracies', [])
                                    accs = [float(a) for a in accs]
                                    acc = sum(accs) / len(accs) if accs else None
                                else:
                                    acc = data.get('val_acc', None)
                                if acc is not None:
                                    records.append({
                                        'group': group,
                                        'sim_name': sim_name,
                                        'classifier': clf,
                                        'distance': distance,
                                        'acc': acc,
                                        'has_data2': 'data2' in sim_name
                                    })
                            except Exception as e:
                                continue
    return pd.DataFrame(records)

def plot_by_group_distance():
    """按组和距离绘制图表"""
    df = collect_all_data()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 为每个组创建子图
    for idx, (group, group_df) in enumerate(df.groupby('group')):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 获取该组所有唯一的模拟名称
        sim_names = sorted(group_df['sim_name'].unique())
        n_sims = len(sim_names)
        cmap = plt.cm.get_cmap('tab10', n_sims)
        color_map = {sim: cmap(i) for i, sim in enumerate(sim_names)}
        
        # 分别绘制有data2和无data2的数据
        for has_data2 in [True, False]:
            data2_df = group_df[group_df['has_data2'] == has_data2]
            
            # 为每个分类器绘制数据
            for clf in CLASSIFIERS:
                clf_df = data2_df[data2_df['classifier'] == clf]
                if not clf_df.empty:
                    for sim_name, sim_df in clf_df.groupby('sim_name'):
                        color = color_map[sim_name]
                        marker = CLASSIFIER_MARKERS[clf]
                        linestyle = '--' if has_data2 else '-'
                        label = f"{sim_name} ({clf}) {'(with data2)' if has_data2 else '(without data2)'}"
                        
                        ax.scatter(sim_df['distance'], sim_df['acc'], 
                                 color=color, marker=marker, s=100,
                                 label=label, alpha=0.7)
                        ax.plot(sim_df['distance'], sim_df['acc'], 
                               color=color, linestyle=linestyle,
                               linewidth=2, alpha=0.5)
        
        ax.set_title(f'组: {group}', fontsize=16)
        ax.set_xlabel('平均距离', fontsize=14)
        ax.set_ylabel('准确率', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 将图例放在图表外部
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('group_distance_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_distance_comparison():
    """比较有data2和无data2的距离分布"""
    df = collect_all_data()
    
    plt.figure(figsize=(12, 8))
    
    # 创建箱线图，使用tab10颜色映射
    sns.boxplot(data=df, x='group', y='distance', hue='has_data2', palette='tab10')
    
    plt.title('各组距离分布比较 (有/无 data2)', fontsize=16)
    plt.xlabel('组', fontsize=14)
    plt.ylabel('平均距离', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='是否包含data2', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    plot_by_group_distance()
    plot_distance_comparison() 