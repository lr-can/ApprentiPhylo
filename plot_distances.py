import os
import json
import matplotlib.pyplot as plt
import re
import math
import itertools
from matplotlib.lines import Line2D
import numpy as np
import glob

# 只保留AACnnClassifier
CLASSIFIERS = {
    'AACnnClassifier': 'o',
}

# 后缀顺序
SUFFIX_ORDERS = [
    ['ext_0', 'ext_0.05', 'ext_0.1', 'ext_0.2', 'ext_0.5'],
    ['root_0', 'root_0.05', 'root_0.1', 'root_0.2', 'root_0.5'],
]

def extract_suffix(sim_name):
    m = re.search(r'(ext|root)_[0-9.]+$', sim_name)
    return m.group(0) if m else sim_name

def get_suffix_order(sim_names):
    for order in SUFFIX_ORDERS:
        if all(any(suf in n for n in sim_names) for suf in order):
            return order
    return sorted(list({extract_suffix(n) for n in sim_names}))

# 获取distance_results的平均距离
def get_average_distance(file_path):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('OVERALL_AVERAGE'):
                    return float(line.strip().split(',')[-1])
    except:
        return None
    return None

# 获取summary.json的准确率
def get_classifier_acc(summary_path, classifier):
    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
        if classifier == 'LogisticRegressionClassifier':
            accs = data.get('fold_accuracies', [])
            accs = [float(a) for a in accs]
            return sum(accs) / len(accs) if accs else None
        else:
            return data.get('val_acc', None)
    except:
        return None

def collect_points_by_group():
    base_path = 'viridiplantae_group_results'
    run_base = 'runs_viridiplantae'
    group_points = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.startswith('distance_results') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                avg_dist = get_average_distance(file_path)
                if avg_dist is None:
                    continue
                # 获取group名
                rel_path = os.path.relpath(root, base_path)
                group = rel_path.split(os.sep)[0]
                sim_name = os.path.basename(os.path.dirname(file_path))
                sim_folder = os.path.join(run_base, sim_name)
                if not os.path.isdir(sim_folder):
                    continue
                clf = 'AACnnClassifier'
                clf_folder = os.path.join(sim_folder, clf)
                summary_path = os.path.join(clf_folder, 'summary.json')
                if os.path.isfile(summary_path):
                    acc = get_classifier_acc(summary_path, clf)
                    if acc is not None:
                        if group not in group_points:
                            group_points[group] = []
                        group_points[group].append({
                            'x': avg_dist,
                            'y': acc,
                            'label': f'{sim_name}-{clf}',
                            'marker': CLASSIFIERS[clf],
                            'sim_name': sim_name,
                            'is_data2': 'data2' in sim_name.lower()
                        })
    return group_points

def plot_points_by_group(connect_line=True):
    group_points = collect_points_by_group()
    for group, points in group_points.items():
        plt.figure(figsize=(12, 6))
        # 先分为with/without data2
        for is_data2 in [True, False]:
            pts = [pt for pt in points if pt['is_data2'] == is_data2]
            if not pts:
                continue
            # 排序
            if group in ['group1_four_model_F', 'group4_WAG_basic_comparison']:
                pts_sorted = sorted(pts, key=lambda pt: pt['x'])
            else:
                sim_names = [pt['sim_name'] for pt in pts]
                suffix_order = get_suffix_order(sim_names)
                pts_sorted = sorted(pts, key=lambda pt: suffix_order.index(extract_suffix(pt['sim_name'])) if extract_suffix(pt['sim_name']) in suffix_order else 99)
            # 画点和legend
            for pt in pts_sorted:
                plt.scatter(pt['x'], pt['y'], marker=pt['marker'], label=pt['label'])
            # 连线
            if connect_line and len(pts_sorted) > 1:
                xs = [pt['x'] for pt in pts_sorted]
                ys = [pt['y'] for pt in pts_sorted]
                linestyle = '-' if is_data2 else ':'
                plt.plot(xs, ys, linestyle=linestyle, color='gray', alpha=0.7)
        plt.title(f'MPD vs AACnnClassifier Accuracy')
        plt.xlabel('MPD')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f'distance_classifier_plot_{group}_AACnn.png', dpi=300, bbox_inches='tight')
        plt.close()

# 新增：所有group的点画在同一张图上，不画连线，group用不同颜色/marker，legend为group名
def plot_all_groups_points_no_line():
    group_points = collect_points_by_group()
    plt.figure(figsize=(12, 8))
    colors = itertools.cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    markers = itertools.cycle(['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8'])
    for group, points in group_points.items():
        color = next(colors)
        marker = next(markers)
        xs = [pt['x'] for pt in points]
        ys = [pt['y'] for pt in points]
        labels = [pt['label'] for pt in points]
        plt.scatter(xs, ys, marker=marker, color=color, label=group, alpha=0.8)
    plt.title('MPD vs AACnnClassifier Accuracy ')
    plt.xlabel('MPD')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('distance_classifier_plot_all_groups_AACnn.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_groups_points_no_line_point_legend():
    group_points = collect_points_by_group()
    plt.figure(figsize=(14, 10))
    colors = itertools.cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    markers = itertools.cycle(['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8'])
    legend_elements = []
    used_labels = set()
    group_color_marker = {}
    # 先为每个group分配颜色和marker
    for group in group_points:
        group_color_marker[group] = (next(colors), next(markers))
    for group, points in group_points.items():
        color, marker = group_color_marker[group]
        for pt in points:
            label = pt['label']
            # 只要每个点一个legend entry
            if label not in used_labels:
                plt.scatter(pt['x'], pt['y'], marker=marker, color=color, label=label, alpha=0.8)
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=label,
                                              markerfacecolor=color, markersize=9, markeredgecolor='k'))
                used_labels.add(label)
            else:
                plt.scatter(pt['x'], pt['y'], marker=marker, color=color, alpha=0.8)
    plt.title('MPD vs AACnnClassifier Accuracy (All Points, Each Point in Legend)')
    plt.xlabel('MPD')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    # legend marker size与点完全一致
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title='Point', handleheight=2, borderaxespad=2, labelspacing=1.2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('distance_classifier_plot_all_groups_AACnn_points_legend.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_groups_points_no_line_point_legend_unique():
    group_points = collect_points_by_group()
    plt.figure(figsize=(14, 10))
    # 使用tab20色板
    cmap = plt.get_cmap('tab20')
    n_colors = 20
    color_list = [cmap(i) for i in range(n_colors)]
    marker_list = ['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8', '>', '<', 'p', 'H', '|', '_', '+', 'x', '1', '2']
    combos = list(itertools.product(color_list, marker_list))
    all_points = []
    for group, points in group_points.items():
        for pt in points:
            all_points.append(pt)
    legend_elements = []
    for i, pt in enumerate(all_points):
        color, marker = combos[i % len(combos)]
        plt.scatter(pt['x'], pt['y'], marker=marker, color=color, label=pt['label'], alpha=0.9, edgecolor='k', linewidths=0.7)
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=pt['label'],
                                      markerfacecolor=color, markersize=9, markeredgecolor='k'))
    plt.title('MPD vs AACnnClassifier Accuracy (All Points Unique Color/Marker)')
    plt.xlabel('MPD')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    # legend marker size与点完全一致
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title='Point', handleheight=2, borderaxespad=2, labelspacing=1.2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('distance_classifier_plot_all_groups_AACnn_points_legend_unique.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_groups_234_points():
    base_path = 'viridiplantae_group_results'
    run_base = 'runs_viridiplantae'
    group_names = [d for d in os.listdir(base_path) if d.startswith('group2') or d.startswith('group3') or d.startswith('group4')]
    all_points = []
    group3_sim_names = set()
    group4_keep = {'WAG_F', 'WAG_F_EP', 'WAG_F_EP_DATA2'}
    # 先收集group3的sim_name
    for group in group_names:
        if not group.startswith('group3'):
            continue
        group_path = os.path.join(base_path, group)
        for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
            data2_path = os.path.join(group_path, data2_type)
            if not os.path.isdir(data2_path):
                continue
            for sim_name in os.listdir(data2_path):
                group3_sim_names.add(sim_name)
    # 再收集所有点，遇到同名sim_name只保留group3
    for group in group_names:
        group_path = os.path.join(base_path, group)
        for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
            data2_path = os.path.join(group_path, data2_type)
            if not os.path.isdir(data2_path):
                continue
            for sim_name in os.listdir(data2_path):
                # 如果不是group3且sim_name已在group3中，跳过
                if not group.startswith('group3') and sim_name in group3_sim_names:
                    continue
                # group4只保留白名单sim_name
                if group.startswith('group4') and sim_name not in group4_keep:
                    continue
                sim_path = os.path.join(data2_path, sim_name)
                if not os.path.isdir(sim_path):
                    continue
                dist_files = glob.glob(os.path.join(sim_path, 'distance_results*.csv'))
                if not dist_files:
                    continue
                dist_file = dist_files[0]
                avg_dist = get_average_distance(dist_file)
                if avg_dist is None:
                    continue
                clf = 'AACnnClassifier'
                sim_folder = os.path.join(run_base, sim_name)
                clf_folder = os.path.join(sim_folder, clf)
                summary_path = os.path.join(clf_folder, 'summary.json')
                if not os.path.isfile(summary_path):
                    continue
                acc = get_classifier_acc(summary_path, clf)
                if acc is None:
                    continue
                all_points.append({
                    'x': avg_dist,
                    'y': acc,
                    'label': f'{group}/{sim_name}',
                })
    # 画图
    plt.figure(figsize=(14, 10))
    # 主色调
    main_colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    # 补充tab20
    cmap = plt.get_cmap('tab20')
    n_colors = 20
    color_list = main_colors + [cmap(i) for i in range(n_colors)]
    marker_list = ['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8', '>', '<', 'p', 'H', '|', '_', '+', 'x', '1', '2']
    combos = list(itertools.product(color_list, marker_list))
    legend_elements = []
    group_colors = {
        'group2': 'yellow',
        'group3': 'blue',
        'group4': 'orange',
    }
    # 1. 先收集group2/group3所有sim_name的key（data2状态+结尾数值）
    def extract_marker_key(sim_name):
        # 判断是否data2
        is_data2 = 'DATA2' in sim_name.upper()
        # 提取结尾数值
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        val = m.group(1) if m else ''
        return f"{'data2' if is_data2 else 'nodata2'}_{val}"
    marker_key_set = set()
    for pt in all_points:
        group_name = pt['label'].split('/')[0]
        if group_name.startswith('group2') or group_name.startswith('group3'):
            sim_name = pt['label'].split('/')[1]
            marker_key_set.add(extract_marker_key(sim_name))
    marker_key_list = sorted(marker_key_set)
    marker_map = {k: marker_list[i % len(marker_list)] for i, k in enumerate(marker_key_list)}
    # 2. 画图
    def get_marker_and_size(sim_name):
        if sim_name.startswith('WAG_EP_DATA2_R_'):
            marker = 's'
        elif sim_name.startswith('WAG_EP_R_'):
            marker = 'D'
        elif sim_name.startswith('WAG_F_P_DATA2_E_'):
            marker = '^'
        elif sim_name.startswith('WAG_F_P_E_'):
            marker = 'v'
        elif sim_name == 'WAG_F_EP_DATA2':
            marker = 'o'
        elif sim_name == 'WAG_F_EP':
            marker = 'p'
        elif sim_name == 'WAG_F':
            marker = '*'
        else:
            marker = 'o'
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        if m:
            size = 80 + float(m.group(1)) * 200
        else:
            size = 120
        return marker, size
    for i, pt in enumerate(all_points):
        group_name = pt['label'].split('/')[0]
        sim_name = pt['label'].split('/')[1]
        # 去除group前缀
        group_name_clean = re.sub(r'^group\d+_?', '', group_name)
        label_clean = f'{group_name_clean}/{sim_name}' if not group_name_clean.startswith('WAG') else sim_name
        if group_name.startswith('group2'):
            color = group_colors['group2']
        elif group_name.startswith('group3'):
            color = group_colors['group3']
        elif group_name.startswith('group4'):
            color = group_colors['group4']
        else:
            color = 'gray'
        marker, size = get_marker_and_size(sim_name)
        plt.scatter(pt['x'], pt['y'], marker=marker, color=color, label=label_clean, alpha=0.9, edgecolor='k', linewidths=0.7, s=size*1.5)
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=label_clean,
                                    markerfacecolor=color, markersize=size/10, markeredgecolor='k'))
    plt.title('MPD vs AACnnClassifier Accuracy ')
    plt.xlabel('MPD')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    # legend顺序
    legend_order = [
        *(f'WAG_EP_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_EP_DATA2_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_F_P_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_F_P_DATA2_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        'WAG_F_EP', 'WAG_F_EP_DATA2', 'WAG_F'
    ]
    def legend_key(label):
        sim_name = label.split('/')[-1]
        try:
            return legend_order.index(sim_name)
        except ValueError:
            return 999
    legend_elements_sorted = sorted(legend_elements, key=lambda l: legend_key(l.get_label()))
    plt.legend(handles=legend_elements_sorted, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title='Point', handleheight=2, borderaxespad=2, labelspacing=1.2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('distance_classifier_plot_groups_234_AACnn.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_group2_group3_separate_with_lines():
    base_path = 'viridiplantae_group_results'
    run_base = 'runs_viridiplantae'
    group_colors = {
        'group2': 'yellow',
        'group3': 'blue',
    }
    marker_list = ['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8', '>', '<', 'p', 'H', '|', '_', '+', 'x', '1', '2']
    suffix_order = ['0', '0.05', '0.1', '0.2', '0.5']
    # marker和size分配函数
    def get_marker_and_size(sim_name):
        if sim_name.startswith('WAG_EP_DATA2_R_'):
            marker = 's'
        elif sim_name.startswith('WAG_EP_R_'):
            marker = 'D'
        elif sim_name.startswith('WAG_F_P_DATA2_E_'):
            marker = '^'
        elif sim_name.startswith('WAG_F_P_E_'):
            marker = 'v'
        elif sim_name == 'WAG_F_EP_DATA2':
            marker = 'o'
        elif sim_name == 'WAG_F_EP':
            marker = 'p'
        elif sim_name == 'WAG_F':
            marker = '*'
        else:
            marker = 'o'
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        if m:
            size = 80 + float(m.group(1)) * 200
        else:
            size = 120
        return marker, size
    def extract_marker_key(sim_name):
        is_data2 = 'DATA2' in sim_name.upper()
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        val = m.group(1) if m else ''
        return f"{'data2' if is_data2 else 'nodata2'}_{val}"
    def extract_suffix(sim_name):
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        return m.group(1) if m else ''
    for group in ['group2', 'group3']:
        # 自动识别真实文件夹名
        group_folder = [d for d in os.listdir(base_path) if d.startswith(group)]
        if not group_folder:
            continue
        group_folder = group_folder[0]
        group_path = os.path.join(base_path, group_folder)
        all_points = []
        for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
            data2_path = os.path.join(group_path, data2_type)
            if not os.path.isdir(data2_path):
                continue
            for sim_name in os.listdir(data2_path):
                sim_path = os.path.join(data2_path, sim_name)
                if not os.path.isdir(sim_path):
                    continue
                dist_files = glob.glob(os.path.join(sim_path, 'distance_results*.csv'))
                if not dist_files:
                    continue
                dist_file = dist_files[0]
                avg_dist = get_average_distance(dist_file)
                if avg_dist is None:
                    continue
                clf = 'AACnnClassifier'
                sim_folder = os.path.join(run_base, sim_name)
                clf_folder = os.path.join(sim_folder, clf)
                summary_path = os.path.join(clf_folder, 'summary.json')
                if not os.path.isfile(summary_path):
                    continue
                acc = get_classifier_acc(summary_path, clf)
                if acc is None:
                    continue
                all_points.append({
                    'x': avg_dist,
                    'y': acc,
                    'sim_name': sim_name,
                    'is_data2': 'DATA2' in sim_name.upper(),
                    'suffix': extract_suffix(sim_name),
                    'label': sim_name,
                })
        # 分配marker
        marker_key_set = set(extract_marker_key(pt['sim_name']) for pt in all_points)
        marker_key_list = sorted(marker_key_set)
        marker_map = {k: marker_list[i % len(marker_list)] for i, k in enumerate(marker_key_list)}
        # 画图
        plt.figure(figsize=(10, 7))
        color = group_colors[group]
        legend_elements = []
        # 先画点
        for i, pt in enumerate(all_points):
            marker, size = get_marker_and_size(pt['sim_name'])
            plt.scatter(pt['x'], pt['y'], marker=marker, color=color, label=pt['label'], alpha=0.9, edgecolor='k', linewidths=0.7, s=size*1.5)
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=pt['label'],
                                        markerfacecolor=color, markersize=size/10, markeredgecolor='k'))
        # 画线
        for is_data2, linestyle in [(True, '-'), (False, ':')]:
            pts = [pt for pt in all_points if pt['is_data2'] == is_data2]
            # 按suffix_order排序
            pts_sorted = sorted(pts, key=lambda pt: suffix_order.index(pt['suffix']) if pt['suffix'] in suffix_order else 99)
            if len(pts_sorted) > 1:
                xs = [pt['x'] for pt in pts_sorted]
                ys = [pt['y'] for pt in pts_sorted]
                plt.plot(xs, ys, linestyle=linestyle, color=color, alpha=0.7, linewidth=2)
        plt.title(f'MPD vs AACnnClassifier Accuracy')
        plt.xlabel('MPD')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        # legend顺序
        legend_order = [
            *(f'WAG_EP_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
            *(f'WAG_EP_DATA2_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
            *(f'WAG_F_P_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
            *(f'WAG_F_P_DATA2_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
            'WAG_F_EP', 'WAG_F_EP_DATA2', 'WAG_F'
        ]
        def legend_key(label):
            sim_name = label.split('/')[-1]
            try:
                return legend_order.index(sim_name)
            except ValueError:
                return 999
        legend_elements_sorted = sorted(legend_elements, key=lambda l: legend_key(l.get_label()))
        plt.legend(handles=legend_elements_sorted, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title='Point', handleheight=2, borderaxespad=2, labelspacing=1.2)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(f'distance_classifier_plot_{group_folder}_AACnn_lines.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_group4_with_special_style():
    base_path = 'viridiplantae_group_results'
    run_base = 'runs_viridiplantae'
    group4_color = 'orange'
    marker_list = ['o', 's', 'v', '^', 'D', 'P', '*', 'X', 'h', '8', '>', '<', 'p', 'H', '|', '_', '+', 'x', '1', '2']
    suffix_order = ['0', '0.05', '0.1', '0.2', '0.5']
    # marker和size分配函数
    def get_marker_and_size(sim_name):
        if sim_name.startswith('WAG_EP_DATA2_R_'):
            marker = 's'
        elif sim_name.startswith('WAG_EP_R_'):
            marker = 'D'
        elif sim_name.startswith('WAG_F_P_DATA2_E_'):
            marker = '^'
        elif sim_name.startswith('WAG_F_P_E_'):
            marker = 'v'
        elif sim_name == 'WAG_F_EP_DATA2':
            marker = 'o'
        elif sim_name == 'WAG_F_EP':
            marker = 'p'
        elif sim_name == 'WAG_F':
            marker = '*'
        else:
            marker = 'o'
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        if m:
            size = 80 + float(m.group(1)) * 200
        else:
            size = 120
        return marker, size
    # 特殊sim_name
    special_names = [
        'WAG_F_P_E_0', 'WAG_F_P_DATA2_E_0', 'WAG_EP_R_0', 'WAG_EP_DATA2_R_0'
    ]
    # 1. 先收集group2/group3的特殊sim_name样式
    special_style = {}
    def extract_marker_key(sim_name):
        is_data2 = 'DATA2' in sim_name.upper()
        m = re.search(r'_(\d+(?:\.\d+)?)$', sim_name)
        val = m.group(1) if m else ''
        return f"{'data2' if is_data2 else 'nodata2'}_{val}"
    # 收集group2/group3的marker分配
    marker_key_set = set()
    marker_key_map = {}
    for group in ['group2', 'group3']:
        group_folder = [d for d in os.listdir(base_path) if d.startswith(group)]
        if not group_folder:
            continue
        group_folder = group_folder[0]
        group_path = os.path.join(base_path, group_folder)
        for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
            data2_path = os.path.join(group_path, data2_type)
            if not os.path.isdir(data2_path):
                continue
            for sim_name in os.listdir(data2_path):
                marker_key = extract_marker_key(sim_name)
                marker_key_set.add(marker_key)
    marker_key_list = sorted(marker_key_set)
    marker_map = {k: marker_list[i % len(marker_list)] for i, k in enumerate(marker_key_list)}
    # 记录特殊sim_name的marker和线型
    for group in ['group2', 'group3']:
        group_folder = [d for d in os.listdir(base_path) if d.startswith(group)]
        if not group_folder:
            continue
        group_folder = group_folder[0]
        group_path = os.path.join(base_path, group_folder)
        for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
            data2_path = os.path.join(group_path, data2_type)
            if not os.path.isdir(data2_path):
                continue
            for sim_name in os.listdir(data2_path):
                if sim_name in special_names:
                    marker = marker_map[extract_marker_key(sim_name)]
                    is_data2 = 'DATA2' in sim_name.upper()
                    linestyle = '-' if is_data2 else ':'
                    color = 'yellow' if group == 'group2' else 'blue'
                    special_style[sim_name] = {'marker': marker, 'color': color, 'linestyle': linestyle}
    # 2. 收集group4所有点
    group4_folder = [d for d in os.listdir(base_path) if d.startswith('group4')]
    if not group4_folder:
        return
    group4_folder = group4_folder[0]
    group4_path = os.path.join(base_path, group4_folder)
    all_points = []
    for data2_type in ['WITH_DATA2', 'withDATA2', 'WITHOUT_DATA2', 'withoutDATA2']:
        data2_path = os.path.join(group4_path, data2_type)
        if not os.path.isdir(data2_path):
            continue
        for sim_name in os.listdir(data2_path):
            sim_path = os.path.join(data2_path, sim_name)
            if not os.path.isdir(sim_path):
                continue
            dist_files = glob.glob(os.path.join(sim_path, 'distance_results*.csv'))
            if not dist_files:
                continue
            dist_file = dist_files[0]
            avg_dist = get_average_distance(dist_file)
            if avg_dist is None:
                continue
            clf = 'AACnnClassifier'
            sim_folder = os.path.join(run_base, sim_name)
            clf_folder = os.path.join(sim_folder, clf)
            summary_path = os.path.join(clf_folder, 'summary.json')
            if not os.path.isfile(summary_path):
                continue
            acc = get_classifier_acc(summary_path, clf)
            if acc is None:
                continue
            all_points.append({
                'x': avg_dist,
                'y': acc,
                'sim_name': sim_name,
                'is_data2': 'DATA2' in sim_name.upper(),
                'suffix': re.search(r'_(\d+(?:\.\d+)?)$', sim_name).group(1) if re.search(r'_(\d+(?:\.\d+)?)$', sim_name) else '',
                'label': sim_name,
            })
    # 3. 分配group4其它点的marker
    group4_other_names = [pt['sim_name'] for pt in all_points if pt['sim_name'] not in special_names]
    group4_marker_map = {name: marker_list[i % len(marker_list)] for i, name in enumerate(group4_other_names)}
    # 4. 画图
    plt.figure(figsize=(10, 7))
    legend_elements = []
    # 只画点，不画线
    for i, pt in enumerate(all_points):
        sim_name = pt['sim_name']
        marker, size = get_marker_and_size(sim_name)
        if sim_name in special_style:
            color = special_style[sim_name]['color']
        else:
            color = group4_color
        plt.scatter(pt['x'], pt['y'], marker=marker, color=color, label=pt['label'], alpha=0.9, edgecolor='k', linewidths=0.7, s=size*1.5)
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=pt['label'],
                                    markerfacecolor=color, markersize=size/10, markeredgecolor='k'))
    plt.title(f'MPD vs AACnnClassifier Accuracy ')
    plt.xlabel('MPD')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    # group4 legend顺序
    legend_order = [
        *(f'WAG_EP_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_EP_DATA2_R_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_F_P_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        *(f'WAG_F_P_DATA2_E_{v}' for v in ['0', '0.05', '0.1', '0.2', '0.5']),
        'WAG_F_EP', 'WAG_F_EP_DATA2', 'WAG_F'
    ]
    def legend_key(label):
        sim_name = label.split('/')[-1]
        try:
            return legend_order.index(sim_name)
        except ValueError:
            return 999
    legend_elements_sorted = sorted(legend_elements, key=lambda l: legend_key(l.get_label()))
    plt.legend(handles=legend_elements_sorted, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, title='Point', handleheight=2, borderaxespad=2, labelspacing=1.2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'distance_classifier_plot_{group4_folder}_AACnn_lines.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_groups_234_points() 
    plot_group2_group3_separate_with_lines()
    plot_group4_with_special_style()