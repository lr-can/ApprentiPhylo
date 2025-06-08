import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 图片文件名和子图标题
files_titles = [
    ('distance_classifier_plot_group1_four_model_F_AACnn.png', 'group1_four_model_F'),
    ('distance_classifier_plot_group2_WAG_F_P_E_AACnn.png', 'group2_WAG_F_P_E'),
    ('distance_classifier_plot_group3_WAG_EP_R_AACnn.png', 'group3_WAG_EP_R'),
    ('distance_classifier_plot_group4_WAG_basic_comparison_AACnn.png', 'group4_WAG_basic_comparison'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (fname, title) in enumerate(files_titles):
    img = mpimg.imread(fname)
    axes[i].imshow(img)
    axes[i].set_title(title)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('distance_classifier_2x2_groups.png', dpi=300, bbox_inches='tight')
plt.close() 