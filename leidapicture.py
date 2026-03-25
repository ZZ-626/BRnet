import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# ==========================================
# 1. 数据准备
# ==========================================



data_arcade = {

}

data_zz = {

}

# ==========================================
# 2. 颜色配置
# ==========================================
colors_list = plt.cm.tab10(np.linspace(0, 1, 10))
model_color_map = {
    'Unet': colors_list[2],
    'Unet++': colors_list[4],
    'Transunet': colors_list[5],
    'UTnet': colors_list[6],
    'SwinUnet': colors_list[8],
    'Mask2Former': colors_list[9],
    'Sam2': '#8c564b',
    'Biformer': '#7f7f7f',
    'EAG-SAE-ESTF': '#1f77b4',  # 蓝色
    'BRNet (Ours)': '#d62728'  # 红色
}


# ==========================================
# 3. 绘图函数 (字体加大版)
# ==========================================


def plot_extra_large_radar(data_dict, title, filename_base):
    df = pd.DataFrame(data_dict)
    categories = list(df.columns[1:])
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # [修改点1] 画布尺寸加大到 16x16 英寸 (非常大)
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # [修改点2] 指标名称字号加大到 24 (加粗)
    # pad=30 让字离图稍微远一点，避免重叠
    plt.xticks(angles[:-1], categories, color='black', size=24, weight='bold')

    # [修改点3] 刻度数值字号加大到 18
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="#444444", size=18, weight='bold')
    plt.ylim(0, 1)

    # 绘图
    for i, row in df.iterrows():
        full_name = row['model']
        base_name = full_name.split('[')[0].strip()
        if "ours" in full_name.lower() or "brnet" in full_name.lower():
            base_name = "BRNet (Ours)"
            label_name = "BRNet (Ours)"
        else:
            label_name = full_name

        values = row.drop('model').values.flatten().tolist()
        values += values[:1]

        color = model_color_map.get(base_name, 'gray')

        # [修改点4] 线条宽度整体加大
        if "BRNet" in base_name:
            lw, alpha, zorder = 5.0, 1.0, 10  # 我们的模型：极粗
        elif "EAG" in base_name:
            lw, alpha, zorder = 4.0, 0.9, 9  # 次优模型：很粗
        else:
            lw, alpha, zorder = 2.0, 0.5, 1  # 其他模型：普通粗细

        ax.plot(angles, values, linewidth=lw, linestyle='solid', label=label_name, color=color, alpha=alpha,
                zorder=zorder)

        if "BRNet" in base_name or "EAG" in base_name:
            ax.fill(angles, values, color=color, alpha=0.08)

    # [修改点5] 标题字号加大到 36
    plt.title(title, size=36, weight='bold', y=1.08, pad=40)

    # [修改点6] 图例字号加大到 18
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=18, frameon=False)

    plt.tight_layout()

    # 保存
    svg_name = f"{filename_base}.svg"
    plt.savefig(svg_name, format='svg', bbox_inches='tight')

    pdf_name = f"{filename_base}.pdf"
    plt.savefig(pdf_name, format='pdf', bbox_inches='tight')

    png_name = f"{filename_base}.png"
    plt.savefig(png_name, dpi=600, bbox_inches='tight')

    print(f"已生成终极加大版: {png_name}, {pdf_name}, {svg_name}")
    plt.close()




# ==========================================
# 4. 执行
# ==========================================

plot_extra_large_radar(data_arcade, "", "Radar_ARCADE_ExtraLarge")
plot_extra_large_radar(data_zz, "", "Radar_ZZ_ExtraLarge")