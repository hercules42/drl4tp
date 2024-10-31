import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter

def read_and_parse_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 忽略空行
                data.extend(map(float, line.split()))
    return data

# 自定义格式化函数
def to_percent(y, position):
    return f'{y * 100:.0f}%'

def plot_data_hgnn(hg_plot, mpnn_plot, nognn_plot):

    hg_plot = np.insert(hg_plot, 0, np.nan)

    mpnn_plot_plot = np.insert(mpnn_plot, 0, np.nan)
    nognn_plot = np.insert(nognn_plot, 0, np.nan)
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(hg_plot)), hg_plot, marker='', label='w/ HGNN', alpha=0.5)
    plt.plot(range(len(mpnn_plot_plot)), mpnn_plot_plot, marker='', label='w/ GNN', color='red')
    plt.plot(range(len(nognn_plot)), nognn_plot, marker='', label='w/o HGNN or GNN', color='green')
    # 将纵坐标刻度格式化为百分数
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(to_percent))

    plt.legend(
    fontsize=10,               # 字体大小
    loc='upper right',         # 位置
    frameon=True,              # 是否绘制边框
    facecolor='white',         # 背景颜色
    framealpha=0.7,            # 边框透明度
    edgecolor='black',         # 边框颜色
    borderpad=1.5              # 边框与内容的间距
    )
    plt.xlabel('epoches')
    plt.ylabel('Validation optimality gap')
    plt.grid(True)

    plt.savefig('figure_hgnn_curve.eps', format='eps')

    plt.show()

def plot_data(gb_plot, pg_plot, ac_plot, mpnn_plot, nognn_plot):

    
    fig, ax = plt.subplots(1,2, figsize=(16, 6))
    
    gb_plot = np.insert(gb_plot, 0, np.nan)
    pg_plot = np.insert(pg_plot, 0, np.nan)
    ac_plot = np.insert(ac_plot, 0, np.nan)


    ax[0].plot(range(len(gb_plot)), gb_plot, marker='', label='w/ greedy rollout baseline', alpha=0.5)
    ax[0].plot(range(len(pg_plot)), pg_plot, marker='', label='w/o baseline', color='red')
    ax[0].plot(range(len(ac_plot)), ac_plot, marker='', label='w/ learnable critic', color='green')
    # 将纵坐标刻度格式化为百分数
    ax[0].yaxis.set_major_formatter(mticker.PercentFormatter(1))

    ax[0].legend(
    fontsize=15,               # 字体大小
    loc='upper right',         # 位置
    frameon=True,              # 是否绘制边框
    facecolor='white',         # 背景颜色
    framealpha=0.7,            # 边框透明度
    edgecolor='black',         # 边框颜色
    borderpad=1.5              # 边框与内容的间距
    )
    ax[0].set_xlabel('epoches')
    ax[0].set_ylabel('Validation optimality gap')
    ax[0].grid(True)

    mpnn_plot = np.insert(mpnn_plot, 0, np.nan)
    nognn_plot = np.insert(nognn_plot, 0, np.nan)

    ax[1].plot(range(len(gb_plot)), gb_plot, marker='', label='w/ HGNN', alpha=0.5)
    ax[1].plot(range(len(mpnn_plot)), mpnn_plot, marker='', label='w/ GNN', color='red')
    ax[1].plot(range(len(nognn_plot)), nognn_plot, marker='', label='w/o HGNN or GNN', color='green')
    # 将纵坐标刻度格式化为百分数
    
    ax[1].yaxis.set_major_formatter(mticker.PercentFormatter(1))
    ax[1].legend(
    fontsize=15,               # 字体大小
    loc='upper right',         # 位置
    frameon=True,              # 是否绘制边框
    facecolor='white',         # 背景颜色
    framealpha=0.7,            # 边框透明度
    edgecolor='black',         # 边框颜色
    borderpad=1.5              # 边框与内容的间距
    )
    ax[1].set_xlabel('epoches')
    ax[1].set_ylabel('Validation optimality gap')
    ax[1].grid(True)

    fig.tight_layout()  # 调整下方留白



    # 调整子图之间的间距
    fig.subplots_adjust(wspace=0.2)  # 设置水平和垂直间距
    fig.savefig('two_curve.eps', format='eps')

    plt.show()

# 文件名
hg_file = 'valid_reward_curve_100e_hg_state.txt'
mpnn_file = 'valid_reward_curve_100e_mpnn_state.txt'
nognn_file = 'valid_reward_curve_100e_nognn.txt'
pg_file = 'valid_reward_curve_100e_pg_state.txt'
ac = 'valid_reward_curve_100e_ac_state.txt'


# 读取和解析文件数据
hg_data = read_and_parse_file(hg_file)
mpnn_data = read_and_parse_file(mpnn_file)
nognn_data = read_and_parse_file(nognn_file)
pg_data = read_and_parse_file(pg_file)
ac_data = read_and_parse_file(ac)

hg_data = [x/6.85 -1  for x in hg_data]
mpnn_data = [x/6.85 -1  for x in mpnn_data]
nognn_data = [x/6.85 -1  for x in nognn_data]

pg_data = [x/6.85 -1  for x in pg_data]
ac_data = [x/6.85 -1  for x in ac_data]

# 平滑数据
window_size = 3  # 必须是奇数
poly_order = 2
hg_data = savgol_filter(hg_data, window_size, poly_order)
mpnn_data = savgol_filter(mpnn_data, window_size, poly_order)
nognn_data = savgol_filter(nognn_data, window_size, poly_order)
pg_data = savgol_filter(pg_data, window_size, poly_order)
ac_data = savgol_filter(ac_data, window_size, poly_order)
# 绘制折线图
plot_data(hg_data, pg_data, ac_data,mpnn_data, nognn_data)



