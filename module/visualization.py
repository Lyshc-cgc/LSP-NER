from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os

from sympy.abc import alpha

vis_path = '../data/vis'

plt.rcParams["font.sans-serif"] = "DejaVu Sans Mono"
plt.rcParams['axes.unicode_minus'] = False

# for mini version
# plt.rcParams.update({"font.size":20})  # controls default text size
# plt. rc ('axes', titlesize=20) # fontsize of the title
# plt. rc ('axes', labelsize=20) # fontsize of the x and y labels
# plt. rc ('xtick', labelsize=20) # fontsize of the x tick labels
# plt. rc ('ytick', labelsize=20) # fontsize of the y tick labels
# plt. rc ('legend', fontsize=20) # fontsize of the legend

def create_multi_bars(ax,
                      xlabels,
                      title,
                      datas,
                      errors,
                      colors,
                      groups,
                      tick_step=1,
                      group_gap=0.2,
                      bar_gap=0):
    '''
    生成多组数据的柱状图， refer to https://blog.csdn.net/mighty13/article/details/113873617

    :param ax: 子图对象
    :param xlabels: x轴坐标标签序列
    :param title: 图表标题
    :param datas: 数据集，二维列表，要求列表中每个一维列表的长度必须与xlabels的长度一致
    :param errors: 数据集的误差，二维列表，要求列表中每个一维列表的长度必须与labels的长度一致
    :param colors: 柱子颜色,对应每个组
    :param groups: 每个组的标签。
    :param tick_step: x轴刻度步长，默认为1，通过tick_step可调整x轴刻度步长。
    :param group_gap: 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    :param bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''

    # x为每组柱子x轴的基准位置
    x = np.arange(len(xlabels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap

    error_params = dict(elinewidth=1, capsize=3)  # 设置误差标记参数
    # 绘制柱子
    for i, (y, portion, std, color) in enumerate(zip(datas, groups, errors, colors)):
        ax.bar(x + i * bar_span, y, bar_width, color=color,
               yerr=std, label=f'{portion}', error_kw=error_params)
    ax.set_ylabel('micro-f1 (%)')
    ax.set_title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels)
    ax.legend(title='portion', loc='upper right',
              frameon=True, fancybox=True, framealpha=0.7)
    ax.grid(True, linestyle=':', alpha=0.6)


class vis_theis:
    def __init__(self):
        self.base_path = '../data/vis/theis'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def vis_demo_num_part1(self):
        """
        In part 1, visualize the repeat number for demonstrations
        :param data:
        :param error:
        :param file_name:
        :return:
        """
        def plot_errorbar(data, error, file_name, file_type='pdf'):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="compressed")
            axs = axs.flatten()
            # colors = ['#9BBBE1', '#F09BA0', '#EAB883', '#9D9EA3']
            colors = ['#F09BA0', '#9BBBE1']
            groups = ['Qwen', 'Mixtral']
            for i in range(len(axs)):
                axs[i].set_ylabel('micro-f1 (%)')
                axs[i].set_xlabel('repeat nums')
            axs[0].set_title('1-shot')
            axs[1].set_title('5-shot')
            x = np.arange(len(data[0]))
            # 1-shot
            axs[0].errorbar(x, data[0], yerr=error[0], color=colors[0], fmt='-', marker='o', label='Qwen', ecolor='#8D0405',
                            elinewidth=1, capsize=3)
            axs[0].errorbar(x, data[1], yerr=error[1], color=colors[1], fmt='-', marker='^', label='Mixtral', ecolor='#060270',
                            elinewidth=1, capsize=3)
            # 5-shot
            axs[1].errorbar(x, data[2], yerr=error[2], color=colors[0], fmt='-', marker='o', label='Qwen', ecolor='#8D0405',
                            elinewidth=1, capsize=3)
            axs[1].errorbar(x, data[3], yerr=error[3], color=colors[1], fmt='-', marker='^', label='Mixtral', ecolor='#060270',
                            elinewidth=1, capsize=3)
            axs[0].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True, framealpha=0.7)
            axs[1].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True, framealpha=0.7)

            # grid
            axs[0].grid(True, linestyle=':', alpha=0.6)
            axs[1].grid(True, linestyle=':', alpha=0.6)
            rep_num_path = os.path.join(self.base_path, 'part1/demo_num')
            if not os.path.exists(rep_num_path):
                os.makedirs(rep_num_path)

            file = os.path.join(rep_num_path, f'{file_name}.{file_type}')
            print('save file:', file)
            plt.savefig(file, dpi=300)

        # 1. conll03
        conll03_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [46.04, 47.24, 47.84, 47.87, 46.45, 43.19],
            [28.10, 27.50, 28.64, 28.43, 29.94, 28.95],  # mixtral
            # 5-shot
            [49.53, 50.34, 49.52, 49.38, 47.74, 49.72],  # qwen
            [44.40, 43.07, 40.61, 41.06, 42.72, 38.41],  # mixtral
        ]
        conll03_errors = [
            # 1-shot
            [1.61, 0.25, 0.88, 0.62, 5.25, 2.09],  # qwen, num 1, 2, 3, 4, 5, 6
            [0.44, 2.05, 1.16, 2.48, 3.06, 0.40],  # mixtral
            # 5-shot
            [3.59, 1.87, 2.58, 1.86, 2.56, 1.31],  # qwen
            [2.26, 2.27, 2.49, 3.19, 2.58, 1.67],  # mixtral
        ]
        plot_errorbar(conll03_datas, conll03_errors, 'conll03_num')

        # 2. onto5
        onto5_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [35.19, 36.92, 35.98, 36.58, 35.34, 35.06],
            [28.33, 27.07, 25.29, 17.99, 15.20, 17.71],  # mixtral
            # 5-shot
            [38.87, 42.19, 42.81, 40.91, 40.43, 37.99],  # qwen
            [19.08, 19.64, 16.47, 1.30, 2.01, 0.25],  # mixtral
        ]
        onto5_errors = [
            # 1-shot
            [1.48, 1.36, 2.48, 2.21, 2.96, 1.12],  # qwen, num 1, 2, 3, 4, 5, 6
            [1.00, 1.56, 0.67, 1.97, 2.12, 3.25],  # mixtral
            # 5-shot
            [2.45, 1.33, 1.50, 1.78, 1.64, 2.18],  # qwen
            [1.30, 1.15, 1.29, 0.82, 1.34, 0.35],  # mixtral
        ]
        plot_errorbar(onto5_datas, onto5_errors, 'onto5_num')

        # 3. movies
        movies_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [67.27, 66.01, 66.72, 67.44, 67.34, 67.28],
            [67.22, 68.71, 67.20, 67.09, 67.10, 67.64],  # mixtral
            # 5-shot
            [64.68, 68.29, 67.34, 67.34, 68.13, 68.53],  # qwen
            [71.03, 71.10, 62.21, 56.00, 26.94, 0.00],  # mixtral
        ]
        movies_errors = [
            # 1-shot
            [1.79, 1.40, 1.77, 1.04, 2.73, 1.26],  # qwen, num 1, 2, 3, 4, 5, 6
            [2.17, 0.71, 0.98, 1.06, 0.76, 0.35],  # mixtral
            # 5-shot
            [2.54, 1.83, 1.50, 1.22, 1.40, 0.89],  # qwen
            [0.70, 2.69, 0.56, 0.27, 8.30, 0.00],  # mixtral
        ]
        plot_errorbar(movies_datas, movies_errors, 'movies_num')

        # 4. restaurant
        restaurant_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [48.99, 49.44, 47.89, 48.40, 47.33, 46.95],
            [49.96, 49.32, 47.80, 49.33, 51.25, 50.71],  # mixtral
            # 5-shot
            [62.45, 64.43, 64.62, 65.27, 64.53, 63.37],  # qwen
            [62.95, 62.77, 60.51, 60.13, 58.02, 37.11],  # mixtral
        ]
        restaurant_errors = [
            # 1-shot
            [2.13, 2.55, 1.47, 1.37, 2.35, 1.90],  # qwen, num 1, 2, 3, 4, 5, 6
            [1.48, 1.11, 1.89, 0.51, 2.10, 0.21],  # mixtral
            # 5-shot
            [2.52, 2.72, 3.44, 3.31, 3.41, 2.70],  # qwen
            [1.29, 1.47, 1.20, 1.29, 2.05, 2.10],  # mixtral
        ]
        plot_errorbar(restaurant_datas, restaurant_errors, 'restaurant_num')

    def vis_label_portion_part1(self,):
        """
        In part 1, visualize the portion of corrected labeled data

        :param data:
        :param error:
        :param file_name: the file name to be saved
        :return:
        """
        def plot_bars(data, error, file_name, file_type='pdf'):
            models = ['Qwen', 'Mixtral']
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="compressed")
            axs = axs.flatten()

            groups = [1, 0.75, 0.5, 0.25, 0]
            colors = ['#e281b1', '#e89fa7', '#ecb6a1', '#f3cf9c', '#fef795']
            create_multi_bars(axs[0],
                              models,
                              '1-shot',
                              # (2, 5) -> (5, 2), 5个portion，2个模型
                              np.array(data[:2]).T,
                              np.array(error[:2]).T,
                              colors=colors,
                              groups=groups,
                              )
            create_multi_bars(axs[1],
                              models,
                              '5-shot',
                              np.array(data[2:]).T,
                              np.array(error[2:]).T,
                              colors=colors,
                              groups=groups,
                              )
            vis_path = os.path.join(self.base_path, 'part1/label_portion')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            file = os.path.join(vis_path, f'{file_name}.{file_type}')
            print('save file:', file)
            plt.savefig(file, dpi=300)

        # 1. conll03
        conll03_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [44.79, 44.55, 45.39, 43.32, 42.23],
            [26.22, 27.32, 27.73, 26.43, 32.26],  # mixtral
            # 5-shot
            [48.67, 47.89, 46.84, 44.16, 42.59],  # qwen
            [43.04, 39.13, 39.45, 40.45, 33.16],  # mixtral
        ]
        conll03_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [1.85, 1.47, 2.05, 3.52, 3.41],
            [1.91, 2.10, 3.54, 3.23, 3.58],  # mixtral
            # 5-shot
            [1.40, 2.85, 1.83, 1.37, 3.79],  # qwen
            [3.41, 4.01, 5.36, 1.37, 1.69],  # mixtral
        ]
        plot_bars(conll03_datas, conll03_errors, 'conll03')

        # 2. ontonotes5
        onto5_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [34.74, 32.50, 28.67, 26.40, 24.13],
            [26.39, 27.57, 25.32, 23.10, 19.59],  # mixtral
            # 5-shot
            [39.49, 31.96, 28.54, 27.33, 21.07],  # qwen
            [16.52, 17.90, 17.31, 13.68, 9.99],  # mixtral
        ]
        onto5_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [1.34, 2.45, 1.65, 1.70, 1.93],
            [2.95, 3.91, 2.30, 2.05, 1.58],  # mixtral
            # 5-shot
            [2.56, 2.77, 3.61, 3.74, 3.09],  # qwen
            [1.87, 1.30, 4.61, 1.50, 2.70],  # mixtral
        ]

        plot_bars(onto5_datas, onto5_errors, 'ontonotes5')

        # 3. movies
        movies_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [67.24, 66.57, 63.81, 59.88, 56.94],
            [68.25, 67.99, 64.75, 60.75, 58.47],  # mixtral
            # 5-shot
            [64.00, 60.65, 54.22, 53.83, 48.38],  # qwen
            [71.02, 70.45, 67.91, 63.22, 50.91],  # mixtral
        ]
        movies_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [0.92, 0.88, 2.80, 1.23, 2.22],
            [1.47, 2.49, 0.95, 2.57, 1.54],  # mixtral
            # 5-shot
            [2.61, 1.58, 2.39, 0.77, 2.57],  # qwen
            [1.04, 2.04, 0.95, 1.78, 4.40],  # mixtral
        ]

        plot_bars(movies_datas, movies_errors, 'movies')

        # 4. restaurant
        restaurant_datas = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [50.27, 50.15, 49.24, 47.18, 43.36],
            [48.71, 49.29, 46.74, 46.81, 43.59],  # mixtral
            # 5-shot
            [62.92, 61.03, 55.97, 53.62, 44.39],  # qwen
            [62.64, 60.82, 56.49, 54.41, 48.12],  # mixtral
        ]
        restaurant_errors = [
            # 1-shot
            # qwen, 对应5个portion, 1, 0.75, 0.5, 0.25, 0
            [2.70, 1.73, 1.49, 0.53, 0.59],
            [1.42, 0.92, 1.84, 1.37, 2.08],  # mixtral
            # 5-shot
            [2.35, 1.15, 1.02, 2.76, 3.16],  # qwen
            [1.17, 2.28, 2.04, 2.60, 2.87],  # mixtral
        ]

        plot_bars(restaurant_datas, restaurant_errors, 'restaurant')

    def vis_partition_times(self):
        """
        In part 2, visualize the partition times for demonstrations, subset candidate
        :return:
        """

        def plot_errorbar(data, error, file_name, file_type='pdf'):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="compressed")
            axs = axs.flatten()
            # colors = ['#9BBBE1', '#F09BA0', '#EAB883', '#9D9EA3']
            colors = ['#F09BA0', '#9BBBE1']
            groups = ['Qwen', 'Mixtral']
            for i in range(len(axs)):
                axs[i].set_ylabel('micro-f1 (%)')
                axs[i].set_xlabel('partition times')
            axs[0].set_title('1-shot')
            axs[1].set_title('5-shot')
            x = np.arange(len(data[0])) + 1
            # 1-shot
            axs[0].errorbar(x, data[0], yerr=error[0], color=colors[0], fmt='-', marker='o', label='Qwen',
                            ecolor='#8D0405', elinewidth=1, capsize=3)
            axs[0].errorbar(x, data[1], yerr=error[1], color=colors[1], fmt='-', marker='^', label='Mixtral',
                            ecolor='#060270', elinewidth=1, capsize=3)
            # 5-shot
            axs[1].errorbar(x, data[2], yerr=error[2], color=colors[0], fmt='-', marker='o', label='Qwen',
                            ecolor='#8D0405', elinewidth=1, capsize=3)
            axs[1].errorbar(x, data[3], yerr=error[3], color=colors[1], fmt='-', marker='^', label='Mixtral',
                            ecolor='#060270', elinewidth=1, capsize=3)
            axs[0].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True)
            axs[1].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True)

            # grid
            axs[0].grid(True, linestyle=':', alpha=0.6)
            axs[1].grid(True, linestyle=':', alpha=0.6)
            rep_num_path = os.path.join(self.base_path, 'part2/partition_times')
            if not os.path.exists(rep_num_path):
                os.makedirs(rep_num_path)

            file = os.path.join(rep_num_path, f'{file_name}.{file_type}')
            print('save file:', file)
            plt.savefig(file, dpi=300)

        # 1. conll03
        conll03_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [47.40, 47.62, 46.90, 47.86, 48.47, 46.11],
            [30.03, 29.52, 30.87, 33.84, 33.55, 31.37],  # mixtral
            # 5-shot
            [54.70, 54.58, 58.29, 57.77, 56.12, 53.52],  # qwen
            [46.74, 45.47, 47.20, 42.15, 30.51, 25.36],  # mixtral
        ]
        conll03_errors = [
            # 1-shot
            [2.48, 2.37, 1.54, 1.39, 2.31, 1.29],  # qwen, num 1, 2, 3, 4, 5, 6
            [2.80, 2.17, 2.46, 1.43, 2.05, 1.81],  # mixtral
            # 5-shot
            [1.34, 3.13, 2.76, 1.51, 2.66, 2.38],  # qwen
            [2.92, 1.33, 4.27, 2.51, 3.27, 3.52],  # mixtral
        ]
        plot_errorbar(conll03_datas, conll03_errors, 'conll03_part_times')

        # 2. onto5
        onto5_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [39.37, 40.58, 41.75, 42.49, 41.41, 42.62],
            [29.96, 26.07, 22.53, 23.26, 16.26, 6.99],  # mixtral
            # 5-shot
            [43.09, 44.81, 38.85, 34.96, 38.05, 38.60],  # qwen
            [21.11, 11.41, 1.22, 0.72, 0, 0],  # mixtral
        ]
        onto5_errors = [
            # 1-shot
            [2.01, 2.87, 4.36, 1.72, 1.09, 1.44],  # qwen, num 1, 2, 3, 4, 5, 6
            [2.68, 0.78, 1.03, 1.78, 3.40, 3.19],  # mixtral
            # 5-shot
            [2.06, 3.14, 2.23, 1.07, 1.87, 1.37],  # qwen
            [0.21, 4.74, 1.26, 0.60, 0, 0],  # mixtral
        ]
        plot_errorbar(onto5_datas, onto5_errors, 'onto5_part_times')

        # 3. movies
        movies_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [65.58, 67.59, 67.53, 67.12, 68.43, 67.50],
            [69.09, 69.85, 68.92, 67.56, 69.47, 67.55],  # mixtral
            # 5-shot
            [67.33, 69.15, 68.26, 70.09, 69.05, 70.28],  # qwen
            [72.52, 57.89, 50.14, 3.83, 0.79, 0.00],  # mixtral
        ]
        movies_errors = [
            # 1-shot
            [1.09, 1.55, 1.49, 1.26, 1.51, 1.65],  # qwen, num 1, 2, 3, 4, 5, 6
            [1.31, 1.81, 1.35, 2.14, 0.16, 0.35],  # mixtral
            # 5-shot
            [0.62, 1.93, 2.12, 1.66, 2.10, 1.15],  # qwen
            [2.43, 1.97, 3.63, 4.28, 0.65, 0.00],  # mixtral
        ]
        plot_errorbar(movies_datas, movies_errors, 'movies_part_times')

        # 4. restaurant
        restaurant_datas = [
            # 1-shot
            # qwen, num 1, 2, 3, 4, 5, 6
            [48.23, 49.31, 47.37, 48.28, 47.69, 49.38],
            [49.20, 51.44, 50.13, 53.79, 51.62, 48.44],  # mixtral
            # 5-shot
            [62.25, 64.79, 64.65, 64.49, 65.20, 66.37],  # qwen
            [63.28, 62.25, 55.36, 51.88, 46.11, 20.80],  # mixtral
        ]
        restaurant_errors = [
            # 1-shot
            [1.34, 3.10, 2.42, 3.34, 2.31, 2.26],  # qwen, num 1, 2, 3, 4, 5, 6
            [2.47, 1.23, 2.05, 1.24, 1.09, 0.55],  # mixtral
            # 5-shot
            [3.27, 2.13, 2.69, 2.51, 2.90, 2.69],  # qwen
            [1.10, 2.25, 1.77, 3.61, 2.09, 7.97],  # mixtral
        ]
        plot_errorbar(restaurant_datas, restaurant_errors, 'restaurant_part_times')

    def vis_subset_size_part2(self):
        """
        In part 2, visualize the label subset size
        :return:
        """

        def plot_errorbar(data, error, file_name, file_type='pdf'):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout="compressed")
            axs = axs.flatten()
            # colors = ['#9BBBE1', '#F09BA0', '#EAB883', '#9D9EA3']
            colors = ['#F09BA0', '#9BBBE1']
            groups = ['Qwen', 'Mixtral']
            for i in range(len(axs)):
                axs[i].set_ylabel('micro-f1 (%)')
                axs[i].set_xlabel('subset portion')
            axs[0].set_title('1-shot')
            axs[1].set_title('5-shot')
            x = np.arange(len(data[0])) * 0.1 + 0.1
            # 1-shot
            axs[0].errorbar(x, data[0], yerr=error[0], color=colors[0], fmt='-', marker='o', label='Qwen',
                            ecolor='#8D0405', elinewidth=1, capsize=3)
            axs[0].errorbar(x, data[1], yerr=error[1], color=colors[1], fmt='-', marker='^', label='Mixtral',
                            ecolor='#060270', elinewidth=1, capsize=3)
            # 5-shot
            axs[1].errorbar(x, data[2], yerr=error[2], color=colors[0], fmt='-', marker='o', label='Qwen',
                            ecolor='#8D0405', elinewidth=1, capsize=3)
            axs[1].errorbar(x, data[3], yerr=error[3], color=colors[1], fmt='-', marker='^', label='Mixtral',
                            ecolor='#060270', elinewidth=1, capsize=3)
            axs[0].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True)
            axs[1].legend(title='models', loc='upper right',
                          frameon=True, fancybox=True)

            # grid
            axs[0].grid(True, linestyle=':', alpha=0.6)
            axs[1].grid(True, linestyle=':', alpha=0.6)
            rep_num_path = os.path.join(self.base_path, 'part2/subset_portion')
            if not os.path.exists(rep_num_path):
                os.makedirs(rep_num_path)

            file = os.path.join(rep_num_path, f'{file_name}.{file_type}')
            print('save file:', file)
            plt.savefig(file, dpi=300)

        # 1. conll03
        conll03_datas = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [47.47, 48.92, 47.38, 47.55, 47.40],
            [31.55, 30.91, 31.27, 32.00, 30.03],  # mixtral
            # 5-shot
            [53.37, 53.25, 52.35, 53.85, 54.70],  # qwen
            [45.08, 44.46, 45.87, 44.68, 46.74],  # mixtral
        ]
        conll03_errors = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [1.84, 2.52, 1.73, 3.12, 2.48],
            [0.89, 1.80, 2.96, 0.58, 2.80],  # mixtral
            # 5-shot
            [3.60, 2.06, 3.61, 1.15, 1.34],  # qwen
            [1.52, 3.94, 4.12, 3.17, 2.92],  # mixtral
        ]
        plot_errorbar(conll03_datas, conll03_errors, 'conll03_subset_portion')

        # 2. onto5
        onto5_datas = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [40.36, 40.95, 40.31, 40.51, 39.37],
            [3.68, 23.07, 29.77, 29.00, 29.96],  # mixtral
            # 5-shot
            [31.83, 41.13, 43.58, 43.26, 43.09],  # qwen
            [0.68, 13.48, 15.46, 15.49, 21.11],  # mixtral
        ]
        onto5_errors = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [1.05, 1.42, 1.42, 3.37, 2.01],
            [2.77, 3.95, 2.55, 3.38, 2.68],  # mixtral
            # 5-shot
            [0.89, 2.19, 0.88, 0.40, 2.06],  # qwen
            [0.96, 3.66, 1.25, 3.01, 0.21],  # mixtral
        ]
        plot_errorbar(onto5_datas, onto5_errors, 'onto5_subset_portion')

        # 3. movies
        movies_datas = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [64.23, 66.90, 66.21, 65.77, 65.58],
            [65.79, 66.91, 67.57, 68.94, 69.09],  # mixtral
            # 5-shot
            [66.32, 67.37, 66.12, 66.55, 67.33],  # qwen
            [28.37, 67.64, 70.06, 69.88, 72.52],  # mixtral
        ]
        movies_errors = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [0.12, 1.30, 0.38, 0.30, 1.09],
            [1.57, 1.55, 1.68, 0.94, 1.31],  # mixtral
            # 5-shot
            [1.53, 1.06, 2.42, 3.10, 0.62],  # qwen
            [10.89, 1.01, 1.69, 0.39, 2.43],  # mixtral
        ]
        plot_errorbar(movies_datas, movies_errors, 'movies_subset_portion')

        # 4. restaurant
        restaurant_datas = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [45.89, 47.17, 46.79, 49.14, 48.23],
            [47.55, 47.61, 47.71, 50.59, 49.20],  # mixtral
            # 5-shot
            [60.88, 61.91, 62.75, 64.12, 62.25],  # qwen
            [58.81, 59.25, 60.55, 60.76, 63.28],  # mixtral
        ]
        restaurant_errors = [
            # 1-shot
            # qwen, portion 0.1, 0.2, 0.3, 0.4, 0.5
            [2.68, 3.43, 3.17, 1.11, 1.34],
            [0.93, 2.19, 1.43, 1.49, 2.47],  # mixtral
            # 5-shot
            [3.53, 3.37, 3.33, 2.90, 3.27],  # qwen
            [2.12, 3.07, 3.37, 2.02, 1.10],  # mixtral
        ]
        plot_errorbar(restaurant_datas, restaurant_errors,
                      'restaurant_subset_portion')

    def main_thesis(self):
        """
        main function for part 1 and 2 of my thesis
        :return:
        """

        # 1. part 1
        # 1.1 the number of demonstrations
        self.vis_demo_num_part1()

        # 1.2. the portion of corrected labeled data
        self.vis_label_portion_part1()

        # 2. part 2
        # 2.1 the repeat number for demonstrations, subset candidate
        self.vis_partition_times()

        # 2.2 the label subset size
        self.vis_subset_size_part2()


def main_paper():
    """
    main function for my paper
    :return:
    """
    # 1. Qwen
    model_name = 'Qwen'
    # 1.1 conll03
    conll03_datas = [
        # 1-shot
        [44.79, 44.55, 45.39, 43.32],  # mt,对应四个portion
        [52.21, 52.46, 49.78, 50.70],  # ours, sc-0.5-r1
        # 5-shot
        [48.67, 47.89, 46.84, 44.16],  # mt
        [50.05, 49.61, 50.75, 51.23],  # ours, sc-0.5-r1
    ]
    conll03_errors = [
        # 1-shot
        [1.85, 1.47, 2.05, 3.52],  # mt
        [1.86, 1.99, 2.93, 2.45],  # ours, sc-0.5-r1
        # 5-shot
        [1.40, 2.85, 1.83, 1.37],  # mt
        [0.93, 1.04, 0.90, 0.78],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(conll03_datas, conll03_errors, model_name, 'conll03')

    # 1.2 ontonotes5
    onto5_datas = [
        # 1-shot
        [34.74, 32.50, 28.67, 26.40],  # mt
        [38.49, 36.37, 34.26, 31.68],  # ours, sc-0.5-r1
        # 5-shot
        [39.49, 31.96, 28.54, 27.33],  # mt
        [36.58, 34.13, 29.66, 29.22],  # ours, sc-0.5-r1
    ]
    onto5_errors = [
        # 1-shot
        [1.34, 2.45, 1.65, 1.70],  # mt
        [1.35, 2.32, 2.36, 1.91],  # ours, sc-0.5-r1
        # 5-shot
        [2.56, 2.77, 3.61, 3.74],  # mt
        [2.17, 2.09, 1.34, 2.65],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(onto5_datas, onto5_errors, model_name, 'ontonotes5')

    # 1.3 movies
    movies_datas = [
        # 1-shot
        [67.24, 66.57, 63.81, 59.88],  # mt
        [64.39, 62.38, 58.99, 61.88],  # ours, sc-0.5-r1
        # 5-shot
        [64.00, 60.65, 54.22, 53.83],  # mt
        [65.92, 63.90, 60.83, 58.52],  # ours, sc-0.5-r1
    ]
    movies_errors = [
        # 1-shot
        [0.92, 0.88, 2.80, 1.23],  # mt
        [1.12, 0.89, 3.10, 1.54],  # ours, sc-0.5-r1
        # 5-shot
        [2.61, 1.58, 2.39, 0.77],  # mt
        [0.39, 1.95, 2.77, 0.86],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(movies_datas, movies_errors, model_name, 'movies')

    # 1.3 restaurant
    restaurant_datas = [
        # 1-shot
        [50.27, 50.15, 49.24, 47.18],  # mt
        [46.50, 47.32, 46.71, 48.23],  # ours, sc-0.5-r1
        # 5-shot
        [62.92, 61.03, 55.97, 53.62],  # mt
        [55.80, 56.45, 55.14, 54.51],  # ours, sc-0.5-r1
    ]
    restaurant_errors = [
        # 1-shot
        [2.70, 1.73, 1.49, 0.53],  # mt
        [3.06, 1.85, 2.24, 0.65],  # ours, sc-0.5-r1
        # 5-shot
        [2.35, 1.15, 1.02, 2.76],  # mt
        [1.79, 1.28, 2.70, 2.19],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(restaurant_datas, restaurant_errors, model_name, 'restaurant')

    # 2. Mixtral
    model_name = 'Mixtral'
    # 2.1 conll03
    conll03_datas = [
        # 1-shot
        [26.22, 27.32, 27.73, 26.43],  # mt
        [32.52, 34.08, 32.65, 33.00],  # ours, sc-0.5-r1
        # 5-shot
        [43.04, 39.13, 39.45, 40.45],  # mt
        [35.13, 35.24, 32.77, 31.22],  # ours, sc-0.5-r1
    ]
    conll03_errors = [
        # 1-shot
        [1.91, 2.10, 3.54, 3.23],  # mt
        [2.02, 3.06, 1.78, 3.27],  # ours, sc-0.5-r1
        # 5-shot
        [3.41, 4.01, 5.36, 1.37],  # mt
        [2.37, 0.39, 1.57, 0.48],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(conll03_datas, conll03_errors, model_name, 'conll03')

    # 2.2 ontonotes5
    onto5_datas = [
        # 1-shot
        [26.39, 27.57, 25.32, 23.10],  # mt
        [23.85, 22.24, 21.45, 20.67],  # ours, sc-0.5-r1
        # 5-shot
        [16.52, 17.90, 17.31, 13.68],  # mt
        [24.17, 22.21, 20.84, 20.14],  # ours, sc-0.5-r1
    ]
    onto5_errors = [
        # 1-shot
        [2.95, 3.91, 2.30, 2.05],  # mt
        [2.10, 3.64, 1.82, 1.52],  # ours, sc-0.5-r1
        # 5-shot
        [1.87, 1.30, 4.61, 1.50],  # mt
        [0.35, 0.54, 1.84, 1.38],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(onto5_datas, onto5_errors, model_name, 'ontonotes5')

    # 2.3 movies
    movies_datas = [
        # 1-shot
        [68.25, 67.99, 64.75, 60.75],  # mt
        [58.61, 59.30, 58.63, 55.48],  # ours, sc-0.5-r1
        # 5-shot
        [71.02, 70.45, 67.91, 63.22],  # mt
        [64.82, 64.26, 60.57, 61.35],  # ours, sc-0.5-r1
    ]
    movies_errors = [
        # 1-shot
        [1.47, 2.49, 0.95, 2.57],  # mt
        [2.17, 1.83, 4.02, 0.64],  # ours, sc-0.5-r1
        # 5-shot
        [1.04, 2.04, 0.95, 1.78],  # mt
        [2.58, 1.64, 0.55, 1.79],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(movies_datas, movies_errors, model_name, 'movies')

    # 2.3 restaurant
    restaurant_datas = [
        # 1-shot
        [48.71, 49.29, 46.74, 46.81],  # mt
        [40.01, 40.09, 39.97, 42.48],  # ours, sc-0.5-r1
        # 5-shot
        [62.64, 60.82, 56.49, 54.41],  # mt
        [51.64, 52.68, 52.85, 48.53],  # ours, sc-0.5-r1
    ]
    restaurant_errors = [
        # 1-shot
        [1.42, 0.92, 1.84, 1.37],  # mt
        [0.74, 2.24, 2.00, 1.16],  # ours, sc-0.5-r1
        # 5-shot
        [1.17, 2.28, 2.04, 2.60],  # mt
        [2.87, 2.81, 2.91, 3.46],  # ours, sc-0.5-r1
    ]
    # vis_label_portion(restaurant_datas, restaurant_errors, model_name, 'restaurant')


if __name__ == '__main__':
    vt = vis_theis()
    vt.main_thesis()
