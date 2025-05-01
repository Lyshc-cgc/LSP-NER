from matplotlib import rcParams, colors
import matplotlib.pyplot as plt
import numpy as np
import os

vis_path = '../data/vis'
res_template = ('../data/{dataset}/eval/span_bio/{method}/{method_setting}/simple_description'
                '/random_sampling/batch_qa/{model}/{model_setting}_res.txt')
# key is the model name, value is the model name in the plot
models = {
    'Qwen1.5': 'Qwen',
    'Mistral': 'Mixtral',
    'deepseek-v3': 'Deepseek'
}
metric = 'f1'


line_colors = ['#F09BA0', '#9BBBE1', '#8FBC8F']
error_colors = ['#8D0405', '#060270', '#006400']
markers = ['o', '^', 's', 'd']

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
                      legend_title='accuracy',
                      tick_step=1,
                      group_gap=0.2,
                      bar_gap=0
                      ):
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
    for i, (y, proportion, std, color) in enumerate(zip(datas, groups, errors, colors)):
        ax.bar(x + i * bar_span, y, bar_width, color=color,
               yerr=std, label=f'{proportion}', error_kw=error_params)
    ax.set_ylabel('micro-f1 (%)')
    ax.set_title(title)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels)
    ax.legend(title=legend_title, loc='upper right', frameon=True, fancybox=True, framealpha=0.7)
    ax.grid(True, linestyle=':', alpha=0.6)

def read_metric_from_txt_file(source_file) -> float:
    with open(source_file, 'r') as f:
        for line in f:
            if line.startswith(f'{metric}:'):
                metric_value = float(line.split(':')[-1].strip())
                return metric_value

class vis_theis:
    def __init__(self, file_type):
        self.base_path = '../data/vis/theis'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.file_type = file_type

    def vis_demo_num_part1(self):
        """
        In part 1, visualize the repeat number for demonstrations
        :param data:
        :param error:
        :param file_name:
        :return:
        """
        error_bar = True
        datasets = ('ontonotes5_en', 'mit_movies', 'ontonotes5_zh', 'CMeEE_V2')
        k_shots = (1, 5)
        seeds = (22, 32, 42)
        demo_nums = (1, 2, 3, 4, 5, 6)
        method = 'mt_fs'

        x_label = 'duplicating nums'
        y_label = 'micro-f1 (%)'
        demo_num_dir = os.path.join(self.base_path, 'part1/demo_num')
        for dataset in datasets:
            save_file = os.path.join(demo_num_dir, f'{dataset}.{self.file_type}')

            # each k_shot has a different ax
            fig, axs = plt.subplots(1, len(k_shots), figsize=(5 * len(k_shots), 5), layout="compressed")
            for shot_idx, k_shot in enumerate(k_shots):
                axs[shot_idx].set_title(f'{k_shots[shot_idx]}-shot')
                axs[shot_idx].set_ylabel(y_label)
                axs[shot_idx].set_xlabel(x_label)

                for model_idx, model in enumerate(models.keys()):
                    model_f1_scores = []  # each model has a different line
                    model_f1_stds = []
                    for demo_num in demo_nums:
                        method_setting = f'rep_{demo_num}'
                        f1_scores = []
                        for seed in seeds:
                            model_setting = f'{model}-mt-{k_shot}-shot-{seed}'
                            res_file = res_template.format(
                                dataset=dataset,
                                method=method,
                                method_setting=method_setting,
                                model=model,
                                model_setting=model_setting
                            )
                            f1_scores.append(read_metric_from_txt_file(res_file))
                        model_f1_scores.append(np.mean(f1_scores))
                        model_f1_stds.append(np.std(f1_scores, ddof=1))  # ddof=1, sample std. ddof=0, population std

                    # plot
                    if error_bar:
                       axs[shot_idx].errorbar(
                           demo_nums,
                           model_f1_scores,
                           yerr=model_f1_stds,
                           color=line_colors[model_idx],
                           fmt='-',
                           marker=markers[model_idx],
                           label=models[model],
                           ecolor=error_colors[model_idx],
                           elinewidth=1,
                           capsize=3
                       )
                    else:
                        axs[shot_idx].plot(
                            demo_nums,
                            model_f1_scores,
                            f'-{markers[model_idx]}',
                            color=line_colors[model_idx],
                            label=models[model]
                        )
                    axs[shot_idx].legend(title='models', loc='upper right', frameon=True, fancybox=True)
                    axs[shot_idx].grid(True, linestyle=':', alpha=0.6)
            print('save file:', save_file)
            fig.savefig(save_file, dpi=300)

    def vis_label_proportion_part1(self,):
        """
        In part 1, visualize the proportion of corrected labeled data

        :param data:
        :param error:
        :param file_name: the file name to be saved
        :return:
        """

        datasets = ('ontonotes5_en', 'mit_movies', 'ontonotes5_zh', 'CMeEE_V2')
        k_shots = (1, 5)
        seeds = (22, 32, 42)
        label_portions = (1, 0.75, 0.5, 0.25, 0)
        bar_colors = ['#e281b1', '#e89fa7', '#ecb6a1', '#f3cf9c', '#fef795']  # colors for each label portion
        method = 'mt_fs'

        # x_label = 'models'
        y_label = 'micro-f1 (%)'
        demo_num_dir = os.path.join(self.base_path, 'part1/label_proportion')
        for dataset in datasets:
            save_file = os.path.join(demo_num_dir, f'{dataset}.{self.file_type}')

            # each model has a different ax
            fig, axs = plt.subplots(1, len(k_shots), figsize=(5 * len(k_shots), 5), layout="compressed")
            for shot_idx, k_shot in enumerate(k_shots):
                axs[shot_idx].set_title(f'{k_shots[shot_idx]}-shot')
                axs[shot_idx].set_ylabel(y_label)
                # axs[shot_idx].set_xlabel(x_label)

                shot_datas, shot_stds = [], []  # store data/stds for each shot
                for label_portion in label_portions:
                    label_portion_f1_scores = []  # store f1 scores for each model
                    label_portion_f1_stds = []  # store f1 std for each model
                    for model_idx, model in enumerate(models.keys()):
                        method_setting = 'rep_1'  # fixed value
                        f1_scores = []
                        for seed in seeds:
                            if label_portion == 1:
                                model_setting = f'{model}-mt-{k_shot}-shot-is-{seed}'
                            else:
                                model_setting = f'{model}-mt-{k_shot}-shot-is-lmp_{label_portion}-{seed}'
                            res_file = res_template.format(
                                dataset=dataset,
                                method=method,
                                method_setting=method_setting,
                                model=model,
                                model_setting=model_setting
                            )
                            f1_scores.append(read_metric_from_txt_file(res_file))
                        label_portion_f1_scores.append(np.mean(f1_scores))
                        label_portion_f1_stds.append(np.std(f1_scores, ddof=1))  # ddof=1, sample std. ddof=0, population std
                    shot_datas.append(label_portion_f1_scores)
                    shot_stds.append(label_portion_f1_stds)

                create_multi_bars(
                    axs[shot_idx],
                    models.values(),
                    f'{k_shot}-shot',
                    shot_datas,
                    shot_stds,
                    bar_colors,
                    label_portions,
                    legend_title='accuracy',
                    )

            print('save file:', save_file)
            fig.savefig(save_file, dpi=300)

    def vis_partition_times(self):
        """
        In part 2, visualize the partition times for demonstrations, subset candidate
        :return:
        """

        error_bar = True
        datasets = ('ontonotes5_en', 'mit_movies', 'ontonotes5_zh', 'CMeEE_V2')
        k_shots = (1, 5)
        seeds = (22, 32, 42)
        rep_nums = (1, 2, 3, 4, 5, 6)
        subset_size = 0.5
        method = 'sc_fs'

        x_label = 'partition times'
        y_label = 'micro-f1 (%)'
        part_times_dir = os.path.join(self.base_path, 'part2/partition_times')
        for dataset in datasets:
            save_file = os.path.join(part_times_dir, f'{dataset}.{self.file_type}')

            # each k_shot has a different ax
            fig, axs = plt.subplots(1, len(k_shots), figsize=(5 * len(k_shots), 5), layout="compressed")
            for shot_idx, k_shot in enumerate(k_shots):
                axs[shot_idx].set_title(f'{k_shots[shot_idx]}-shot')
                axs[shot_idx].set_ylabel(y_label)
                axs[shot_idx].set_xlabel(x_label)

                for model_idx, model in enumerate(models.keys()):
                    model_f1_scores = []  # each model has a different line
                    model_f1_stds = []
                    for rep_num in rep_nums:
                        method_setting = f'{method}-size_{subset_size}-rep_{rep_num}'
                        f1_scores = []
                        for seed in seeds:
                            model_setting = f'{model}-sc-{k_shot}-shot-{seed}'
                            res_file = res_template.format(
                                dataset=dataset,
                                method=method,
                                method_setting=method_setting,
                                model=model,
                                model_setting=model_setting
                            )
                            f1_scores.append(read_metric_from_txt_file(res_file))
                        model_f1_scores.append(np.mean(f1_scores))
                        model_f1_stds.append(np.std(f1_scores, ddof=1))  # ddof=1, sample std. ddof=0, population std

                    # plot
                    if error_bar:
                        axs[shot_idx].errorbar(
                            rep_nums,
                            model_f1_scores,
                            yerr=model_f1_stds,
                            color=line_colors[model_idx],
                            fmt='-',
                            marker=markers[model_idx],
                            label=models[model],
                            ecolor=error_colors[model_idx],
                            elinewidth=1,
                            capsize=3
                        )
                    else:
                        axs[shot_idx].plot(
                            rep_nums,
                            model_f1_scores,
                            f'-{markers[model_idx]}',
                            color=line_colors[model_idx],
                            label=models[model]
                        )
                    axs[shot_idx].legend(title='models', loc='upper right', frameon=True, fancybox=True)
                    axs[shot_idx].grid(True, linestyle=':', alpha=0.6)
            print('save file:', save_file)
            fig.savefig(save_file, dpi=300)

    def vis_subset_size_part2(self):
        """
        In part 2, visualize the label subset size
        :return:
        """

        error_bar = True
        datasets = ('ontonotes5_en', 'mit_movies', 'ontonotes5_zh', 'CMeEE_V2')
        k_shots = (1, 5)
        seeds = (22, 32, 42)
        rep_num = 1
        subset_sizes = (0.1, 0.2, 0.3, 0.4, 0.5)
        method = 'sc_fs'

        x_label = 'subset proportion'
        y_label = 'micro-f1 (%)'
        part_times_dir = os.path.join(self.base_path, 'part2/subset_proportion')
        for dataset in datasets:
            save_file = os.path.join(part_times_dir, f'{dataset}.{self.file_type}')

            # each k_shot has a different ax
            fig, axs = plt.subplots(1, len(k_shots), figsize=(5 * len(k_shots), 5), layout="compressed")
            for shot_idx, k_shot in enumerate(k_shots):
                axs[shot_idx].set_title(f'{k_shots[shot_idx]}-shot')
                axs[shot_idx].set_ylabel(y_label)
                axs[shot_idx].set_xlabel(x_label)

                for model_idx, model in enumerate(models.keys()):
                    model_f1_scores = []  # each model has a different line
                    model_f1_stds = []
                    for subset_size in subset_sizes:
                        method_setting = f'{method}-size_{subset_size}-rep_{rep_num}'
                        f1_scores = []
                        for seed in seeds:
                            model_setting = f'{model}-sc-{k_shot}-shot-{seed}'
                            res_file = res_template.format(
                                dataset=dataset,
                                method=method,
                                method_setting=method_setting,
                                model=model,
                                model_setting=model_setting
                            )
                            f1_scores.append(read_metric_from_txt_file(res_file))
                        model_f1_scores.append(np.mean(f1_scores))
                        model_f1_stds.append(np.std(f1_scores, ddof=1))  # ddof=1, sample std. ddof=0, population std

                    # plot
                    if error_bar:
                        axs[shot_idx].errorbar(
                            subset_sizes,
                            model_f1_scores,
                            yerr=model_f1_stds,
                            color=line_colors[model_idx],
                            fmt='-',
                            marker=markers[model_idx],
                            label=models[model],
                            ecolor=error_colors[model_idx],
                            elinewidth=1,
                            capsize=3
                        )
                    else:
                        axs[shot_idx].plot(
                            subset_sizes,
                            model_f1_scores,
                            f'-{markers[model_idx]}',
                            color=line_colors[model_idx],
                            label=models[model]
                        )
                    axs[shot_idx].legend(title='models', loc='upper right', frameon=True, fancybox=True)
                    axs[shot_idx].grid(True, linestyle=':', alpha=0.6)
            print('save file:', save_file)
            fig.savefig(save_file, dpi=300)

    def main_thesis(self):
        """
        main function for part 1 and 2 of my thesis
        :return:
        """

        # 1. part 1
        # 1.1 the number of demonstrations
        self.vis_demo_num_part1()

        # 1.2. the proportion of corrected labeled data
        self.vis_label_proportion_part1()

        # 2. part 2
        # 2.1 the repeat number for demonstrations, subset candidate
        self.vis_partition_times()

        # 2.2 the label subset size
        self.vis_subset_size_part2()

if __name__ == '__main__':
    file_type = 'pdf'  # 'pdf', 'png'
    vt = vis_theis(file_type)
    vt.main_thesis()
