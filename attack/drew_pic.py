import os
import pandas
import numpy
import matplotlib.pyplot as plt
from itertools import chain

model_list = ["FDSR_HesRFA_32", "FDSR_HesRFA_32_FDL_l2_gaus", "DeFiAN", "MSRN", "OISR"]
name_list = ["FDSR", "FDSR w/ FDL", "DeFiAN-s", "MSRN", "OISR-RK2"]
base_model = ["edsr_baseline", "carn"]
model_name_in_title = ["EDSR_baseline", "CARN"]
eps = [1, 2, 4, 8, 16, 32]


root = "test_result"

results = []

for folder, name in zip(model_list, name_list):
    result_file = os.path.join(root, folder, "net_x4_1000_Y.csv")
    csv_data = pandas.read_csv(result_file)
    res = numpy.array(csv_data.values)
    test_set = res[:, 0]
    res = res[:, 1]
    results.append(list(res))

results = numpy.array(results)
results = numpy.transpose(results)
# print(results)

plt.figure(figsize = (16, 5))
plt.tight_layout()
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=0.5)

for model, title_model_name in zip(base_model, model_name_in_title):
    # print(base_set, eps)
    if model == base_model[0]:
        plt.subplot(121)
        curr_data = results[:6, :]
    elif model == base_model[1]:
        plt.subplot(122)
        curr_data = results[6:, :]
    
    curr_data = numpy.transpose(curr_data)
    print(curr_data)
    tick_step = 1
    group_gap = 0.2
    bar_gap = 0
    x = numpy.arange(len(curr_data[0])) * 1
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(curr_data)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = tick_step - group_gap
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # 绘制柱子
    for index, y in enumerate(curr_data):
        plt.bar(x + index*bar_span, y, bar_width, label=name_list[index])
    plt.ylim(numpy.min(curr_data)-1, numpy.max(curr_data)+0.5)
    plt.ylabel('PSNR/dB')
    plt.xlabel("α(/255)")
    title_str = "(a) " if model == base_model[0] else "(b) "
    title_str += 'Generated based '+ title_model_name
    plt.title(title_str, y=-0.25, fontproperties="Times New Roman", fontsize=15)
    # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, eps)
    plt.legend()
plt.show()