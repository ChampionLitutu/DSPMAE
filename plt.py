# import matplotlib.pyplot as plt
#
# # 如果不指定x，默认从0开始的自然数序列
# x = [1, 2, 3, 4, 5, 6]
#
# y1 = [84.41, 84.62, 84.56, 84.44, 84.41, 84.31]
# y2 = [73.44, 73.37, 73.34, 73.11, 73.03, 73.01]
# plt.plot(x, y1, label='Cora', marker='o', color='green', linestyle='-', linewidth=1)
# plt.plot(x, y2, label='CiteSeer', marker='o', color='red', linestyle='-', linewidth=1)
# plt.title('Cora')
# plt.ylim(0, 25)  # 设置 y 轴范围
# # plt.legend(fontsize=12)
# plt.ylabel('Accuracy (%)')
# plt.xlabel('Prompt Edge Number')
# plt.grid(alpha=0.5)
#
# plt.tight_layout()
# plt.show()
#

import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5, 6]
y1 = [84.41, 84.62, 84.56, 84.44, 84.41, 84.31]
y2 = [73.44, 73.37, 73.34, 73.11, 73.03, 73.01]

# 创建图形
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
fig, ax1 = plt.subplots()

# 绘制 y1 数据
ax1.plot(x, y1, label='Cora      ', marker='o', color='green', linestyle='-', linewidth=1)
ax1.set_xlabel('Prompt Edge Number')
ax1.set_ylabel('Accuracy (%)', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# ax1.legend()
# 创建共享 x 轴的第二个 y 轴
ax2 = ax1.twinx()
ax2.plot(x, y2, label='CiteSeer', marker='s', color='red', linestyle='-', linewidth=1)
ax2.set_ylabel('Accuracy (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper right', bbox_to_anchor=(1, 1.0), labelspacing=0.5, frameon=False)
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9), labelspacing=0.5, frameon=False)

plt.grid(True)
# 设置图例
fig.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
#
# # 数据
# x = [1, 2, 3, 4, 5, 6]
# y1 = [84.41, 84.62, 84.56, 84.44, 84.41, 84.31]
# y2 = [73.44, 73.37, 73.34, 73.11, 73.03, 73.01]
#
# # 统一比例尺范围，考虑两个数据的最小值和最大值
# y_min = min(min(y1), min(y2)) - 1  # 适当向下扩展
# y_max = max(max(y1), max(y2)) + 1  # 适当向上扩展
#
# # 绘制曲线
# plt.plot(x, y1, label='Cora', marker='o', color='green', linestyle='-', linewidth=1)
# plt.plot(x, y2, label='CiteSeer', marker='o', color='red', linestyle='-', linewidth=1)
#
# # 设置统一的 y 轴范围
# plt.ylim(y_min, y_max)
#
# # 添加标签和图例
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Unified Scale for Two Curves')
# plt.grid(True)  # 可选：添加网格方便观察
# plt.show()