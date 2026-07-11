import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []

def add_md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def add_code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ==================== Cell 1: Title ====================
add_md("""# 基于熵权法-灰色关联分析的公司综合评价

## 项目概述
本项目使用**熵权法（Entropy Weight Method）**确定各评价指标的客观权重，
再结合**灰色关联分析（Grey Relational Analysis, GRA）**对 a、b、c、d、e、f、g 七家公司进行综合评价与排序。

## 方法流程
1. **数据预处理**：根据指标类型（偏大型/偏小型/中间型）进行无量纲化处理
2. **熵权法求权重**：基于信息熵计算各指标的客观权重
3. **灰色关联分析**：计算各公司与参考序列的灰色关联度
4. **结果分析**：根据关联度排序，得出综合评价结论
""")

# ==================== Cell 2: Import ====================
add_code("""# ============================================
# 导入所需库
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

# 设置中文字体（Windows 使用 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 显示设置
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print("所有库导入成功！")
""")

# ==================== Cell 3: Load Data ====================
add_md("""## 1. 数据读取

从 Excel 文件中读取原始数据，包含：
- 19个评价指标（C1~C19）
- 3种指标类型：**偏大型**（效益型，越大越好）、**偏小型**（成本型，越小越好）、**中间型**（适中型，越接近某值越好）
- 7家待评价公司（a~g）
""")

add_code("""# ============================================
# 读取数据
# ============================================
df = pd.read_excel('company_data.xlsx')

# 查看原始数据
print("原始数据：")
print(df.to_string(index=False))

# 提取关键信息
indicators = df['指标'].tolist()        # 指标名称列表
indicator_types = df['指标类型'].tolist()  # 指标类型列表
companies = ['a', 'b', 'c', 'd', 'e', 'f', 'g']  # 公司列表

# 提取数值矩阵：每行是一家公司，每列是一个指标
data = df[companies].values.T  # shape: (7家公司, 19个指标)

print(f"\\n数据规模: {data.shape[0]} 家公司, {data.shape[1]} 个指标")
print(f"公司列表: {companies}")
print(f"\\n指标类型分布:")
for t in set(indicator_types):
    count = indicator_types.count(t)
    idx_list = [indicators[i] for i, typ in enumerate(indicator_types) if typ == t]
    print(f"  {t} ({count}个): {idx_list}")
""")

# ==================== Cell 4: Normalization ====================
add_md("""## 2. 数据归一化（无量纲化处理）

由于各指标的量纲和数量级不同，需要先进行归一化处理。归一化方法取决于指标类型：

- **偏大型（效益型）**：$x'_{ij} = \\frac{x_{ij} - \\min(x_j)}{\\max(x_j) - \\min(x_j)}$
- **偏小型（成本型）**：$x'_{ij} = \\frac{\\max(x_j) - x_{ij}}{\\max(x_j) - \\min(x_j)}$
- **中间型（适中型）**：$x'_{ij} = 1 - \\frac{|x_{ij} - x_{best}|}{\\max(|x_j - x_{best}|)}$

其中 $x_{best}$ 取该指标在所有公司中的**均值**作为最佳值。
""")

add_code("""# ============================================
# 数据归一化
# ============================================

def normalize_matrix(data, indicator_types):
    \"\"\"
    根据指标类型对数据矩阵进行归一化处理

    参数:
        data: numpy array, shape (n_companies, n_indicators)
        indicator_types: list, 每个指标的类型标签

    返回:
        norm_data: 归一化后的矩阵 (0~1之间)
    \"\"\"
    n_companies, n_indicators = data.shape
    norm_data = np.zeros_like(data, dtype=np.float64)

    for j in range(n_indicators):
        col = data[:, j].astype(np.float64)
        col_min = col.min()
        col_max = col.max()
        typ = indicator_types[j]

        if typ == '偏大型':  # 效益型：越大越好
            if col_max == col_min:
                norm_data[:, j] = 1.0  # 所有值相同则归一化为1
            else:
                norm_data[:, j] = (col - col_min) / (col_max - col_min)

        elif typ == '偏小型':  # 成本型：越小越好
            if col_max == col_min:
                norm_data[:, j] = 1.0
            else:
                norm_data[:, j] = (col_max - col) / (col_max - col_min)

        elif typ == '中间型':  # 适中型：越接近均值越好
            best_val = col.mean()  # 取均值作为最佳值
            max_deviation = np.abs(col - best_val).max()
            if max_deviation == 0:
                norm_data[:, j] = 1.0
            else:
                norm_data[:, j] = 1 - np.abs(col - best_val) / max_deviation
        else:
            raise ValueError(f"未知的指标类型: {typ}")

    return norm_data

# 执行归一化
norm_data = normalize_matrix(data, indicator_types)

# 将归一化结果转为 DataFrame 便于查看
norm_df = pd.DataFrame(norm_data.T, index=indicators, columns=companies).T
norm_df.index.name = '公司'

print("归一化后的数据（0~1，越接近1越优）：")
print(norm_df.round(4).to_string())
""")

# ==================== Cell 5: Entropy Weight ====================
add_md("""## 3. 熵权法计算指标权重

熵权法的核心思想：**信息熵越小，说明该指标数据的差异越大，提供的信息越多，权重应越大。**

### 计算步骤：
1. 计算每个指标下各公司的**比重**：$p_{ij} = \\frac{x'_{ij}}{\\sum_{i=1}^{n} x'_{ij}}$
2. 为避免 $p_{ij}=0$ 导致 $\\ln(0)$ 无意义，做微小平移修正
3. 计算**信息熵**：$e_j = -\\frac{1}{\\ln(n)} \\sum_{i=1}^{n} p_{ij} \\ln(p_{ij})$
4. 计算**权重**：$w_j = \\frac{1 - e_j}{\\sum_{j=1}^{m} (1 - e_j)}$
""")

add_code("""# ============================================
# 熵权法求权重
# ============================================

def entropy_weight_method(norm_data):
    \"\"\"
    使用熵权法计算各指标的客观权重

    参数:
        norm_data: 归一化后的矩阵, shape (n_companies, n_indicators)

    返回:
        weights: 各指标权重, shape (n_indicators,)
    \"\"\"
    n, m = norm_data.shape

    # Step 1: 计算比重 p_ij
    # 为避免除以0，对全零列做处理
    col_sums = norm_data.sum(axis=0)
    p = norm_data / col_sums

    # Step 2: 处理 p_ij = 0 的情况（微小平移，保证 ln 有意义）
    # 若 p_ij = 0，则 p_ij * ln(p_ij) = 0（取极限），因此直接设 epsilon
    epsilon = 1e-12
    p_safe = np.where(p < epsilon, epsilon, p)

    # Step 3: 计算信息熵 e_j
    k = 1.0 / np.log(n)  # 归一化常数
    e = -k * np.sum(p_safe * np.log(p_safe), axis=0)

    # Step 4: 计算差异系数 d_j = 1 - e_j
    d = 1 - e

    # Step 5: 计算权重 w_j
    weights = d / d.sum()

    return weights, e, d

# 执行熵权法
weights, entropy, diff_coeff = entropy_weight_method(norm_data)

# 整理权重结果为 DataFrame
weight_df = pd.DataFrame({
    '指标': indicators,
    '指标类型': indicator_types,
    '信息熵': entropy.round(4),
    '差异系数': diff_coeff.round(4),
    '权重': weights.round(4)
})
weight_df = weight_df.sort_values('权重', ascending=False)

print("熵权法求得的各指标权重（按权重降序排列）：")
print(weight_df.to_string(index=False))
print(f"\\n权重之和: {weights.sum():.6f}")
""")

# ==================== Cell 6: Weight Visualization ====================
add_md("""### 3.1 权重可视化

通过柱状图直观展示各指标的权重分布。
""")

add_code("""# ============================================
# 权重可视化
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：权重柱状图
ax1 = axes[0]
colors = ['#e74c3c' if w > weights.mean() else '#3498db' for w in weights]
bars = ax1.barh(range(len(indicators)), weights, color=colors, edgecolor='white')
ax1.set_yticks(range(len(indicators)))
ax1.set_yticklabels(indicators, fontsize=9)
ax1.set_xlabel('权重', fontsize=12)
ax1.set_title('各指标权重分布（熵权法）', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.axvline(x=1/len(indicators), color='gray', linestyle='--', alpha=0.7, label=f'平均权重={1/len(indicators):.4f}')
ax1.legend(fontsize=9)

# 标注权重值
for i, (bar, w) in enumerate(zip(bars, weights)):
    ax1.text(w + 0.002, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontsize=8)

# 右图：信息熵饼图（按指标分类）
ax2 = axes[1]
type_weight = {}
for typ, w in zip(indicator_types, weights):
    type_weight[typ] = type_weight.get(typ, 0) + w

type_labels = list(type_weight.keys())
type_vals = list(type_weight.values())
colors_pie = ['#2ecc71', '#e74c3c', '#3498db']
wedges, texts, autotexts = ax2.pie(
    type_vals, labels=type_labels, autopct='%1.1f%%',
    colors=colors_pie[:len(type_labels)],
    textprops={'fontsize': 11}
)
ax2.set_title('各类型指标权重占比', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('weight_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("权重分布图已保存为 weight_distribution.png")
""")

# ==================== Cell 7: GRA ====================
add_md("""## 4. 灰色关联分析 (GRA)

灰色关联分析的基本思想：**通过比较各评价对象与"理想对象"的相似程度来判断优劣。**

### 计算步骤：
1. 用归一化数据乘以权重，得到**加权归一化矩阵**
2. 确定**参考序列**（理想解）：每个指标取最大值
3. 计算**绝对差矩阵**：$\\Delta_{ij} = |r_{0j} - r_{ij}|$
4. 计算**灰色关联系数**：$\\xi_{ij} = \\frac{\\min\\Delta + \\rho \\cdot \\max\\Delta}{\\Delta_{ij} + \\rho \\cdot \\max\\Delta}$
   - $\\rho$ 为分辨系数，通常取 0.5
5. 计算**灰色关联度**（加权求和）：$\\gamma_i = \\sum_{j=1}^{m} w_j \\cdot \\xi_{ij}$
""")

add_code("""# ============================================
# 灰色关联分析
# ============================================

def grey_relational_analysis(norm_data, weights, rho=0.5):
    \"\"\"
    进行灰色关联分析

    参数:
        norm_data: 归一化后的矩阵, shape (n_companies, n_indicators)
        weights: 各指标权重, shape (n_indicators,)
        rho: 分辨系数，默认0.5

    返回:
        rel_grade: 各公司的灰色关联度, shape (n_companies,)
        rel_coeff: 灰色关联系数矩阵, shape (n_companies, n_indicators)
    \"\"\"
    n, m = norm_data.shape

    # Step 1: 加权归一化矩阵
    weighted_data = norm_data * weights

    # Step 2: 确定参考序列（每列取最大值作为理想值）
    ref_sequence = np.max(weighted_data, axis=0)

    # Step 3: 计算绝对差矩阵
    delta = np.abs(ref_sequence - weighted_data)

    # Step 4: 计算灰色关联系数
    delta_min = delta.min()
    delta_max = delta.max()

    rel_coeff = (delta_min + rho * delta_max) / (delta + rho * delta_max)

    # Step 5: 计算灰色关联度（按权重加权）
    # 由于前面已经乘过权重，这里直接用均值即可
    # 也可以重新加权：rel_grade = np.sum(rel_coeff * weights, axis=1)
    rel_grade = np.sum(rel_coeff * weights, axis=1)

    return rel_grade, rel_coeff, weighted_data, ref_sequence, delta

# 执行灰色关联分析
rel_grade, rel_coeff, weighted_data, ref_sequence, delta = grey_relational_analysis(
    norm_data, weights, rho=0.5
)

# 整理结果
result_df = pd.DataFrame({
    '公司': companies,
    '灰色关联度': rel_grade.round(6)
})
result_df = result_df.sort_values('灰色关联度', ascending=False)
result_df['排名'] = range(1, len(companies) + 1)
result_df = result_df.reset_index(drop=True)

print("=" * 50)
print("灰色关联分析最终评价结果")
print("=" * 50)
print(result_df.to_string(index=False))

# 关联系数矩阵
coeff_df = pd.DataFrame(
    rel_coeff.T, index=indicators, columns=companies
).T
coeff_df.index.name = '公司'
print(f"\\n灰色关联系数矩阵：")
print(coeff_df.round(4).to_string())
""")

# ==================== Cell 8: Results Visualization ====================
add_md("""## 5. 结果可视化

对最终评价结果进行可视化展示。
""")

add_code("""# ============================================
# 结果可视化
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 排序后的结果
sorted_companies = result_df['公司'].tolist()
sorted_grades = result_df['灰色关联度'].tolist()

# 左图：灰色关联度柱状图
ax1 = axes[0]
grade_colors = []
top1 = result_df['公司'].iloc[0]
for c in sorted_companies:
    if c == top1:
        grade_colors.append('#e74c3c')  # 最优为红色
    else:
        grade_colors.append('#3498db')

bars = ax1.bar(sorted_companies, sorted_grades, color=grade_colors, edgecolor='white', width=0.6)
ax1.set_xlabel('公司', fontsize=12)
ax1.set_ylabel('灰色关联度', fontsize=12)
ax1.set_title('各公司灰色关联度排名', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(sorted_grades) * 1.15)

# 标注数值
for bar, grade in zip(bars, sorted_grades):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{grade:.4f}', ha='center', fontsize=10, fontweight='bold')

# 标注最优
ax1.annotate(f'最优: 公司{top1}', xy=(0, sorted_grades[0]),
             xytext=(0.5, sorted_grades[0] + 0.02),
             fontsize=11, fontweight='bold', color='#e74c3c',
             ha='center',
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# 右图：雷达图 (Radar Chart)
ax2 = axes[1]
ax2.set_aspect('equal')

# 计算各公司各指标的加权得分用于雷达图
n_indicators = len(indicators)
angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 选择前3名和后2名画雷达图
top3 = result_df['公司'].head(3).tolist()
bottom2 = result_df['公司'].tail(2).tolist()
plot_companies = top3 + bottom2

for i, company in enumerate(plot_companies):
    idx = companies.index(company)
    values = weighted_data[idx].tolist()
    values += values[:1]

    if i < 3:
        alpha_val = 0.3
        lw_val = 1.5
        label = f'{company} (第{i+1}名)'
    else:
        alpha_val = 0.15
        lw_val = 1.0
        label = f'{company} (第{5 if i==3 else 6}名)'

    ax2.fill(angles, values, alpha=alpha_val)
    ax2.plot(angles, values, linewidth=lw_val, label=label)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(indicators, fontsize=7)
ax2.set_title('各公司加权指标雷达图（前3+后2）', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("评价结果图已保存为 evaluation_results.png")
""")

# ==================== Cell 9: Detailed Comparison ====================
add_md("""## 6. 各指标关联系数热力图

通过热力图直观展示各公司在每个指标上的灰色关联系数，帮助识别各公司的优势和劣势。
""")

add_code("""# ============================================
# 关联系数热力图
# ============================================

fig, ax = plt.subplots(figsize=(14, 6))

im = ax.imshow(rel_coeff.T, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)

# 坐标轴设置
ax.set_xticks(range(len(companies)))
ax.set_yticks(range(len(indicators)))
ax.set_xticklabels(companies, fontsize=11)
ax.set_yticklabels(indicators, fontsize=9)

# 在每个单元格标注数值
for i in range(len(indicators)):
    for j in range(len(companies)):
        val = rel_coeff[j, i]
        text_color = 'white' if val < 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)

ax.set_xlabel('公司', fontsize=12)
ax.set_ylabel('指标', fontsize=12)
ax.set_title('各公司-指标灰色关联系数热力图', fontsize=14, fontweight='bold')

# 颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('关联系数', fontsize=11)

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("热力图已保存为 correlation_heatmap.png")
""")

# ==================== Cell 10: Summary ====================
add_md("""## 7. 结论与分析

### 评价方法总结
本分析采用 **熵权法 + 灰色关联分析** 的组合评价模型：
- **熵权法**客观地从数据本身的信息量出发确定权重，避免了主观赋权的随意性
- **灰色关联分析**通过衡量各公司与理想解的"距离"，给出综合评分

### 指标合理性说明
- 对于具有**固定最佳值**的中间型指标（如产品品质、产品包装），若存在公认的标准值，可替换均值作为最佳值以提高合理性
- 若有专家经验，可在熵权法基础上结合**层次分析法（AHP）**进行组合赋权
""")

add_code("""# ============================================
# 输出最终评价结果表
# ============================================

# 最终汇总表
print("╔" + "═" * 60 + "╗")
print("║" + "  基于熵权法-灰色关联分析的公司综合评价结果".center(56) + "║")
print("╠" + "═" * 60 + "╣")
print(f"║  {'排名':<4} {'公司':<6} {'灰色关联度':<12} {'评价等级':<10}       ║")
print("╠" + "═" * 60 + "╣")

max_grade = result_df['灰色关联度'].max()
min_grade = result_df['灰色关联度'].min()

for _, row in result_df.iterrows():
    rank = int(row['排名'])
    company = row['公司']
    grade = row['灰色关联度']

    # 评价等级
    if grade >= max_grade * 0.9:
        level = '优秀'
    elif grade >= max_grade * 0.8:
        level = '良好'
    elif grade >= max_grade * 0.7:
        level = '中等'
    else:
        level = '一般'

    print(f"║  {rank:<4} {company:<6} {grade:<12.6f} {level:<10}       ║")

print("╚" + "═" * 60 + "╝")

print(f"\\n最佳公司: 公司{result_df['公司'].iloc[0]} (关联度={result_df['灰色关联度'].iloc[0]:.6f})")
print(f"最差公司: 公司{result_df['公司'].iloc[-1]} (关联度={result_df['灰色关联度'].iloc[-1]:.6f})")
print(f"极差: {max_grade - min_grade:.6f}")
""")

# ==================== Assemble ====================
nb.cells = cells

# 写入 .ipynb 文件
output_path = r'D:\shumo\grey ralational anlysis\grey_relational_analysis.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook 创建成功！共 {len(cells)} 个单元格。")
print(f"文件路径: {output_path}")
