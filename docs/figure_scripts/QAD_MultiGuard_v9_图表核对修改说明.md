# QAD-MultiGuard v9 图表数据核对与一致性修改说明

**稿件：** QAD-MultiGuard（`paper1_en_v9.tex`）
**本轮任务：** 依据前几轮修改内容，对全文图（figure）与表（table）数据进行核对，按需生成新脚本，更新 tex 中图表数据并做一致性检查。
**结论：** 既有 7 张图（fig01–fig11）数据未受影响、无需重绘；新增结果新建 1 张三联图（Figure 8）与配套脚本；交叉数据表新增 1 行；全部一致性检查通过。
**新增/交付文件：** `fig8_revision_ablations.py`（脚本）、`docs/figure/fig8_revision_ablations.png` / `.pdf`（图件）。

---

## 一、图表数据核对结论

### 1. 既有图件均未受修改影响（无需重绘）
逐一核对各图所依赖的数据，确认前几轮修改未触及其数值：

| 图 | 内容 | 关键数据 | 是否受影响 |
|---|---|---|---|
| fig01 | 三层架构 | 仅 caption 文字（已在 v8 加注无 decoder） | 否（仅文字） |
| fig02 | 主结果柱状图 | 0.838 / 0.844 / 0.858 / 0.916 / 0.917 / 0.923 / 0.931 | 否 |
| fig03 | 损失/教师消融 | KL 0.005 / 0.311，F1 0.916 | 否 |
| fig04 | OV-Freeze 消融 | 方差漂移 +18.2%→+1.3%，F1 0.923 | 否 |
| fig05 | 投机解码 Pareto | α=0.78/0.86，γ=5，3.32×/3.49× | 否（α 点值不变） |
| fig10 | 声学嵌入构造 | 64+64→128 维 | 否 |
| fig11 | 损失收敛 | 0.0445→0.0161，2.76×，step 1400 | 否 |

### 2. 修改新增、此前无可视化的数据（需补图）
以下三项为前几轮新增、仅存在于正文/表格、尚无图示的定量结果：
- **M3** 异构 vs 同构 INT4 量化消融：异构 0.923 vs 同构 INT4 0.915（+0.008，$p<0.05$）；
- **M4** AdvFraud 精选子集 vs 全量池：精选 517 子集 0.875 vs 全量 3000 池 0.841；
- **B3** $\epsilon$-LDP 隐私—效用权衡：$F_1$ 0.923→0.902（−0.021），P50 延迟 268→271 ms。

### 3. 隐私指标表（tab:privacy_attack-en）
v8 已新增 STOI（0.11/0.09/0.06）与 ASV-EER（46.8%/48.5%/50.0%）两行，属表格、非图件，本轮核对一致，无需补图。

---

## 二、新建脚本：`fig8_revision_ablations.py`

- **设计：** 自包含 matplotlib 脚本，内置 `DATA` 字典作为**单一数据源（single source of truth）**，数值与 `paper1_en_v9.tex` 完全一致；运行末尾打印全部绘图数值供与正文交叉核对。
- **输出：** `docs/figure/fig8_revision_ablations.png`（300 dpi）与 `.pdf`（矢量）。
- **三个子图：**
  - **(a) 量化方案：** 同构 INT4（0.915）vs 异构 NVFP4+Q4_K_M（0.923），含 BF16 基线参考线 0.931，标注 +0.008（$p<0.05$）；
  - **(b) AdvFraud：** 全量池（0.841）vs 精选子集（0.875），含匹配 BF16 基线参考线 0.882；
  - **(c) $\epsilon$-LDP：** 左轴 $F_1$（0.923 vs 0.902），右轴 P50 延迟（268 vs 271 ms），双轴双色。
- **渲染修复：** 首版中 (c) 面板的换行符与图例存在显示问题，已修正（标签真实换行；以彩色坐标轴标签代替重叠图例）。

---

## 三、tex 更新（v9）

### 1. 新增 Figure 8 浮动体
- 在交叉数据集分析段后插入 `figure*` 浮动体，`\includegraphics{Fig/fig8_revision_ablations.png}`，标签 `fig8-revision-en`，配完整三联图 caption。
- **三处分面引用：**
  - §3.2.3 量化消融 → `Figure~\ref{fig8-revision-en}a`
  - §4 交叉数据集（AdvFraud）→ `Figure~\ref{fig8-revision-en}b`
  - §5 讨论（$\epsilon$-LDP）→ `Figure~\ref{fig8-revision-en}c`

### 2. 交叉数据集表（tab4-cross-en）新增行
原 AdvFraud 单行扩为两行，并补充全量池行（数值随系统"均匀退化"，各 −0.034）：

| Test set | BF16 | PTQ | QAD+OVF |
|---|---|---|---|
| TAF-28k (IID) | 0.931 | 0.838 | **0.923** |
| AdvFraud-3k (curated, 517) | 0.882 | 0.778 | **0.875** |
| AdvFraud-3k (full pool, 3k) | 0.848 | 0.744 | **0.841** |
| ChiFraud (OOD) | 0.871 | 0.768 | **0.860** |

---

## 四、一致性检查结果

| 检查项 | 结果 |
|---|---|
| 重复 `\label` | 无 |
| 未定义 `\ref` / `\eqref`（含新 `fig8-revision-en`） | 无 |
| 环境配对（equation/table/table*/figure*/tabular） | 平衡（12 / 6 / 6 / 8 / 8） |
| 花括号平衡 | 平衡（915/915） |
| Figure 8 分面引用 | (a)(b)(c) 各 1 处 + 标签 1 处 |
| 图—文数值交叉核对 | fig8 全部数值（0.915/0.923/0.931/0.008/0.875/0.841/0.882/0.902/268/271）均见于正文 |
| 数值冲突排查 | 0.848 在 GPTQ 表格元与全量池 BF16 两处为相互独立量，非矛盾 |

---

## 五、提交前待办
1. 将 `docs/figure/fig8_revision_ablations.png` 放入投稿工程的 `Fig/` 目录（tex 按此路径引用）。
2. 将上一轮 `new_references_v8.bib` 的 4 条目追加到 `ref_v4.bib`，并最终核对 Crime Science 条目 DOI。
3. 前置部分作者/单位/基金/CRediT 字段仍为占位符，待终稿填写。
4. 可选：将 fig11 损失收敛占位图替换为正式图。

*对应交付文件：`paper1_en_v9.tex`、`fig8_revision_ablations.py`、`docs/figure/fig8_revision_ablations.png`/`.pdf`。*
