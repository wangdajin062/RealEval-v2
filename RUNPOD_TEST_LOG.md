# QAD-MultiGuard H100 复现:RunPod 容器测试记录

**容器**: `b0c35c15caf3` / `b34650b090d3`（跨会话有过重建）
**硬件**: 1× NVIDIA H100 80GB HBM3
**软件**: torch 2.13.0+cu130, transformers 5.13.1, peft 0.19.1, datasets 5.0.0（venv: `/workspace/venv/bin/python`）
**代码库**: `/workspace/H100_package_realeval`（另有 `repo/`、根目录三份历史拷贝，已确认权威路径需固定）
**目标**: 验证 v25 论文（`v25_blind.tex`）中 QAD + OV-Freeze 声称的实验结果是否可在真实 H100 上复现

---

## 阶段总览

| 阶段 | 发现的问题 | 状态 |
|---|---|---|
| 1 | `run_all.sh` 环境未配置（realeval 不可 import，包缺失，数据路径硬编码） | ✅ 已修 |
| 2 | 多份 `realeval` 拷贝，PYTHONPATH 指向不确定 | ✅ 固定为软链 |
| 3 | `group_split` 不存在，`load_dataset` 不存在 | ✅ 已实现 |
| 4 | 训练数据是恒定的音频任务提示词模板（TAF28k/taf28k.jsonl），文本零区分度 | ✅ 已定位并改用正确语料 |
| 5 | LoRA adapter 从未被任何实验加载（checkpoint 断链） | ✅ student_loader 接线完成 |
| 6 | 训练数据静默回退到合成占位符（`synthetic_normal_0`） | ✅ 已阻断静默回退 |
| 7 | Label masking 全部置 `-100`（loss=0 根因） | ✅ 精确边界重写 |
| 8 | HF Trainer + 梯度累积路径必然 NaN（bf16 LoRA + device_map=auto） | ✅ 绕开，改手工训练循环 |
| 9 | 任务饱和（balanced4k 63 步收敛到 F1=1.0），无法区分方法差异 | ✅ 改用 AdvFraud 困难集 |
| 10 | ChiFraud 标签定义与 balanced4k 不兼容（不同任务） | ⚠️ 排除出评测 |
| 11 | OV-Freeze 首次实现方向错误（冻结权重而非方差正则） | ✅ 按论文 Eq.3/4/6 重写 |
| 12 | PEFT 包装后层名匹配三次失败（`isinstance(nn.Linear)` 排除了 LoRA 包装器） | ✅ 正则锚定 + 显式排除 base_layer/lora_* |
| 13 | 磁盘配额超限（逐臂落盘 adapter） | ✅ 改用临时目录 + 不持久化 |
| 14 | 正则激活后数值发散（reg 达 69万，F1 从 1.0 崩到 0.66） | 🔴 **进行中**：需按论文附录 Lemma A.1 加界 |

---

## 阶段 1-3：环境与包问题

### 1.1 运行 `run_all.sh` 的初始失败

```
ModuleNotFoundError: No module named 'realeval'
ModuleNotFoundError: No module named 'transformers'
FileNotFoundError: data/TAF28k/taf28k.npz
```

**根因**：
- `realeval` 包未在 `PYTHONPATH` 上
- 系统 `python3` 与项目 venv（`/workspace/venv/bin/python`）不是同一个解释器；包只装在 venv 里
- `_env.py` 硬编码了 `/workspace/datasets/`，实际数据在 `/workspace/data/`

**修复**：
- `run_all.sh` 增加 STEP 0：自动探测 `realeval/` 所在目录并 `export PYTHONPATH`
- 自动检测并 `pip install --break-system-packages` 缺失包
- `_env.py` 的 `load_taf28k()` 改为尝试 5 个候选路径，找不到给出明确报错

### 1.2 容器重启导致软链丢失

```
ModuleNotFoundError: No module named 'realeval'
ls: cannot access '/workspace/realeval/': No such file or directory
```

**根因**：`/workspace/realeval` 是符号链接，容器重建后未持久化；根目录另有一个孤儿 `data.py`（`/workspace/data.py`）会遮蔽包内的 `realeval.data`。

**修复**：
```bash
ln -sfn /workspace/H100_package_realeval/realeval /workspace/realeval
mv /workspace/data.py /workspace/data.py.orphan
```
并写入 `setup_workspace.sh`，容器重启后一条命令恢复环境、并自检五项补丁是否都在（`group_split`/`load_dataset`/`load_text_corpus`/`student_loader.py`/`real_backend` 接线）。

---

## 阶段 4：数据语料路由错误

### 4.1 发现：exp5/9/10/12/14 全部读取音频任务的提示词模板

```python
ds = data.load_taf28k(max_samples=...)   # 错误：这是给音频模型用的固定指令
```

检查 `TAF28k/taf28k.jsonl` 发现：

```
text: 1 unique   （4000 条样本，text 字段完全相同）
label: 2 unique
```

内容是：
> "根据听到的音频内容，分析该通话是否涉及诈骗。请按照以下格式输出:..."

这是**音频模型的固定 prompt 模板**，不是逐样本变化的文本语料。任何走纯文本路径的实验读到它都只能得到类别先验（accuracy≈0.5）。

**连锁效应**：
- `exp5` cross-dataset：accuracy 0.5088（TAF28k）/ 0.4867（AdvFraud）—— 均≈随机
- `exp3` OV-Freeze 消融：13 个消融臂 F1 全部等于 0.6744 —— 因为评的都是同一个未受影响的模型输出
- `exp1` 的高 F1（0.9988）与上述形成矛盾，因为它走的是另一条数据路径（`load_chifraud_balanced()`，指向 `balanced4k.jsonl`，真实短信文本）

### 4.2 真实文本语料盘点

```
data/balanced4k/balanced4k.jsonl    真实中文短信，label 平衡（诈骗 vs 正常）
data/spam11358/spam11358.jsonl      真实垃圾短信
data/AdvFraud3k/advfraud3k.jsonl    真实对抗诈骗文本（全部 label=1，2119 条，1292 distinct）
data/ChiFraud/chifraud.jsonl        灰产广告 vs 新闻/公告（不同标注体系，见 4.4）
```

### 4.3 修复：`load_dataset()` / `load_text_corpus()`

新增函数路由到正确文件，并对退化语料（distinct < 2）主动拒绝：

```python
def load_text_corpus(name="balanced4k", max_samples=None):
    ...
    if n_uniq < 2:
        raise ValueError(f"{path} has {n_uniq} distinct text(s); carries no input signal")
```

`exp5/exp9/exp10/exp12/exp14` 改为调用 `data.load_dataset(config.get("data",{}).get("dataset","balanced4k"), ...)`。`exp13` 保留 `load_taf28k(..., source="multimodal")` 不变（它需要音频 NPZ 嵌入，用法本身正确）。

批量替换时发生过一次正则截断 bug（`config.get("data")` 丢了 `{}` 默认值导致 `AttributeError`），已修复。

### 4.4 ChiFraud 标签体系不兼容

```
label 分布: {0: 346, 1: 54}
label=0: 金融新闻、网站免责声明、电网调度公告   （长篇正规文本）
label=1: 信用卡套现广告、色情招嫖广告           （灰产广告）
```

用 balanced4k 训出的模型在 ChiFraud 上准确率 0.200（远低于随机），排查后确认这是**任务定义不同**（"诈骗短信 vs 日常短信" ≠ "违规广告 vs 正规文章"），不是模型缺陷或跨域泛化失败。**已从跨数据集鲁棒性评测中排除**，论文中若引用需注明标注体系差异。

---

## 阶段 5-6：Checkpoint 断链与训练数据造假

### 5.1 Adapter 从未被加载

`grep` 全部实验文件：`save_sites=[]`，`load_sites=[]`。但 `/workspace/outputs/sft_checkpoints/checkpoint-250/` 确实存在（35MB LoRA adapter，由 `cluster/train_sft.py` 产出）。

**结论**：所有下游实验（exp3/exp5/exp6 等）此前评估的都是**未微调的 base Qwen2.5-0.5B**。

### 5.2 修复：`student_loader.py`

新增 `resolve_adapter()` / `attach_adapter()` / `load_student()`，接入 `real_backend.real_llm_classify`。**关键设计**：非 `base` 变体找不到对应 adapter 时**抛错**，不静默回退——这正是此前问题的根源（静默回退制造了"看起来在跑实验，实际全评同一个模型"的假象）。

```python
if variant not in ("base", None, "") :
    raise RuntimeError(f"student_variant='{variant}' requested but no adapter found ...")
```

`config/experiments.yaml` 新增：
```yaml
student_variant: qad_ovf
students:
  qad_ovf: /workspace/outputs/sft_checkpoints/checkpoint-250
```

### 5.3 训练数据静默退化为合成占位符

```
HF bucket load failed (...), falling back to synthetic: No module named 'datasets'
text[:200]: 'synthetic_normal_0'
```

`prepare_sft_dataset()` 在任何异常（含 `ImportError`）时无条件回退到 `load_synthetic()`。**checkpoint-250 实际训练数据是占位字符串**，不是真实语料。

**修复**：
- `pip install datasets`
- `train_sft.py` 改为直接读本地语料：`load_text_corpus(args.corpus, ...)` + `group_split()`
- `prepare_sft_dataset` 的静默回退改为默认抛错，需显式 `REALEVAL_ALLOW_SYNTHETIC=1` 才允许

Dry-run 验证：
```
corpus=balanced4k  source=/workspace/data/balanced4k/balanced4k.jsonl  n=4000  distinct=3842
Train: 3200 total (1600 fraud / 1600 normal)
Test:  800 total (400 fraud / 400 normal)
```

---

## 阶段 7：Label Masking 全 -100（loss=0 直接根因）

### 7.1 现象

`trainer_state.json`（checkpoint-250）：
```
{'epoch': 0.08, 'grad_norm': 0.0, 'loss': 0.0, 'step': 10}
...
{'eval_loss': nan}
```

`loss` 从第 10 步起恒为 0，`grad_norm` 恒为 0，`eval_loss` 为 `nan`。

### 7.2 根因

原 `tokenize_fn` 分别对 `"...Answer: "`（prompt-only）和 `"...Answer: fraud"`（full）两次独立 tokenize，用长度差推断监督边界：

```python
prompt_len = len([t for t in prompt_ids if t != pad_token_id])
for j in range(min(prompt_len, len(label_row))):
    label_row[j] = -100
```

实测：两者 tokenize 后**长度相同**（33 tokens）——`"Answer: "` 结尾的空格自成一个 token，与 `" fraud"`（带前导空格）占据同一位置。于是 `prompt_len == len(full_ids)`，掩码覆盖了整个序列，包括答案本身。

```
full len: 33  prompt len(non-pad): 33
supervised tokens left: 0   <-- bug
```

### 7.3 修复

改为分别 tokenize prompt 和 answer（`add_special_tokens=False`），显式拼接，边界由构造保证：

```python
p_ids = tokenizer(PROMPT.format(text=t), add_special_tokens=False)["input_ids"]
a_ids = tokenizer(" fraud"/" normal", add_special_tokens=False)["input_ids"] + [eos]
labels = [-100]*len(p_ids) + a_ids
```

自检验证（3 条真实中文样本）：`supervised=2~3` tokens/样本，不再为 0。

---

## 阶段 8：HF Trainer 梯度累积路径 NaN

### 8.1 现象

即便 masking 修复后，仍在训练中期出现：
```
{'loss': '0', 'grad_norm': 'nan', ...}
```

### 8.2 系统性排查（对照实验）

| 配置 | 结果 |
|---|---|
| `bs=8, ga=4`（原配置） | NaN at step ~10 |
| `bs=32, ga=1`（同等效批量，单步） | **20 步全部干净**，loss 10.3→3.5 |
| 手工训练循环（无 Trainer/accelerate） | **14 步干净**，loss 10.35→1.64，post-clip 精确=1.0 |

**结论**：NaN 与数据、模型、超参无关，特定发生在 `gradient_accumulation_steps>1` + `device_map="auto"` + bf16 LoRA 的 accelerate 累积路径中。

### 8.3 修复：弃用 HF Trainer，改用显式训练循环

`train_lora_manual.py`：手写 forward/backward/clip/step，NaN 或非有限梯度立即 `sys.exit`（不再产出"看似训练成功实则是 NaN 污染的空 adapter"）。

**副作用发现**：任务在此配置下极易饱和——`balanced4k` 63 步即收敛至 loss≈0，此后 Adam 二阶矩在近平坦损失面上塌缩，`grad/sqrt(v_hat)` 在 bf16 下溢出。加入"连续 20 步 loss<0.02 则提前停止"的收敛保护后消除。

---

## 阶段 9：评测集饱和 → 改用困难样本

### 9.1 balanced4k 无法区分方法

```
[epoch 0] eval_loss 0.0009  F1 1.0  acc 1.0  P 1.0  R 1.0   (63 步收敛)
```

800 条留出测试样本零错误。经检查非数据泄漏（`group_split` 已按归一化文本分组，无重叠），而是**任务本身对 0.5B 模型太简单**——训练样本含大量表面标记（微信号、长数字串、URL）。

### 9.2 泛化测试确认能力边界

用 balanced4k 训出的 LoRA 测试：

| 语料 | 结果 |
|---|---|
| base 模型（零样本）在 AdvFraud | recall = 0.000 |
| LoRA（balanced4k 训练）在 AdvFraud | **recall = 0.914** |
| LoRA 在 spam11358（同源） | accuracy = 0.990 |
| LoRA 在 ChiFraud | accuracy = 0.200（见 4.4，任务不兼容，非泛化失败）|

AdvFraud-3k（2119 条，全部真实对抗诈骗文本，**无任何表面标记**）是唯一具有区分度、且模型未完全饱和（91.4% vs 满分仍有 8.6 点空间）的困难集。

### 9.3 困难混合集构建

```
2119 条 AdvFraud 正例 + 2119 条 balanced4k 负例 = 4238 条
group_split(test_ratio=0.2, seed=42)
```

---

## 阶段 10：OV-Freeze 首次实现方向错误

### 10.1 误判

第一版实现将 "OV-Freeze" 理解为"冻结 value/output 投影权重"（不给 v_proj/o_proj 挂 LoRA），四臂各训一个 adapter：

```
arm             frozen  bf16 F1  int4 F1    drop    drift%
no_reg               0   1.0000   0.9079  0.0921   23.17
ovf_full             24   1.0000   0.8800  0.1200   19.57
```

覆盖越多、int4 掉点越大——与预期方向相反。

### 10.2 核对论文原文后确认误判

读取 `v25_blind.tex` 第 304-350 行，OV-Freeze 的真实定义：

**Eq. (eq:ovf-loss)**：
```
L_OVF = λ · Σ_{ℓ∈P} ||Var_EMA(y_ℓ) − σ²_BF16,ℓ||²₂,  P={q,k,v,o}_proj,  λ=0.01
```

**Eq. (eq:ema)**：`Var_EMA^(t) = ρ·Var_EMA^(t-1) + (1-ρ)·Var_batch^(t)`，`ρ=0.95`

**Eq. (eq:ovf-rescale-en)**：`y'_ℓ = y_ℓ · c_ℓ`，`c_ℓ = sg[√(σ²_BF16,ℓ / (Var(y_ℓ)+ε))]`（stop-gradient）

**调度**：仅在训练最后 30% 步激活

这是一个**方差正则 + 前向重缩放机制**，不是权重冻结。第一版实现完全推翻，重写。

### 10.3 重写实现（`train_ovfreeze_paper.py`）

- 离线标定：teacher 在校准集上前向，记录 `{q,k,v,o}_proj` 静态 BF16 方差 `σ²_BF16`
- Forward hook：EMA 追踪 + stop-gradient 重缩放 + 正则项累加
- 三组消融：覆盖顺序（q→v→k→o→+FFN）、激活窗口（0/10/20/30/50/70%）、方差估计策略（batch/EMA/global）

---

## 阶段 11：层名匹配 —— 三次失败后的根因定位

PEFT 包装后，投影层路径从 `model.layers.0.self_attn.q_proj`（teacher，`nn.Linear`）变为 `base_model.model.model.layers.0.self_attn.q_proj`（student，PEFT `lora.Linear`）。

### 11.1 尝试一：`name.find("model.layers.")`

失败：`find()` 匹配到**第一个** `model.`，对 `base_model.model.model.layers...` 会截出 `model.model.layers...`（多一层 `model.`），与 teacher 的 key 对不上。`hooks attached: 0`。

### 11.2 尝试二：`name.rfind(...)` + 去掉 `isinstance(nn.Linear)`

`rfind` 修好了字符串定位，但仍然 `hooks attached: 0`。逐条件排查发现：**PEFT 的 `lora.Linear` 不继承 `nn.Linear`**（是 `nn.Module + LoraLayer` mixin），`isinstance` 检查把所有包装器排除掉了；而其内部真正是 `nn.Linear` 的 `.base_layer`（无 LoRA 增量、名字带 `.base_layer` 后缀）反而被误纳入。

### 11.3 尝试三：去掉 isinstance，仅按名字过滤

仍然 0。原因：patch 只改了部分代码路径，另一处仍保留旧的 `isinstance` 门禁（补丁式修改遗漏）。

### 11.4 最终修复：完整重写，正则锚定 + 离线验证

```python
_LAYER_RE = re.compile(r"(layers\.\d+\..*)$")
def canon(name):
    m = _LAYER_RE.search(name)
    return f"model.{m.group(1)}" if m else name

def is_ovf_target(name, suffixes):
    if name.endswith(".base_layer") or ".lora_" in name:
        return False
    return any(canon(name).endswith(s) for s in suffixes)
```

正则锚定 `layers\.\d+\..*$`，不依赖前缀有几层 `model.`；显式排除 `.base_layer` 和 `.lora_*` 子模块，只保留 LoRA 包装器本身（其输出已包含 LoRA delta，正是论文要约束的 `y_ℓ`）。

**离线验证**（不依赖 GPU/模型下载，纯字符串逻辑）：用实际观测到的 12 条真实 PEFT 命名样本跑通 `canon()`/`is_ovf_target()`，确认 4 处命中（v/o/q proj 顶层 + 另一层 q_proj）全部正确，`base_layer`/`lora_A`/`lora_B`/`lora_embedding_A` 全部正确排除，teacher/student 的 `canon()` 结果完全一致。

**容器内 `--self-test` 验证通过**：
```
[self-test] hooks attached: 96  (want 96)
[self-test] reg terms collected: 96
[self-test] reg loss: 4.943808078765869
[self-test] PASS
```

---

## 阶段 12：磁盘配额

```
OSError: [Errno 122] Disk quota exceeded
```

`outputs/` 下累积了多轮实验的 adapter（`ovfreeze/`、`ovfreeze_s42/`、`sft_checkpoints/` 等，合计约 1.5GB，超出项目配额而非文件系统总容量——`df` 显示 215T 可用具有误导性）。

**修复**：
- 清理历史无用 adapter 目录
- int4 评测改为写入 `tempfile.mkdtemp()` 临时目录，评测完立即 `shutil.rmtree`，不再逐臂持久化保存

---

## 阶段 13（当前）：正则激活后数值发散

### 13.1 现象

覆盖消融（`--ablation coverage`）中，OV-Freeze 一旦在第 144 步激活：

```
step  144/206  loss  0.0001  gn   0.01  <-- OV-Freeze ON
step  150/206  loss  1.4509  reg 40.5067  gn 182.99
step  175/206  loss 11.6453  reg 36.4722  gn 242.29
```

`bf16 F1` 从 1.0 崩至 0.66～0.72，`ppl_fluct` 达 10 万～68 万量级（`qvko_ffn` 臂 `reg` 甚至到 69 万）。

### 13.2 根因

重缩放系数无上下界：
```python
c = sqrt(σ²_BF16 / (Var(y_batch) + ε))
```
模型收敛后（step ~65 起 loss≈0），某些投影层的批次方差可能瞬时塌缩至接近 0，导致 `c` 发散，将该层输出直接放大到失控量级。

### 13.3 论文附录核实

`v25_blind.tex` 第 899-950 行，**Lemma A.1（correction factor 有界性）** 与 **Proposition A.2（SGD 收敛界）**：

- Lemma A.1 证明 `c_ℓ ∈ [c_min, c_max]` **存在**（数学存在性证明：有限 mini-batch 上 `Var(y_ℓ)` 有限、`ε>0`，故 `c_ℓ` 有限；训练轨迹有限步，上下确界必被达到且有限）
- **但证明本身不提供具体数值** `c_min`/`c_max`——只证明"某个界存在"，不构造性给出界是多少
- Proposition A.2 依赖 `c_max < ∞` 这个前提反推收敛速率，同样未给出工程实现应设的具体裁剪范围

### 13.4 当前状态：待处理

需要在代码里加一个**论文未明确数值、但工程上必要**的裁剪（如 `c.clamp(min=0.1, max=10.0)`），并在代码注释与论文复现说明中明确标注：此界限为补充工程假设，非论文原文数值。尚未应用并重新验证。

---

## 关键交付物清单

| 文件 | 用途 |
|---|---|
| `apply_all_fixes.py` | 一键应用 group_split / student_loader / real_backend 接线 / config 补全，含幂等性与 5 项后置校验 |
| `fix_train_sft.py` | 修复 label masking 全 -100 的问题 |
| `train_lora_manual.py` | 绕开 Trainer NaN 路径的手工训练循环，含 NaN abort 与收敛保护 |
| `train_ovfreeze.py` | （已废弃）第一版 OV-Freeze 误实现（权重冻结），保留作对照 |
| `train_ovfreeze_paper.py` | 按论文 Eq.3/4/6 重写的正确实现，含 `--self-test` |
| `diagnose_v25_run.py` | 三项根因（specdec α=0 / 数据泄漏 / checkpoint 断链）自动化诊断脚本 |
| `setup_workspace.sh` | 容器重启后一键恢复环境 + 五项补丁自检 |

## 尚未解决 / 待办

1. **correction factor 裁剪范围** —— 论文只证明存在性，需要工程补充数值，待应用并重跑覆盖消融
2. **AdvFraud 困难集上的完整三组消融**（coverage/window/estimator）尚未在裁剪修复后重新产出可信数据
3. **ChiFraud 标签体系不兼容**问题需要在论文中如实说明，或替换为语义可比的 OOD 基准
4. **exp6 投机解码 α=0** 的独立代码路径 bug 尚未在本容器内修复验证（此前仅在手工探针中确认逻辑本身正确，α_probe=0.733）
5. 三份 `realeval` 历史拷贝仍未清理归一，建议后续统一为单一权威路径
