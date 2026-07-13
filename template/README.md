# H100 RealEval Template

QAD-MultiGuard H100 paper-validation pipeline — 一键部署的多服务 Docker 模板。

## 快速开始

```bash
# 1. 配置环境变量
cp .env.template .env

# 2. 构建并启动
docker compose up -d

# 3. 一键运行实验
docker exec -it realeval-h100 bash run_all.sh
```

## 服务端口

| 服务         | 端口   | 访问地址                    |
|-------------|--------|----------------------------|
| SSH         | 22     | `ssh root@localhost`       |
| Jupyter Lab | 8888   | `http://localhost:8888`    |
| VSCode      | 3000   | `http://localhost:3000`    |
| Ollama      | 11434  | `http://localhost:11434`   |
| API         | 8000   | `http://localhost:8000`    |
| API Docs    | 8000   | `http://localhost:8000/docs` |

## 目录结构

```
/workspace/
├── models/          # 模型权重 (Qwen2.5, Whisper)
├── hf_cache/        # HuggingFace 缓存
├── outputs/         # 实验结果
│   ├── results/     # 结果文件
│   ├── metrics/     # 评估指标
│   ├── tables/      # LaTeX 表格
│   └── figures/     # 图表
├── data/            # 数据集
├── logs/            # 服务日志
├── repo/            # RealEval 代码
└── .ollama/         # Ollama 模型存储
```

## run_all.sh 用法

```bash
bash run_all.sh                  # 完整 paper pipeline
bash run_all.sh --smoke          # 冒烟测试 (无需 GPU)
bash run_all.sh --distributed    # 多 GPU (NCCL + torchrun)
bash run_all.sh --setup          # 仅环境配置
bash run_all.sh --notebook       # 仅启动 Jupyter Lab
bash run_all.sh --skip-models    # 跳过模型下载
```

## 环境变量

| 变量                    | 默认值              | 说明            |
|------------------------|--------------------|----------------|
| SSH_PORT               | 22                 | SSH 端口        |
| JUPYTER_PORT           | 8888               | Jupyter 端口    |
| VSCODE_PORT            | 3000               | VSCode 端口     |
| API_PORT               | 8000               | API 端口        |
| OLLAMA_PORT            | 11434              | Ollama 端口     |
| JUPYTER_TOKEN          | realeval           | Jupyter 令牌    |
| VSCODE_PASSWORD        | realeval           | VSCode 密码     |
| NVIDIA_VISIBLE_DEVICES | all                | GPU 可见性      |
| HOST_MODEL_CACHE       | /dev/null          | 宿主机模型缓存路径 |

## API 端点

```bash
# 健康检查
curl http://localhost:8000/health

# GPU 状态
curl http://localhost:8000/gpu

# 运行实验
curl -X POST http://localhost:8000/experiments/run \
  -H "Content-Type: application/json" \
  -d '{"experiments": "all", "mode": "paper"}'

# 查看结果
curl http://localhost:8000/results
```

## 模型清单

自动下载的模型 (通过 `cluster/manage_models.sh`):

- `Qwen/Qwen2.5-0.5B-Instruct` — 主实验模型
- `Qwen/Qwen2.5-0.5B` — 基线模型
- `openai/whisper-tiny` — 语音识别

设置 `STAGE_LARGE=1` 额外下载:
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

## Ollama 模型

启动后自动拉取 (通过 `services/ollama/setup.sh`):
- `qwen2.5:0.5b` — 轻量蒸馏实验
- `qwen2.5:1.5b` — 中型对比
- `qwen2.5:7b` — 教师模型

## 独立 Ollama 服务

```bash
docker compose --profile ollama-standalone up -d
```

## 实验列表 (14 项)

RealEval v4.2.0 包含 14 个实验，涵盖蒸馏、隐私、推测解码和基准测试。

见项目 [README.md](../README.md) 获取完整实验矩阵。

## 系统要求

- NVIDIA GPU (推荐 H100/A100，CUDA 12.4+)
- Docker 24+ with `nvidia-container-toolkit`
- 磁盘: 50GB+ (模型 + 输出)
- 内存: 32GB+

## 安全须知

- 生产环境请修改默认密码 (`JUPYTER_TOKEN`, `VSCODE_PASSWORD`)
- SSH 默认允许 root 登录，请修改密码或使用密钥认证
- API 无认证，仅适合内网使用
