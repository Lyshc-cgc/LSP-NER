# LSP: 基于标签子集划分增强示例的少样本命名实体识别

[English Version (英文版本)](./README.md)

本仓库是论文 **"LSP: Empowering Few-Shot NER with Demonstration Augmentation via Label Subset Partition"** 的官方实现。

## 1. 环境配置

使用 mamba 创建 conda 环境：

```bash
# 从 yml 文件创建环境
mamba env create -f lsp-env.yml

# 激活环境
mamba activate lsp

# 导出你的 API 密钥
export DASHSCOPE_API_KEY="你的-dashscope-密钥"
export OPENAI_API_KEY="你的-openai-密钥"
```

**说明：** 如果仅通过 API 模式运行（推荐），不需要安装 vLLM 等本地推理相关依赖。环境文件已经包含了 API 模式运行所需的全部依赖。

## 2. 配置说明

所有配置文件都存放在 `cfgs/` 目录下，你需要自定义以下设置：

### API 配置 (`cfgs/api.yml`)

**重要说明：** `api.yml` 中的 `api_key` 字段填写的是**环境变量名称**，而不是实际的 API 密钥。你需要通过环境变量设置真实的 API 密钥。

配置示例：

```yaml
qwen:
  base_url: 你的-qwen-api-base-url
  model: qwen1.5-32b-chat
  api_key: DASHSCOPE_API_KEY      # 这里填写环境变量名称
  concurrency_level: 10

gpt:
  base_url: 你的-openai-api-base-url
  model: gpt-4o-mini
  api_key: OPENAI_API_KEY        # 这里填写环境变量名称
  concurrency_level: 10

deepseek:
  base_url: 你的-deepseek-api-base-url
  model: deepseek-chat
  api_key: DEEPSEEK_API_KEY      # 这里填写环境变量名称
  concurrency_level: 10
```

然后通过环境变量设置你真实的 API 密钥：

```bash
# Linux/Mac 示例
export DASHSCOPE_API_KEY="你真实的-dashscope-api-密钥"
export OPENAI_API_KEY="你真实的-openai-api-密钥"
export DEEPSEEK_API_KEY="你真实的-deepseek-api-密钥"
```

### 其他配置
- **数据集配置**: `cfgs/data_cfgs/<数据集名称>.yml` - 定义数据路径和数据集特定设置
- **标签配置**: `cfgs/label_cfgs/<数据集名称>.yml` - 定义实体类型标签体系
- **方法配置**: 查看 `single_type/`, `multi_type/`, `subset_cand/`, `retrieval_lsp/` 等目录获取不同提示策略的配置

主配置入口是 `config.yml`，它注册了所有数据集和标注配置。

## 3. 数据集准备

下载数据集并按照以下目录结构组织：

```
LSP-NER/
└── data/
    ├── CMeEE_V2/
    │   ├── CMeEE_V2.py
    │   └── raw/
    │       ├── train.json
    │       ├── dev.json
    │       └── test.json
    ├── ontonotes5_en/
    │   ├── ontonotes5_en.py
    │   └── raw/
    │       ├── train.conll
    │       ├── dev.conll
    │       └── test.conll
    ├── ontonotes5_zh/
    │   ├── ontonotes5_zh.py
    │   └── raw/
    │       ├── train.conll
    │       ├── dev.conll
    │       └── test.conll
    └── mit_movies/
        ├── mit_movies.py
        └── raw/
            ├── train.bio
            ├── dev.bio
            └── test.bio
```

**下载地址：**
即将公布...

## 4. 运行示例

基本命令格式：
```bash
python run.py --datasets <数据集名称> --method <方法> --prompt-types <提示类型> [选项]
```

### LSP 方法（本文提出）
我们的标签子集划分方法：
```bash
# 在 OntoNotes-NLM-en 上使用子集候选提示的 LSP 方法
python run.py --datasets ontonotes5_en --method lsp --prompt-types sc_fs

# 使用 GPT API 推理
python run.py --datasets mit_movies --method lsp --use-api --api-model gpt --prompt-types sc_fs

# 快速测试 20 个样本
python run.py --datasets ontonotes5_en --method lsp --test-subset-size 20 --prompt-types sc_fs --repeat-num 1
```

可用数据集: `ontonotes5_en`, `ontonotes5_zh`, `CMeEE_V2`, `mit_movies`

### Vanilla 少样本基线
Vanilla 少样本设置（不进行子集划分的完整演示）：
```bash
# 单类型少样本（每个演示一个实体类型）
python run.py --datasets ontonotes5_en --method lsp --prompt-types st_fs

# 多类型少样本（一个演示包含所有实体类型）
python run.py --datasets ontonotes5_en --method lsp --prompt-types mt_fs
```

## 5. 引用

即将公布...
