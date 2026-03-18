# LSP: Empowering Few-Shot NER with Demonstration Augmentation via Label Subset Partition

[中文版本 (Chinese Version)](./README_zh.md)

This repository is the official implementation of the paper "LSP: Empowering Few-Shot NER with Demonstration Augmentation via Label Subset Partition".

## 1. Environment Configuration

Create the conda environment using mamba:

```bash
# Create environment from the yml file
mamba env create -f lsp-env.yml

# Activate the environment
mamba activate lsp

# Export your API keys
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

**Note:** If you only run via API mode (recommended), you don't need to install vLLM and other local inference dependencies. The environment file already includes all necessary dependencies for API mode.

## 2. Configuration

All configuration files are located in the `cfgs/` directory. You need to customize the following settings:

### API Configuration (`cfgs/api.yml`)

**Important:** The `api_key` field in `api.yml` expects the **name of the environment variable**, not the actual API key itself. You need to set the real API key via environment variable.

Example configuration:

```yaml
qwen:
  base_url: your-base-url-for-qwen
  model: qwen1.5-32b-chat
  api_key: DASHSCOPE_API_KEY      # This is the environment variable name
  concurrency_level: 10

gpt:
  base_url: your-base-url-for-openai
  model: gpt-4o-mini
  api_key: OPENAI_API_KEY        # This is the environment variable name
  concurrency_level: 10

deepseek:
  base_url: your-base-url-for-deepseek
  model: deepseek-chat
  api_key: DEEPSEEK_API_KEY      # This is the environment variable name
  concurrency_level: 10
```

Then set your actual API key via environment variable:

```bash
# Example for Linux/Mac
export DASHSCOPE_API_KEY="your-actual-dashscope-api-key-here"
export OPENAI_API_KEY="your-actual-openai-api-key-here"
export DEEPSEEK_API_KEY="your-actual-deepseek-api-key-here"
```

### Other Configurations
- **Dataset configurations**: `cfgs/data_cfgs/<dataset_name>.yml` - defines data paths and dataset-specific settings
- **Label configurations**: `cfgs/label_cfgs/<dataset_name>.yml` - defines entity type label schemas
- **Method-specific configurations**: Check `single_type/`, `multi_type/`, `subset_cand/`, `retrieval_lsp/`, etc. for different prompting strategies

The main configuration entry is `config.yml` which registers all datasets and annotation configurations.

## 3. Dataset Preparation

Download the datasets and organize them following the directory structure below:

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

**Download URLs:**
Coming soon...

## 4. Running Examples

Basic command format:
```bash
python run.py --datasets <dataset_name> --method <method> --prompt-types <prompt_type> [options]
```

### LSP Method (Proposed)
Our Label Subset Partition method:
```bash
# LSP with subset candidate prompting on OntoNotes-NLM-en
python run.py --datasets ontonotes5_en --method lsp --prompt-types sc_fs

# LSP using GPT API for inference
python run.py --datasets mit_movies --method lsp --use-api --api-model gpt --prompt-types sc_fs

# Quick test with 20 samples
python run.py --datasets ontonotes5_en --method lsp --test-subset-size 20 --prompt-types sc_fs --repeat-num 1
```

Available datasets: `ontonotes5_en`, `ontonotes5_zh`, `CMeEE_V2`, `mit_movies`

### Vanilla/Few-Shot Baseline
Vanilla few-shot setting (full demonstration without subset partition):
```bash
# Single-type few-shot (one entity type per demonstration)
python run.py --datasets ontonotes5_en --method lsp --prompt-types st_fs

# Multi-type few-shot (all entity types in one demonstration)
python run.py --datasets ontonotes5_en --method lsp --prompt-types mt_fs
```

## 5. Citation

Coming soon...
