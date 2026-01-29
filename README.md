# Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages

Code for the [2026 EACL MME Workshop](https://multilingual-multicultural-evaluation.github.io/) paper: [OpenReview Link](https://openreview.net/forum?id=OgcrR6WgWp)

## Setup

```bash
pip install -e .
cp .env.example .env  # then add your OPENAI_API_KEY
```

This project uses [liteLLM](https://docs.litellm.ai/) for unified LLM provider support, enabling seamless switching between OpenAI, Groq, Anthropic, and other providers while maintaining identical APIs and functionality. Use the `--provider` flag to switch between providers (OpenAI, Groq, etc.) without changing any other code.

## Step 1: Generate Conversations

Generate synthetic customer support conversations in Finno-Ugric languages (Estonian, Finnish, Hungarian) and English. Agent messages use actual agent names (e.g., "Amanda S.") instead of generic identifiers like "Agent 1".

```bash
# OpenAI provider (default)
python -m conversation_generator -n 100 -l et  # 100 Estonian conversations
python -m conversation_generator -n 100 -l fi  # 100 Finnish conversations
python -m conversation_generator -n 100 -l hu  # 100 Hungarian conversations

# Groq provider with different models
python -m conversation_generator -n 100 -l fi --provider groq -m qwen/qwen3-32b

python -m conversation_generator --help        # all options
```

Output files are named `convo_{model}_{lang}_{datetime}.jsonl` (e.g., `convo_gpt-4.1-mini_et_20260127-143052.jsonl`).

## Step 2: Combine and Validate

Combine JSONL files from multiple generation runs into a single JSON file grouped by model. The combined output is the central file used for all downstream analyses.

```bash
# Combine all languages
python -m conversation_parser data/convo_*.jsonl --stats

# Combine single language files
python -m conversation_parser data/convo_*_et_*.jsonl --stats
```

Output is JSON grouped by model:
```json
{
  "gpt-4.1-mini": [
    {"subject": "...", "messages": [...], "source_file": "...", "_metadata": {...}},
    ...
  ],
  "gpt-4": [...]
}
```

Output files are named `combined_{lang}.json`:
- Single language input → `combined_et.json`
- Multiple languages → `combined_mixed.json`

Use `--stats` alone to validate without writing output:
```bash
python -m conversation_parser data/convo_*.jsonl --stats
```

## Step 3: LLM-as-a-Judge Evaluation

Evaluate conversations on linguistic quality using an LLM judge. Supports multiple providers via liteLLM.

```bash
# OpenAI provider (default)
python -m llm_judge data/combined_et.json --stats

# Use different reasoning effort (supported by compatible models)
python -m llm_judge data/combined_fi.json --reasoning-effort high --stats

# Groq provider with various models
python -m llm_judge data/combined_et.json --provider groq -m qwen/qwen3-32b

# Specify output path
python -m llm_judge data/combined_hu.json -o results/judge_hu.jsonl
```

Output is JSONL with scores for each conversation:
```json
{
  "conversation_id": 0,
  "generator_model": "gpt-4.1-mini",
  "judge_model": "gpt-5-mini",
  "G": 4, "R": 3, "C": 3, "F": 2,
  "explanation": "...",
  "source_file": "data/combined_et.json"
}
```

**Scoring criteria:**
- **Grammaticality (G)**: 0-4 - grammatical correctness
- **Readability (R)**: 0-4 - ease of reading, natural flow
- **Content Coherence (C)**: 0-3 - logical customer support dialogue
- **Fluency (F)**: 0-3 - native speaker naturalness

### Analyze Existing Results

Load and analyze existing JSONL results without re-running evaluation:

```bash
# Analyze single file
python -m llm_judge --from-results data/judge_et_20260127-143052.jsonl

# Analyze multiple files (supports glob patterns)
python -m llm_judge --from-results data/judge_et_*.jsonl

# Analyze all results
python -m llm_judge --from-results data/judge_*.jsonl
```

Output includes per-model statistics (mean, std, min, max, distribution) and a model comparison table.

## Step 4: Label Recovery Classification

Classify conversations to recover original generation parameters (industry, problem type, channel, agent experience, agent type). This enables evaluation of how well models preserve the intended characteristics.

```bash
# OpenAI provider (default)
python -m label_recovery data/combined_et.json --stats

# Groq provider with various models
python -m label_recovery data/combined_fi.json --provider groq -m qwen/qwen3-32b

# Specify output path
python -m label_recovery data/combined_et.json -o results/label_recovery_et.jsonl
```

Output is JSONL with classifications for each conversation:
```json
{
  "conversation_id": 0,
  "generator_model": "gpt-4.1-mini",
  "judge_model": "gpt-5-mini",
  "industry": "e-commerce",
  "problem": "payment_issue",
  "channel": "chat",
  "agent_experience": "senior",
  "agent_type": "human",
  "explanation": "...",
  "source_file": "data/combined_et.json"
}
```

**Classification categories:**
- **Industry**: 40+ specific industries (manufacturing, e-commerce, retail, automotive, etc.)
- **Problem**: 20 types (create_account, delete_account, payment_issue, complaint, etc.)
- **Channel**: email, chat
- **Agent Experience**: junior, senior
- **Agent Type**: human, bot

### Analyze Existing Results

Load and analyze existing JSONL results without re-running classification:

```bash
# Analyze single or multiple files (supports glob patterns)
python -m label_recovery --from-results data/label_recovery_*.jsonl
```

### Evaluate Against Ground Truth

Compare predictions against ground truth configuration files to calculate accuracy and F1 scores:

```bash
# Evaluate predictions against ground truth
python -m label_recovery --evaluate data/label_recovery_*.jsonl --ground-truth data/config.json
```

**Ground truth JSON format:**
```json
[
  {
    "ticket_id": "id_0",
    "config": {
      "industry": "e-commerce",
      "problem": "payment_issue",
      "channel": "chat",
      "agent_experience": "senior",
      "agent_type": "human"
    }
  }
]
```

The evaluation displays Rich tables showing:
- Overall accuracy and F1 scores per category
- Detailed per-category breakdown with macro and weighted F1

## Step 5: Judge Ablation Analysis

Compare label recovery results from different judge models to analyze judge consistency and inter-judge agreement.

```bash
# Compare results from different judge models
python -m judge_ablation data/label_recovery_*.jsonl --ground-truth data/combined_et.json

# Save detailed results to JSON
python -m judge_ablation data/label_recovery_*.jsonl --ground-truth config.json -o results.json

# Load and display saved analysis results
python -m judge_ablation --from-results results.json
```

The analysis displays:
- **Overall Comparison Table**: Average accuracy and std dev per judge model
- **Per-Category Tables**: Accuracy breakdown by category (industry, problem, channel, etc.)
- **Inter-Judge Agreement**: Pairwise agreement between judges on same conversations

**Output filename format:**
- Label recovery: `data/label_recovery_{model}_{lang}_{datetime}.jsonl`
- LLM judge: `data/judge_{model}_{lang}_{datetime}.jsonl`

For models using reasoning effort, the model name includes the effort level:
- `data/label_recovery_o3-medium_et_20260127-143052.jsonl`

## Step 7: Ranking Inversions Analysis

Analyze cross-language ranking stability by computing ranking correlations (Kendall tau, Spearman rho) and pairwise inversions across language pairs.

```bash
# Analyze judge results across languages
python -m ranking_inversions data/judge_*.jsonl --languages et fi hu en

# Combined analysis with label recovery
python -m ranking_inversions --judge data/judge_*.jsonl \
                              --label-recovery data/label_recovery_*.jsonl \
                              --ground-truth data/combined_et.json

# Quick analysis without bootstrap/permutation tests (faster)
python -m ranking_inversions data/judge_*.jsonl --no-bootstrap --no-permutation

# Save results for later analysis
python -m ranking_inversions data/judge_*.jsonl --output analysis.json

# Display previously saved results
python -m ranking_inversions --from-results analysis.json
```

The analysis computes:
- **Kendall τ** and **Spearman ρ** correlations between model rankings across language pairs
- **Ranking inversions**: count of model pairs with different relative ordering between languages
- **Bootstrap confidence intervals**: uncertainty estimates for correlations (default: 2000 iterations)
- **Permutation tests**: statistical significance of inversions (default: 5000 iterations)

**Input metrics:**
- From llm_judge: G (Grammar), R (Readability), C (Coherence), F (Fluency)
- From label_recovery: LRA (Label Recovery Accuracy)

**Output displays:**
- Rich tables showing pairwise comparisons between language pairs
- Per-metric correlation statistics with confidence intervals
- Model ranking tables by language and metric
- Statistical significance indicators

## Step 6: Upload to HuggingFace Hub

Upload conversation datasets to HuggingFace Hub for sharing and collaboration. The uploader automatically organizes datasets by model with language subsets and supports incremental updates.

### Setup

Add HuggingFace credentials to your `.env` file:

```bash
HF_TOKEN=hf_...                 # Get from https://huggingface.co/settings/tokens
HF_USERNAME=your-username       # Your HuggingFace username
```

### Basic Usage

```bash
# Upload all conversations for a specific model (creates/updates dataset)
python -m hf_uploader --model gpt-4.1-mini

# Upload specific files using glob pattern
python -m hf_uploader --files "data/convo_gpt-4.1_*.jsonl"

# Preview what would be uploaded without actually uploading
python -m hf_uploader --model gpt-4.1-mini --dry-run
```

### Dataset Organization

The uploader creates datasets organized as:
- **Dataset name**: `controlled-generated-convos-{model}` (e.g., `controlled-generated-convos-gpt-4.1-mini`)
- **Language subsets**: `et`, `fi`, `hu`, `en-us` (separate subset for each language)
- **Message structure**: Each row is a single message with conversation metadata

Example dataset URL: `https://huggingface.co/datasets/isaac-chung/controlled-generated-convos-gpt-4.1-mini`

### Incremental Updates

**Default behavior** - Smart merging with existing datasets:
```bash
# Adds new conversations to existing dataset
python -m hf_uploader --model gpt-4.1-mini

# Multiple files of same model/language automatically combine
python -m hf_uploader --files "data/convo_gpt-4.1-mini_et_*.jsonl"
```

**How incremental updates work:**
- **Same model + new language** → Creates new language subset
- **Same model + same language** → Appends new conversations to existing subset
- **Duplicate conversations** → Automatically detected and skipped
- **Different models** → Creates separate datasets

**Override behavior** - Replace entire dataset:
```bash
# Replace existing dataset entirely instead of merging
python -m hf_uploader --model gpt-4.1-mini --overwrite
```

### Advanced Options

```bash
# Filter by specific languages
python -m hf_uploader --model gpt-4.1-mini --languages et,fi

# Create private dataset
python -m hf_uploader --model gpt-4.1-mini --private

# Custom dataset name and description
python -m hf_uploader --model gpt-4.1-mini \
  --dataset-name my-conversations \
  --description "Custom dataset description"

# Verbose output for debugging
python -m hf_uploader --model gpt-4.1-mini --verbose
```

### Example Workflows

**Initial upload** - Create new dataset:
```bash
# Generate conversations
python -m conversation_generator -n 50 -l et -m gpt-4.1-mini
python -m conversation_generator -n 50 -l fi -m gpt-4.1-mini

# Upload to HuggingFace (creates new dataset)
python -m hf_uploader --model gpt-4.1-mini
# Creates: controlled-generated-convos-gpt-4.1-mini with et and fi subsets
```

**Adding more data** - Incremental update:
```bash
# Generate more conversations
python -m conversation_generator -n 30 -l hu -m gpt-4.1-mini  # New language
python -m conversation_generator -n 20 -l et -m gpt-4.1-mini  # More Estonian

# Upload new conversations (merges with existing)
python -m hf_uploader --model gpt-4.1-mini
# Result: et subset grows, hu subset added, fi subset unchanged
```

**Multiple models** - Creates separate datasets:
```bash
# Upload different models
python -m hf_uploader --model gpt-4.1-mini    # → controlled-generated-convos-gpt-4.1-mini
python -m hf_uploader --model gpt-4.1         # → controlled-generated-convos-gpt-4.1
python -m hf_uploader --model claude-3-sonnet # → controlled-generated-convos-claude-3-sonnet
```

### Dataset Structure

Each uploaded dataset contains message-based rows with rich metadata:

```python
from datasets import load_dataset

# Load specific language subset
dataset = load_dataset("isaac-chung/controlled-generated-convos-gpt-4.1-mini", "et")

# Load all subsets
dataset = load_dataset("isaac-chung/controlled-generated-convos-gpt-4.1-mini")

# Example row structure
print(dataset["et"][0])
# {
#   'conversation_id': 'gpt-4.1-mini_et_20260127-173115_0000',
#   'message_id': 0,
#   'from_name': 'Klient',  # Customer message
#   'from_type': 'customer',
#   'message': 'Tere! Minu tellimus pidi juba kohal olema...',
#   'subject': 'Saatmise hilinemine tellimusele',
#   'model': 'gpt-4.1-mini',
#   'language': 'et',
#   'industry': 'Music & Audio',
#   'problem': 'Shipping delay',
#   # ... plus other metadata
# }
#
# # Agent message example:
# {
#   'from_name': 'Amanda S.', 
#   'from_type': 'agent',
#   'message': '...',
# }
```

## Environment Variables

This project uses liteLLM for multi-provider support. Set the appropriate API keys for your desired providers:

```bash
# OpenAI (default provider)
OPENAI_API_KEY=sk-...           # Required for OpenAI provider
OPENAI_BASE_URL=...             # Optional, defaults to EU endpoint

# Groq
GROQ_API_KEY=gsk_...            # Required for Groq provider

# HuggingFace (for dataset uploads)
HF_TOKEN=hf_...                 # Required for uploading datasets
HF_USERNAME=your-username       # Your HuggingFace username
```
