# Finno-Ugric Conversation Diversity Metrics

Automatic metrics for assessing dataset diversity and detecting formulaic patterns across different LLM-generated conversations in Estonian, Hungarian, and Finnish. This tool implements paper metrics for cross-lingual stability analysis of LLM judges.

## Features

- **Multi-language support**: Estonian (EstNLTK), Hungarian (Stanza), Finnish (Stanza)
- **Paper metrics implementation**:
  - Type-Token Ratio (TTR) - lexical diversity
  - Moving Average TTR (MATTR) - length-independent lexical diversity
  - Self-BLEU (Full/Agent/Client) - formulaic pattern detection
  - Intra-Model Similarity - semantic similarity within model conversations
- **Rich table visualization** with color-coded metrics
- **Multi-model comparison** - analyze all models or single model
- **CSV export** for further analysis
- **Helper scripts** for displaying metric statistics

## Environment Setup
   ```
   cd ~/2025_mme_workshop
   conda create -n mme_workshop python=3.11
   conda activate mme_workshop
   ```

2. **Install uv and dependencies:**
   ```bash
   pip install uv
   uv pip install -e .
   ```

## Language Processing Tools
We provide installing commands in case uv and dependencies installation fails.

### Estonian - EstNLTK
EstNLTK (Estonian Natural Language Toolkit) v1.7.4 provides optimal morphological analysis for Estonian:

```bash
pip install estnltk
```
Script falls back to regex tokenization if EstNLTK unavailable.

### Hungarian & Finnish - Stanza
Stanza provides morphological analysis for Hungarian and Finnish:
```bash
pip install stanza
python -c "import stanza; stanza.download('hu')"  # Hungarian
python -c "import stanza; stanza.download('fi')"  # Finnish
```
The scripts will fall back to regex-based tokenization if language-specific tools are unavailable.

### Missing NLTK Data

Script auto-downloads required NLTK data. If issues occur:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Step 1. Prepare Data Input
The script can process conversations in two ways:

1. **JSON file**: Place your conversation data in the expected JSON format at the default path or specify a custom path using `--json-file` argument.
2. **Built-in examples**: metric.py has customer service conversation examples for each language for testing the metrics. If JSON files are not found, the script falls back to built-in example conversations.

### Default JSON File Paths
- **Estonian**: `../../paper_data/combined_et.json`
- **Hungarian**: `../../paper_data/combined_hu.json`
- **Finnish**: `../../paper_data/combined_fi.json`

### JSON Structure
The JSON files follow a unified structure where model names are top-level keys, each containing an array of conversation objects:
```json
{
  "model-name": [
    {
      "subject": "...",
      "messages": [
        {
          "from_name": "Agent Name",
          "from_type": "agent",
          "message": "..."
        },
        {
          "from_name": "Customer Name",
          "from_type": "customer",
          "message": "..."
        }
      ],
      "model": "model-name",
      "_metadata": {
        "language": "et|hu|fi",
        "industry": "...",
        "problem": "place_order|cancel_order|return_request|general_inquiry",
        "n_messages": 4,
        "n_agents": 2,
        "channel": "chat|email|phone",
        "agent_experience": "junior|senior",
        "agent_type": "bot|human",
        "model": "model-name"
      },
      "source_file": ""
    }
  ]
}
```
Note that we have an experimental dataset in en as well, but no automatic metrics are run on that.

## Step 2. Run metrics on the conversations
**Analyze Estonian conversations with all models (default):**
```bash
python metrics.py
python metrics.py -l et
python metrics.py -l et -m all
```

**Analyze Hungarian conversations with all models:**
```bash
python metrics.py -l hu
```

**Analyze Finnish conversations with all models:**
```bash
python metrics.py -l fi
```

**Analyze specific model on Estonian conversations:**
```bash
python metrics.py -m gpt-4.1-mini
python metrics.py -l et -m gpt-4.1-mini
python metrics.py -l et -m gpt-4.1-mini --json-file /path/to/custom.json
```
**Analyze specific model on Finnish/Hungarian conversations:**
```bash
python metrics.py -l fi -m meta.llama3-8b-instruct-v1:0
python metrics.py -l hu -m cohere.command-r-v1:0
```

**Test with limited conversations:**
```bash
python metrics.py -l hu --limit 10  # Analyze only 10 conversations per model on Hungarian conversations
python metrics.py -l fi -m gpt-4.1-mini --limit 10  # Analyze only 10 conversations and gpt-4.1-mini on Finnish conversations
```
**Specify custom paths for JSON files:**
```bash
# Custom JSON file for any language
python metrics.py -l hu --json-file /path/to/custom_conversations.json
python metrics.py -l et --json-file /path/to/custom_et.json
python metrics.py -l fi -m gpt-4.1-mini --json-file /path/to/custom_fi.json
```

### Available Models

The script supports the following models:
- `anthropic.claude-sonnet-4-20250514-v1:0`
- `cohere.command-r-v1:0`
- `gpt-4.1-mini`
- `meta.llama3-70b-instruct-v1:0`,
- `meta.llama3-8b-instruct-v1:0`
- `mistral.mixtral-8x7b-instruct-v0:1`

### Output Files for metrics.py

**CSV Files** (one per language, appear in the same folder as metrics.py):
- `estonian_conversation_metrics.csv` - Estonian metrics per conversation
- `hungarian_conversation_metrics.csv` - Hungarian metrics per conversation
- `finnish_conversation_metrics.csv` - Finnish metrics per conversation

**CSV Columns**:
- `conversation_id` - Conversation index
- `ttr` - Type-Token Ratio (per conversation)
- `mattr` - Moving Average TTR (per conversation)
- `dataset_full_conversation_self_bleu_self_bleu` - Full conversation Self-BLEU (dataset-level)
- `dataset_agent_response_self_bleu_self_bleu` - Agent Self-BLEU (dataset-level)
- `dataset_client_response_self_bleu_self_bleu` - Client Self-BLEU (dataset-level)
- `avg_similarity` - Average intra-model semantic similarity (dataset-level)
- `std_similarity` - Standard deviation of intra-model similarity
- `min_similarity` - Minimum pairwise similarity
- `max_similarity` - Maximum pairwise similarity
- `median_similarity` - Median pairwise similarity
- `model_name` - Model identifier

## Step 3. Display the metrics with their Standard Deviation
For that we have a script to display various metrics from conversation metrics CSV files:
- **Intra-Model Similarity** (with std)
- **Self-BLEU Scores** (full, agent, client - no std available)
- **TTR/MATTR** (Type-Token Ratio and Moving Average TTR)

### Options
- `-l, --language {et,hu,fi,all}` - Language to display (default: all)
- `--metric {similarity,self-bleu,ttr}` - Metric to display (default: similarity)
- `--format {compact,table,detailed}` - Output format (default: compact)
- `--precision INT` - Decimal precision for values (default: 2)

### Usage
**Show TTR and MATTR for Finnish:**
```bash
python show_intra_similarity.py -l fi --metric ttr
```
**Show Intra-Model Similarity for Hungarian:**
```bash
python show_intra_similarity.py -l hu #similarity is default metric
python show_intra_similarity.py -l hu --metric similarity
```
**Show self-BLEU for Estonian:** (full, agent, client - no std available)
```bash
python show_intra_similarity.py -l et --metric self-bleu
```
**Show TTR for all languages:**
```bash
python show_intra_similarity.py --metric ttr  # By default shows Estonian, Hungarian, Finnish
```
**Show Intra-Model Similarity for all languages in compact format:**
```bash
python show_intra_similarity.py
```
**Detailed table format:**
```bash
python show_intra_similarity.py -l hu # By default shows compact
python show_intra_similarity.py -l hu --format table
python show_intra_similarity.py -l et --metric similarity --format detailed
```

# Output Formats

**1. Compact (default)**
Quick overview showing `Avg ± Std` for each model:
```
🇭🇺 Hungarian Intra-Model Similarity

│ cohere.command-r-v1:0                    │  0.89±0.03
│ gpt-4.1-mini                             │  0.93±0.02
...
```
**2. Table**
Rich table with all statistics:
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Model              ┃ Avg ± Std ┃    Min ┃    Max ┃ Median ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ cohere.command-r   │ 0.89±0.03 │ 0.7917 │ 0.9687 │ 0.8876 │
...
```
**3. Detailed**
Multi-line format with all metrics:
```
cohere.command-r-v1:0:
  Avg Similarity: 0.8877
  Std Similarity: 0.0268
  Min Similarity: 0.7917
  Max Similarity: 0.9687
  Median Similarity: 0.8876
```


### Core Modules

1. **metrics.py** - Main entry point
   - Argument parsing (`-l`, `-m`, `--limit`)
   - Language-specific workflow orchestration
   - JSON data loading
   - CSV export coordination

2. **conversation_metrics.py** - Base classes and shared functionality
   - `ConversationData` - Structured conversation data
   - `ConversationTurn` - Individual speaker turn
   - `ConversationMetrics` - Parent class with shared methods:
     - `self_bleu()` - Language-agnostic Self-BLEU calculation
     - `create_model_metrics_table()` - Generic Rich table display
     - `save_model_results_to_csv()` - CSV export helper
     - Semantic model initialization (multilingual-e5-large-instruct)

3. **et_metrics.py** - Estonian-specific metrics
   - `EstonianConversationMetrics(ConversationMetrics)`
   - EstNLTK integration for morphological analysis
   - Estonian stopwords (5128 words)
   - Methods:
     - `estonian_type_token_ratio()`
     - `estonian_moving_average_ttr()`
     - `estonian_conversation_self_bleu()`
     - `calculate_intra_model_conversation_similarity()`
     - `batch_analyze_estonian_conversations()`

4. **hu_metrics.py** - Hungarian-specific metrics
   - `HungarianConversationMetrics(ConversationMetrics)`
   - Stanza integration for Hungarian morphological analysis
   - Hungarian stopwords (826 words)
   - Methods:
     - `hungarian_type_token_ratio()`
     - `hungarian_moving_average_ttr()`
     - `hungarian_conversation_self_bleu()`
     - `calculate_intra_model_conversation_similarity()`
     - `batch_analyze_hungarian_conversations()`

5. **fi_metrics.py** - Finnish-specific metrics
   - `FinnishConversationMetrics(ConversationMetrics)`
   - Stanza integration for Finnish morphological analysis
   - Finnish stopwords (885 words)
   - Methods:
     - `finnish_type_token_ratio()`
     - `finnish_moving_average_ttr()`
     - `finnish_conversation_self_bleu()`
     - `calculate_intra_model_conversation_similarity()`
     - `batch_analyze_finnish_conversations()`

6. **show_intra_similarity.py** - Metric display utility
   - Reads CSV files and displays formatted statistics
   - Supports 3 metrics: `similarity`, `self-bleu`, `ttr`
   - 3 display formats: `compact` (default), `table`, `detailed`
   - Handles multiple column naming conventions

## Metrics Explained

### Type-Token Ratio (TTR)
**Formula**: `unique_words / total_words`

**Range**: 0.0 - 1.0  
**Better**: Higher (more diverse vocabulary)

**Description**: Measures lexical diversity by comparing unique words to total words. Uses language-specific lemmatization (EstNLTK for Estonian, Stanza for Hungarian/Finnish) and stopword filtering.

**Reference Values**:
- 0.3-0.5: Low diversity, repetitive language
- 0.5-0.7: Moderate diversity, typical conversation
- 0.7-0.9: High diversity, rich vocabulary
- 0.9+: Very high diversity, academic/literary text

### Moving Average Type-Token Ratio (MATTR)
**Formula**: Average TTR across sliding windows of 100 tokens

**Range**: 0.0 - 1.0  
**Better**: Higher (more consistent lexical diversity)

**Description**: Length-independent version of TTR. Calculates TTR for each 100-token window and averages the results. More reliable for comparing texts of different lengths.

**Reference Values**:
- 0.4-0.6: Low lexical diversity, repetitive
- 0.6-0.8: Moderate diversity, natural conversation
- 0.8-0.95: High diversity, rich vocabulary
- 0.95+: Exceptionally diverse, likely academic text

### Self-BLEU (Full/Agent/Client)
**Formula**: For each text, calculate BLEU score against all other texts in dataset

**Range**: 0.0 - 1.0  
**Better**: Lower (more diverse, less formulaic)

****Language-specific NLP tools**:
  - EstNLTK (Estonian morphological analysis)
  - Stanza (Hungarian and Finnish morphological analysis)
- **ML models**:
  - Transformers and Sentence-Transformers
  - intfloat/multilingual-e5-large-instruct (semantic embeddings)
- **Language-Specific NLP Tools


### Memory Issues

For large datasets:
- Use `--limit` to process fewer conversations: `python metrics.py -l hu --limit 50`
- Process one model at a time: `python metrics.py -l et -m gpt-4.1-mini`
- Use CPU-only PyTorch if GPU memory limited


### CSV Column Names

The `show_intra_similarity.py` script handles multiple column naming conventions:
- `ttr` or `type_token_ratio` or `lexical_diversity_type_token_ratio`
- `mattr` or `moving_avg_ttr` or `lexical_diversity_moving_avg_ttr`
- `dataset_*_self_bleu_self_bleu` or `*_self_bleu`

If you encounter column errors, check CSV headers with:
```bash
head -1 hungarian_conversation_metrics.csv
```
## Note
read_and_evaluate_manual_ratings.py is 

## Contributing

This tool is part of the paper "Cross-Lingual Stability of LLM Judges Under Controlled Generation: Evidence from Finno-Ugric Languages" by Isaac Chung and Linda Freienthal. See the main repository's contribution guidelines for development practices.