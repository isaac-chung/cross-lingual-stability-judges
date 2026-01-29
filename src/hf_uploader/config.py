"""
Configuration module for HuggingFace dataset upload functionality.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class HFConfig:
    """Configuration for HuggingFace dataset upload operations."""

    token: str
    username: str
    private: bool = False
    license: str = "mit"
    description_template: str = "Synthetic customer support conversations generated using {model}"
    overwrite: bool = False
    dataset_name_override: Optional[str] = None
    custom_description: Optional[str] = None

    @classmethod
    def from_environment(
        cls,
        private: bool = False,
        overwrite: bool = False,
        dataset_name_override: Optional[str] = None,
        custom_description: Optional[str] = None
    ) -> "HFConfig":
        """Create configuration from environment variables.

        Args:
            private: Whether to make datasets private
            overwrite: Whether to allow overwriting existing datasets
            dataset_name_override: Custom dataset name to use
            custom_description: Custom description for the dataset

        Returns:
            HFConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        # Load environment variables from .env file
        load_dotenv()

        token = os.getenv("HF_TOKEN")
        username = os.getenv("HF_USERNAME")

        if not token:
            raise ValueError(
                "HF_TOKEN environment variable is required. "
                "Get your token from https://huggingface.co/settings/tokens"
            )

        if not username:
            raise ValueError(
                "HF_USERNAME environment variable is required. "
                "This should be your HuggingFace username."
            )

        return cls(
            token=token,
            username=username,
            private=private,
            overwrite=overwrite,
            dataset_name_override=dataset_name_override,
            custom_description=custom_description
        )

    def get_dataset_name(self, model: str) -> str:
        """Generate dataset name for a given model.

        Args:
            model: Model name (e.g., "gpt-4.1-mini")

        Returns:
            Dataset name (e.g., "conversations-gpt-4.1-mini")
        """
        if self.dataset_name_override:
            return self.dataset_name_override

        # Clean model name for dataset naming
        clean_model = model.replace("/", "-").replace("_", "-").lower()
        return f"controlled-generated-convos-{clean_model}"

    def get_description(self, model: str, languages: list[str]) -> str:
        """Generate dataset description.

        Args:
            model: Model name
            languages: List of language codes

        Returns:
            Dataset description
        """
        if self.custom_description:
            return self.custom_description

        description = self.description_template.format(model=model)
        if languages:
            lang_str = ", ".join(languages)
            description += f" Available in languages: {lang_str}."

        return description


# Default supported languages
SUPPORTED_LANGUAGES = ["et", "fi", "hu", "en-us"]

# Dataset card template
DATASET_CARD_TEMPLATE = """---
license: mit
task_categories:
- text-generation
- text-classification
- conversational
language:
{yaml_languages}
tags:
- customer-support
- synthetic-data
- multilingual
- conversation
- cross-lingual
- finno-ugric
- {model_tag}
size_categories:
- {size_category}
---

# Controlled Generated Conversations: {model}

## Dataset Description

This dataset contains **synthetic customer support conversations** generated using **{model}** as part of research on cross-lingual stability of LLM judges. The conversations are designed for evaluating how well language models maintain consistent performance across different languages, with a focus on Finno-Ugric languages (Estonian, Finnish, Hungarian) and English.

### Dataset Summary

- **Languages**: {language_list}
- **Domain**: Customer support conversations
- **Generator Model**: {model}
- **Total Messages**: {total_messages:,}
- **Total Conversations**: {total_conversations:,}
- **Generation Method**: Controlled synthetic generation with structured parameters

### Supported Tasks

- **Cross-lingual evaluation**: Compare model performance across languages
- **Conversation analysis**: Study dialogue patterns and structures
- **Quality assessment**: Evaluate linguistic quality (grammar, fluency, coherence)
- **Label recovery**: Classify conversation characteristics (industry, problem type, etc.)
- **Ranking stability**: Analyze consistency of model rankings across languages

## Languages

{language_info}

## Dataset Structure

### Data Instances

Each row represents a **single message** within a conversation, allowing for granular analysis of dialogue turns:

```json
{{
  "conversation_id": "gpt-4.1-mini_et_20260127-173115_0000",
  "message_id": 0,
  "from_name": "Klient",
  "from_type": "customer",
  "message": "Tere! Minu tellimus pidi juba kohal olema...",
  "subject": "Saatmise hilinemine tellimusele",
  "model": "gpt-4.1-mini",
  "language": "et",
  "industry": "Music & Audio",
  "problem": "Shipping delay",
  "channel": "chat",
  "agent_experience": "senior",
  "agent_type": "bot"
}}
```

### Data Fields

#### Core Message Fields
- `conversation_id` (string): Unique identifier for the conversation
- `message_id` (int): Position of message within conversation (0-indexed)
- `from_name` (string): Speaker name (e.g., "Customer", "Agent 1")
- `from_type` (string): Speaker role ("customer" or "agent")
- `message` (string): The actual message content

#### Conversation Metadata
- `subject` (string): Conversation topic/title
- `model` (string): Generator model used
- `language` (string): Language code (et, fi, hu, en-us)

#### Generation Parameters
- `industry` (string): Business domain (40+ categories)
- `problem` (string): Issue type (20+ categories)
- `channel` (string): Communication channel (email, chat)
- `agent_experience` (string): Agent level (junior, senior)
- `agent_type` (string): Agent type (human, bot)
- `n_messages` (int): Total messages in conversation
- `n_agents` (int): Number of agents involved

#### File Provenance
- `file_model` (string): Model from source filename
- `file_language` (string): Language from source filename
- `file_datetime` (string): Generation timestamp from filename

### Data Splits

This dataset uses a single `train` split containing all conversations. For evaluation purposes, users can create their own splits based on:
- Language (cross-lingual evaluation)
- Industry/problem type (domain adaptation)
- Conversation length (complexity analysis)

## Dataset Creation

### Curation Rationale

This dataset was created to study **cross-lingual stability** of language models in evaluation tasks. Finno-Ugric languages (Estonian, Finnish, Hungarian) were specifically chosen as they are:
- Morphologically complex with rich inflectional systems
- Less represented in training data compared to English
- Suitable for testing model robustness across linguistic diversity

### Source Data

#### Data Collection
Conversations were synthetically generated using structured prompts with controlled parameters:

1. **Industry Selection**: Random sampling from 40+ business domains
2. **Problem Generation**: 20+ customer support issue types
3. **Channel & Agent**: Varied communication contexts
4. **Language**: Native speaker patterns for each target language

#### Data Processing
- **Message-level structuring**: Each conversation split into individual messages
- **Unique ID generation**: Traceable identifiers linking messages to conversations
- **Metadata preservation**: All generation parameters maintained for analysis
- **Quality validation**: Automatic filtering for completeness and format

### Annotation Process

No human annotation was performed. All labels represent the **controlled generation parameters** used to create each conversation, enabling evaluation of how well models can recover the original intent.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset enables research into:
- **Language equity**: Understanding model performance disparities across languages
- **Evaluation fairness**: Developing more inclusive assessment methods
- **Multilingual capabilities**: Improving cross-lingual transfer in NLP systems

### Discussion of Biases

**Limitations:**
- **Synthetic nature**: May not capture all nuances of real customer support
- **Generation model bias**: Inherits biases from the generator model ({model})
- **Language representation**: Quality may vary based on model's training data per language
- **Cultural context**: May not reflect local business practices or communication styles

**Mitigation strategies:**
- Use alongside real conversation data when possible
- Consider cultural and linguistic context in analysis
- Validate findings across multiple generator models

### Other Known Limitations

- **Temporal coverage**: Generated at a single point in time
- **Domain scope**: Limited to customer support scenarios
- **Generator dependency**: Results tied to specific model capabilities
- **Language variety**: Standard varieties only, no dialectal variation

## Additional Information

### Dataset Curators

Created by the Cross-lingual Stability of LLM Judges research project.

### Licensing Information

Licensed under MIT License. See LICENSE file for details.

### Citation Information

If you use this dataset in your research, please cite:

```bibtex
@dataset{{controlled_generated_convos_{clean_model},
    title={{Controlled Generated Conversations: Cross-lingual Customer Support Dialogues - {model}}},
    author={{{username}}},
    year={{2024}},
    publisher={{Hugging Face}},
    url={{https://huggingface.co/datasets/{full_dataset_name}}},
    note={{Synthetic customer support conversations for cross-lingual evaluation research}}
}}
```

### Contributions

Thanks to the contributors of this dataset and the broader research community working on cross-lingual NLP evaluation.

## Statistics

{statistics}

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset

# Load specific language subset
dataset = load_dataset("{full_dataset_name}", "et")
print(f"Estonian subset: {{len(dataset)}} messages")

# Load all subsets
dataset = load_dataset("{full_dataset_name}")
print(f"Available languages: {{list(dataset.keys())}}")

# Access conversation metadata
et_data = dataset["et"]
first_msg = et_data[0]
print(f"Industry: {{first_msg['industry']}}")
print(f"Problem: {{first_msg['problem']}}")
```

### Basic Analysis

```python
# Group messages by conversation
conversations = {{}}
for msg in dataset["et"]:
    conv_id = msg["conversation_id"]
    if conv_id not in conversations:
        conversations[conv_id] = []
    conversations[conv_id].append(msg)

# Analyze conversation lengths
lengths = [len(conv) for conv in conversations.values()]
print(f"Avg conversation length: {{sum(lengths)/len(lengths):.1f}} messages")
```

### Cross-lingual Analysis

```python
# Compare industry distribution across languages
for lang in dataset.keys():
    industries = [msg["industry"] for msg in dataset[lang]]
    unique_industries = set(industries)
    print(f"{{lang}}: {{len(unique_industries)}} unique industries")
```
"""