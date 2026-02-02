"""
Core loader functionality for converting HuggingFace datasets to analysis format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


class HFLoader:
    """Loads HuggingFace datasets and converts to analysis format."""

    def __init__(self, dataset_name: str, token: Optional[str] = None):
        """Initialize loader with dataset name.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'isaacchung/controlled-generated-convos-gpt-4.1-mini')
            token: Optional HuggingFace token for private datasets
        """
        self.dataset_name = dataset_name
        self.token = token

    def load(
        self,
        languages: Optional[List[str]] = None,
        max_conversations: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load dataset and convert to analysis format.

        Args:
            languages: Optional list of language codes to include (e.g., ['et', 'fi'])
            max_conversations: Optional limit on conversations per language

        Returns:
            Dictionary mapping model names to list of conversations
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        # Load dataset from HuggingFace
        dataset_dict = load_dataset(self.dataset_name, token=self.token)

        if not isinstance(dataset_dict, DatasetDict):
            # Single split dataset
            dataset_dict = DatasetDict({"default": dataset_dict})

        # Filter languages if specified
        if languages:
            available = set(dataset_dict.keys())
            requested = set(languages)
            missing = requested - available
            if missing:
                logger.warning(f"Languages not found in dataset: {missing}")
            dataset_dict = DatasetDict({
                lang: ds for lang, ds in dataset_dict.items()
                if lang in languages
            })

        logger.info(f"Available languages: {list(dataset_dict.keys())}")

        # Convert to analysis format
        return self._convert_to_analysis_format(dataset_dict, max_conversations)

    def _convert_to_analysis_format(
        self,
        dataset_dict: DatasetDict,
        max_conversations: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Convert flattened message dataset to conversation-based format.

        Args:
            dataset_dict: HuggingFace DatasetDict with language subsets
            max_conversations: Optional limit on conversations per language

        Returns:
            Dictionary mapping model names to list of conversations
        """
        # Group all messages by model, then by conversation_id
        model_conversations: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

        total_messages = 0
        for lang, dataset in dataset_dict.items():
            logger.info(f"Processing {lang}: {len(dataset)} messages")

            for row in dataset:
                total_messages += 1
                conv_id = row["conversation_id"]
                model = row.get("model") or row.get("file_model", "unknown")

                if conv_id not in model_conversations[model]:
                    # Initialize conversation
                    model_conversations[model][conv_id] = {
                        "subject": row.get("subject", ""),
                        "messages": [],
                        "source_file": f"hf://{self.dataset_name}/{lang}",
                        "_metadata": {
                            "language": row.get("language") or row.get("file_language", lang),
                            "industry": row.get("industry", ""),
                            "problem": row.get("problem", ""),
                            "channel": row.get("channel", ""),
                            "agent_experience": row.get("agent_experience", ""),
                            "agent_type": row.get("agent_type", ""),
                            "n_messages": row.get("n_messages", 0),
                            "n_agents": row.get("n_agents", 0),
                        },
                        "_message_ids": [],  # For sorting
                    }

                # Add message
                model_conversations[model][conv_id]["messages"].append({
                    "from_name": row.get("from_name", ""),
                    "from_type": row.get("from_type", ""),
                    "message": row.get("message", ""),
                })
                model_conversations[model][conv_id]["_message_ids"].append(
                    row.get("message_id", len(model_conversations[model][conv_id]["messages"]) - 1)
                )

        logger.info(f"Processed {total_messages} messages total")

        # Sort messages within each conversation and convert to final format
        result: Dict[str, List[Dict[str, Any]]] = {}

        for model, conversations in model_conversations.items():
            conv_list = []

            for conv_id, conv_data in conversations.items():
                # Sort messages by message_id
                message_ids = conv_data.pop("_message_ids")
                sorted_indices = sorted(range(len(message_ids)), key=lambda i: message_ids[i])
                conv_data["messages"] = [conv_data["messages"][i] for i in sorted_indices]

                # Update n_messages if not set
                if not conv_data["_metadata"]["n_messages"]:
                    conv_data["_metadata"]["n_messages"] = len(conv_data["messages"])

                conv_list.append(conv_data)

            # Apply max_conversations limit if specified
            if max_conversations and len(conv_list) > max_conversations:
                conv_list = conv_list[:max_conversations]

            result[model] = conv_list
            logger.info(f"Model {model}: {len(conv_list)} conversations")

        return result

    def save(
        self,
        output_path: Path,
        languages: Optional[List[str]] = None,
        max_conversations: Optional[int] = None
    ) -> Dict[str, int]:
        """Load dataset and save to JSON file.

        Args:
            output_path: Path to output JSON file
            languages: Optional list of language codes to include
            max_conversations: Optional limit on conversations per language

        Returns:
            Dictionary with statistics (conversations per model)
        """
        data = self.load(languages=languages, max_conversations=max_conversations)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved to {output_path}")

        # Return statistics
        stats = {model: len(convs) for model, convs in data.items()}
        return stats
