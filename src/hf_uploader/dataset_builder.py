"""
Dataset building functionality for HuggingFace dataset creation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datasets import Dataset, DatasetDict


logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Builds HuggingFace datasets from conversation JSONL files."""

    def __init__(self):
        self.conversation_counter = 0

    def build_from_files(self, files: List[Path]) -> DatasetDict:
        """Build HuggingFace DatasetDict from conversation files.

        Args:
            files: List of Path objects pointing to JSONL conversation files

        Returns:
            DatasetDict with language subsets

        Raises:
            ValueError: If no valid conversations are found
        """
        logger.info(f"Building dataset from {len(files)} files")

        all_conversations = []
        for file_path in files:
            try:
                conversations = self._load_conversations(file_path)
                all_conversations.extend(conversations)
                logger.debug(f"Loaded {len(conversations)} conversations from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load conversations from {file_path}: {e}")
                continue

        if not all_conversations:
            raise ValueError("No valid conversations found in provided files")

        # Group conversations by language
        language_groups = self._group_by_language(all_conversations)
        logger.info(f"Grouped conversations by language: {list(language_groups.keys())}")

        # Build datasets for each language
        dataset_dict = {}
        for language, conversations in language_groups.items():
            messages = self._flatten_to_messages(conversations)
            if messages:
                hf_data = self._structure_for_hf(messages)
                dataset_dict[language] = Dataset.from_dict(hf_data)
                logger.info(f"Created {language} subset with {len(messages)} messages")

        if not dataset_dict:
            raise ValueError("No datasets created - all language groups were empty")

        return DatasetDict(dataset_dict)

    def _load_conversations(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load conversations from a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of conversation dictionaries with file metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        conversations = []
        file_info = self._extract_file_info(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        conversation = json.loads(line)

                        # Add file metadata to conversation
                        conversation["_file_info"] = file_info
                        conversation["_file_path"] = str(file_path)

                        # Generate unique conversation ID
                        conversation["conversation_id"] = self._generate_conversation_id(
                            file_path, len(conversations)
                        )

                        conversations.append(conversation)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue

        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        return conversations

    def _extract_file_info(self, file_path: Path) -> Dict[str, str]:
        """Extract model, language, and datetime from filename.

        Args:
            file_path: Path to conversation file

        Returns:
            Dictionary with extracted information
        """
        filename = file_path.stem

        # Pattern: convo_{model}_{language}_{datetime}
        pattern = r"convo_(.+)_([a-z-]+)_(\d{8}-\d{6})"
        match = re.match(pattern, filename)

        if match:
            return {
                "model": match.group(1),
                "language": match.group(2),
                "datetime": match.group(3)
            }
        else:
            logger.warning(f"Could not parse filename: {filename}")
            return {
                "model": "unknown",
                "language": "unknown",
                "datetime": "unknown"
            }

    def _generate_conversation_id(self, file_path: Path, conv_index: int) -> str:
        """Generate unique conversation identifier.

        Args:
            file_path: Path to the conversation file
            conv_index: Index of conversation within the file

        Returns:
            Unique conversation ID
        """
        file_info = self._extract_file_info(file_path)
        return f"{file_info['model']}_{file_info['language']}_{file_info['datetime']}_{conv_index:04d}"

    def _group_by_language(self, conversations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group conversations by language code.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Dictionary mapping language codes to conversation lists
        """
        language_groups = {}

        for conversation in conversations:
            # Get language from metadata or file info
            language = (
                conversation.get("_metadata", {}).get("language") or
                conversation.get("_file_info", {}).get("language") or
                "unknown"
            )

            if language not in language_groups:
                language_groups[language] = []

            language_groups[language].append(conversation)

        return language_groups

    def _flatten_to_messages(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten conversations to individual message records.

        Args:
            conversations: List of conversation dictionaries

        Returns:
            List of message dictionaries
        """
        messages = []

        for conversation in conversations:
            conversation_id = conversation.get("conversation_id", f"conv_{len(messages)}")
            conv_messages = conversation.get("messages", [])

            if not conv_messages:
                logger.warning(f"No messages in conversation {conversation_id}")
                continue

            for message_idx, message in enumerate(conv_messages):
                message_record = {
                    "conversation_id": conversation_id,
                    "message_id": message_idx,
                    "from_name": message.get("from_name", ""),
                    "from_type": message.get("from_type", ""),
                    "message": message.get("message", ""),
                    # Include conversation-level metadata
                    "subject": conversation.get("subject", ""),
                    "model": conversation.get("model", ""),
                    # File info
                    "file_model": conversation.get("_file_info", {}).get("model", ""),
                    "file_language": conversation.get("_file_info", {}).get("language", ""),
                    "file_datetime": conversation.get("_file_info", {}).get("datetime", ""),
                }

                # Add metadata fields if available
                metadata = conversation.get("_metadata", {})
                if metadata:
                    message_record.update({
                        "language": metadata.get("language", ""),
                        "industry": metadata.get("industry", ""),
                        "problem": metadata.get("problem", ""),
                        "channel": metadata.get("channel", ""),
                        "agent_experience": metadata.get("agent_experience", ""),
                        "agent_type": metadata.get("agent_type", ""),
                        "n_messages": metadata.get("n_messages", 0),
                        "n_agents": metadata.get("n_agents", 0),
                    })

                messages.append(message_record)

        logger.info(f"Flattened {len(conversations)} conversations to {len(messages)} messages")
        return messages

    def _structure_for_hf(self, messages: List[Dict[str, Any]]) -> Dict[str, List]:
        """Structure message data for HuggingFace Dataset format.

        Args:
            messages: List of message dictionaries

        Returns:
            Dictionary with lists of values for each field
        """
        if not messages:
            return {}

        # Get all unique keys from messages
        all_keys = set()
        for message in messages:
            all_keys.update(message.keys())

        # Structure data as lists for HuggingFace
        hf_data = {key: [] for key in sorted(all_keys)}

        for message in messages:
            for key in hf_data:
                value = message.get(key, "")
                # Ensure all values are serializable
                if value is None:
                    value = ""
                elif isinstance(value, (list, dict)):
                    value = str(value)
                hf_data[key].append(value)

        return hf_data

    def count_conversations_from_files(self, files: List[Path]) -> Dict[str, int]:
        """Count total conversations from files.

        Args:
            files: List of Path objects pointing to JSONL conversation files

        Returns:
            Dictionary with 'total' and per-language counts
        """
        counts = {"total": 0}

        for file_path in files:
            try:
                file_info = self._extract_file_info(file_path)
                language = file_info.get("language", "unknown")

                if language not in counts:
                    counts[language] = 0

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)  # Validate it's valid JSON
                            counts[language] += 1
                            counts["total"] += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Failed to count conversations from {file_path}: {e}")
                continue

        return counts

    def get_dataset_statistics(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """Generate statistics for the dataset.

        Args:
            dataset_dict: DatasetDict to analyze

        Returns:
            Dictionary with statistics
        """
        from collections import Counter

        stats = {
            "total_languages": len(dataset_dict),
            "languages": list(dataset_dict.keys()),
            "total_messages": sum(len(ds) for ds in dataset_dict.values()),
            "language_stats": {}
        }

        for lang, dataset in dataset_dict.items():
            # Count messages per conversation in O(n) using Counter
            conversation_counts = Counter(dataset["conversation_id"])
            lengths = list(conversation_counts.values())

            stats["language_stats"][lang] = {
                "messages": len(dataset),
                "conversations": len(conversation_counts),
                "avg_messages_per_conversation": sum(lengths) / len(lengths) if lengths else 0,
                "min_conversation_length": min(lengths) if lengths else 0,
                "max_conversation_length": max(lengths) if lengths else 0,
                "unique_industries": len(set(dataset["industry"])) if "industry" in dataset.column_names else 0,
                "unique_problems": len(set(dataset["problem"])) if "problem" in dataset.column_names else 0,
            }

        return stats