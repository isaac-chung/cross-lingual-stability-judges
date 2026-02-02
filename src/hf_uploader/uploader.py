"""
Core upload functionality for HuggingFace datasets.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time

from datasets import DatasetDict, load_dataset, concatenate_datasets

from .config import HFConfig, DATASET_CARD_TEMPLATE, SUPPORTED_LANGUAGES
from .auth import HFAuthenticator
from .dataset_builder import DatasetBuilder


logger = logging.getLogger(__name__)


class HFUploader:
    """Handles uploading conversation datasets to HuggingFace Hub."""

    def __init__(self, config: HFConfig, validate_credentials: bool = True):
        """Initialize uploader with configuration.

        Args:
            config: HFConfig instance with credentials and settings
            validate_credentials: Whether to validate credentials on initialization
        """
        self.config = config
        self.auth = HFAuthenticator(config.token, config.username)
        self.builder = DatasetBuilder()

        # Validate credentials on initialization (skip for dry-run)
        if validate_credentials and not self.auth.validate_credentials():
            raise ValueError("Invalid HuggingFace credentials")

    async def upload_by_model(
        self,
        model: str,
        languages: Optional[List[str]] = None,
        dry_run: bool = False,
        skip_card: bool = False
    ) -> Dict[str, str]:
        """Upload all conversations for a specific model.

        Args:
            model: Model name to filter by
            languages: Optional list of language codes to include
            dry_run: If True, only preview operations without uploading
            skip_card: If True, skip dataset card generation

        Returns:
            Dictionary mapping dataset names to their URLs

        Raises:
            ValueError: If no matching files found or upload fails
        """
        logger.info(f"Starting upload for model: {model}")

        # Find matching files
        files = self._find_files_by_model(model, languages)
        if not files:
            raise ValueError(f"No conversation files found for model: {model}")

        logger.info(f"Found {len(files)} files for model {model}")

        if dry_run:
            return await self._dry_run_preview(files, model, languages)

        # Group files by model (should all be the same model, but keeps structure consistent)
        grouped_files = self._group_files_by_model(files)

        results = {}
        for file_model, file_list in grouped_files.items():
            try:
                dataset_url = await self._upload_model_dataset(file_model, file_list, skip_card=skip_card)
                results[file_model] = dataset_url
            except Exception as e:
                logger.error(f"Failed to upload dataset for {file_model}: {e}")
                raise

        return results

    async def upload_files(
        self,
        file_paths: List[Path],
        dry_run: bool = False,
        skip_card: bool = False
    ) -> Dict[str, str]:
        """Upload specific conversation files.

        Args:
            file_paths: List of paths to conversation files
            dry_run: If True, only preview operations without uploading
            skip_card: If True, skip dataset card generation

        Returns:
            Dictionary mapping dataset names to their URLs

        Raises:
            ValueError: If no valid files or upload fails
        """
        logger.info(f"Starting upload for {len(file_paths)} specific files")

        # Validate files exist
        valid_files = []
        for path in file_paths:
            if path.exists():
                valid_files.append(path)
            else:
                logger.warning(f"File not found: {path}")

        if not valid_files:
            raise ValueError("No valid files found")

        if dry_run:
            # Group by model for preview
            grouped = self._group_files_by_model(valid_files)
            results = {}
            for model, files in grouped.items():
                preview_result = await self._dry_run_preview(files, model)
                results.update(preview_result)
            return results

        # Group files by model and upload each dataset
        grouped_files = self._group_files_by_model(valid_files)

        results = {}
        for model, files in grouped_files.items():
            try:
                dataset_url = await self._upload_model_dataset(model, files, skip_card=skip_card)
                results[model] = dataset_url
            except Exception as e:
                logger.error(f"Failed to upload dataset for {model}: {e}")
                raise

        return results

    def _find_files_by_model(
        self,
        model: str,
        languages: Optional[List[str]] = None
    ) -> List[Path]:
        """Find conversation files matching model and language criteria.

        Args:
            model: Model name to match
            languages: Optional list of language codes to include

        Returns:
            List of matching file paths
        """
        data_dir = Path("data")
        if not data_dir.exists():
            return []

        # Pattern: convo_{model}_{language}_{datetime}.jsonl
        files = []
        for file_path in data_dir.glob("convo_*.jsonl"):
            file_info = self.builder._extract_file_info(file_path)

            # Check model match
            if file_info["model"] != model:
                continue

            # Check language filter if specified
            if languages and file_info["language"] not in languages:
                continue

            files.append(file_path)

        return sorted(files)

    def _group_files_by_model(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by model name.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping model names to file lists
        """
        grouped = defaultdict(list)

        for file_path in files:
            file_info = self.builder._extract_file_info(file_path)
            model = file_info["model"]
            grouped[model].append(file_path)

        return dict(grouped)

    async def _upload_model_dataset(self, model: str, files: List[Path], skip_card: bool = False) -> str:
        """Upload dataset for a specific model.

        Args:
            model: Model name
            files: List of conversation files for this model
            skip_card: If True, skip dataset card generation

        Returns:
            Dataset URL

        Raises:
            RuntimeError: If upload fails
        """
        dataset_name = self.config.get_dataset_name(model)
        full_dataset_name = f"{self.config.username}/{dataset_name}"

        logger.info(f"Creating dataset: {full_dataset_name}")

        # Build new dataset from files
        try:
            logger.info("Building dataset from conversation files...")
            new_dataset_dict = self.builder.build_from_files(files)

            # Get languages from new dataset
            new_languages = list(new_dataset_dict.keys())
            total_convos = sum(len(set(ds["conversation_id"])) for ds in new_dataset_dict.values())
            logger.info(f"Built new dataset with {len(new_languages)} language subsets: {new_languages}")
            logger.info(f"Total conversations: {total_convos:,}")
            for lang, ds in new_dataset_dict.items():
                logger.info(f"  {lang}: {len(set(ds['conversation_id'])):,} conversations")

        except Exception as e:
            raise RuntimeError(f"Failed to build dataset: {e}")

        # Check if dataset exists and handle merging/overwriting
        dataset_exists = self.auth.check_dataset_exists(dataset_name)

        if dataset_exists:
            if self.config.overwrite:
                logger.info(f"Dataset exists and will be overwritten: {full_dataset_name}")
                final_dataset_dict = new_dataset_dict
                all_languages = new_languages
            else:
                logger.info(f"Dataset exists, merging with new conversations: {full_dataset_name}")
                final_dataset_dict = await self._merge_with_existing_dataset(
                    full_dataset_name, new_dataset_dict
                )
                all_languages = list(final_dataset_dict.keys())
                logger.info(f"Merged dataset has {len(all_languages)} language subsets: {all_languages}")
        else:
            logger.info(f"Creating new dataset: {full_dataset_name}")
            final_dataset_dict = new_dataset_dict
            all_languages = new_languages

        # Create or update dataset repository
        try:
            if not dataset_exists:
                self.auth.create_dataset_repo(
                    dataset_name=dataset_name,
                    private=self.config.private,
                    description=self.config.get_description(model, all_languages)
                )

            # Upload dataset with progress tracking
            logger.info("Uploading dataset to HuggingFace Hub...")
            await self._upload_with_progress(final_dataset_dict, full_dataset_name)

            # Create and upload dataset card (unless skipped)
            if skip_card:
                logger.info("Skipping dataset card generation (--skip-card)")
            else:
                await self._create_dataset_card(
                    full_dataset_name, model, all_languages, final_dataset_dict
                )

            dataset_url = f"https://huggingface.co/datasets/{full_dataset_name}"
            logger.info(f"Successfully uploaded dataset: {dataset_url}")

            return dataset_url

        except Exception as e:
            raise RuntimeError(f"Failed to upload dataset: {e}")

    async def _merge_with_existing_dataset(
        self,
        full_dataset_name: str,
        new_dataset_dict: DatasetDict
    ) -> DatasetDict:
        """Merge new conversations with existing dataset.

        Args:
            full_dataset_name: Full dataset name (username/dataset_name)
            new_dataset_dict: New dataset to merge

        Returns:
            Merged DatasetDict

        Raises:
            RuntimeError: If merging fails
        """
        try:
            logger.info("Loading existing dataset for merging...")
            existing_dataset_dict = load_dataset(full_dataset_name, token=self.config.token)

            merged_dict = {}

            # Process each language in the new dataset
            for language, new_dataset in new_dataset_dict.items():
                if language in existing_dataset_dict:
                    # Language exists - merge conversations
                    existing_dataset = existing_dataset_dict[language]

                    # Check for duplicate conversation IDs
                    existing_conv_ids = set(existing_dataset["conversation_id"])
                    new_conv_ids = set(new_dataset["conversation_id"])

                    duplicate_ids = existing_conv_ids.intersection(new_conv_ids)
                    if duplicate_ids:
                        logger.warning(f"Found {len(duplicate_ids)} duplicate conversation IDs in {language} subset:")
                        for dup_id in sorted(list(duplicate_ids)[:5]):  # Show first 5
                            logger.warning(f"  - {dup_id}")
                        if len(duplicate_ids) > 5:
                            logger.warning(f"  ... and {len(duplicate_ids) - 5} more")
                        logger.info("Skipping duplicate conversations (keeping existing ones)")

                        # Filter out duplicate conversations from new dataset
                        filtered_indices = [
                            i for i, conv_id in enumerate(new_dataset["conversation_id"])
                            if conv_id not in existing_conv_ids
                        ]

                        if filtered_indices:
                            filtered_new_dataset = new_dataset.select(filtered_indices)
                            merged_dict[language] = concatenate_datasets([existing_dataset, filtered_new_dataset])
                            logger.info(f"Merged {language} subset: {len(existing_dataset)} existing + {len(filtered_new_dataset)} new = {len(merged_dict[language])} total messages")
                        else:
                            logger.info(f"No new conversations to add for {language} subset")
                            merged_dict[language] = existing_dataset
                    else:
                        # No duplicates - safe to merge
                        merged_dict[language] = concatenate_datasets([existing_dataset, new_dataset])
                        logger.info(f"Merged {language} subset: {len(existing_dataset)} existing + {len(new_dataset)} new = {len(merged_dict[language])} total messages")
                else:
                    # New language - add as new subset
                    merged_dict[language] = new_dataset
                    logger.info(f"Added new {language} subset: {len(new_dataset)} messages")

            # Add any existing languages not present in new dataset
            for language, existing_dataset in existing_dataset_dict.items():
                if language not in merged_dict:
                    merged_dict[language] = existing_dataset
                    logger.info(f"Preserved existing {language} subset: {len(existing_dataset)} messages")

            return DatasetDict(merged_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to merge with existing dataset: {e}")

    async def _upload_with_progress(
        self,
        dataset_dict: DatasetDict,
        full_dataset_name: str
    ) -> None:
        """Upload dataset with progress tracking.

        Args:
            dataset_dict: DatasetDict to upload
            full_dataset_name: Full dataset name (username/dataset)
        """
        def upload_task():
            dataset_dict.push_to_hub(
                full_dataset_name,
                token=self.config.token,
                private=self.config.private
            )

        # Run upload in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        upload_future = loop.run_in_executor(None, upload_task)
        await upload_future
        logger.info("Dataset upload complete")

    async def _create_dataset_card(
        self,
        full_dataset_name: str,
        model: str,
        languages: List[str],
        dataset_dict: DatasetDict
    ) -> None:
        """Create and upload dataset card.

        Args:
            full_dataset_name: Full dataset name
            model: Model name
            languages: List of language codes
            dataset_dict: DatasetDict for statistics
        """
        try:
            # Generate statistics
            stats = self.builder.get_dataset_statistics(dataset_dict)

            # Language mappings for better display
            language_names = {
                "et": "Estonian",
                "fi": "Finnish",
                "hu": "Hungarian",
                "en-us": "English (US)"
            }

            # Calculate totals
            total_conversations = sum(s['conversations'] for s in stats['language_stats'].values())
            total_messages = stats['total_messages']

            # Format YAML language list for frontmatter
            yaml_languages = "\n".join([f"- {lang}" for lang in languages])

            # Format human-readable language list
            language_display_names = [language_names.get(lang, lang.title()) for lang in languages]
            language_list = ", ".join(language_display_names)

            # Format detailed language information
            language_info_sections = []
            for lang in languages:
                lang_stats = stats["language_stats"][lang]
                lang_name = language_names.get(lang, lang.title())

                # Calculate average conversation length
                avg_length = lang_stats.get('avg_messages_per_conversation', 0)
                min_length = lang_stats.get('min_conversation_length', 0)
                max_length = lang_stats.get('max_conversation_length', 0)

                lang_section = f"""### {lang_name} ({lang})

- **Messages**: {lang_stats['messages']:,}
- **Conversations**: {lang_stats['conversations']:,}
- **Avg. conversation length**: {avg_length:.1f} messages
- **Conversation length range**: {min_length}-{max_length} messages
- **Industries covered**: {lang_stats.get('unique_industries', 'N/A')}
- **Problem types**: {lang_stats.get('unique_problems', 'N/A')}"""

                language_info_sections.append(lang_section)

            # Determine size category based on total messages
            if total_messages < 1000:
                size_category = "n<1K"
            elif total_messages < 10000:
                size_category = "1K<n<10K"
            elif total_messages < 100000:
                size_category = "10K<n<100K"
            elif total_messages < 1000000:
                size_category = "100K<n<1M"
            else:
                size_category = "n>1M"

            # Create clean model tag
            model_tag = model.replace("/", "-").replace("_", "-").replace(".", "-").lower()

            # Format detailed statistics
            statistics_sections = []

            # Overall statistics
            statistics_sections.append(f"""### Dataset Overview

| Metric | Value |
|--------|--------|
| **Total Languages** | {stats['total_languages']} |
| **Total Messages** | {total_messages:,} |
| **Total Conversations** | {total_conversations:,} |
| **Average Messages per Conversation** | {total_messages/total_conversations:.1f} |""")

            # Per-language breakdown
            lang_table_rows = []
            for lang in languages:
                lang_stats = stats["language_stats"][lang]
                lang_name = language_names.get(lang, lang.title())
                avg_length = lang_stats.get('avg_messages_per_conversation', 0)

                lang_table_rows.append(
                    f"| {lang_name} ({lang}) | {lang_stats['messages']:,} | "
                    f"{lang_stats['conversations']:,} | {avg_length:.1f} |"
                )

            statistics_sections.append(f"""### Per-Language Statistics

| Language | Messages | Conversations | Avg. Length |
|----------|----------|---------------|-------------|
{chr(10).join(lang_table_rows)}""")

            # Create dataset card content
            dataset_card = DATASET_CARD_TEMPLATE.format(
                # YAML frontmatter variables
                yaml_languages=yaml_languages,
                model_tag=model_tag,
                size_category=size_category,

                # Content variables
                model=model,
                language_list=language_list,
                total_messages=total_messages,
                total_conversations=total_conversations,
                language_info="\n".join(language_info_sections),
                full_dataset_name=full_dataset_name,
                statistics="\n".join(statistics_sections),
                clean_model=model.replace(".", "_").replace("-", "_").replace("/", "_"),
                username=self.config.username
            )

            # Upload dataset card
            from huggingface_hub import HfApi
            api = HfApi(token=self.config.token)

            api.upload_file(
                path_or_fileobj=dataset_card.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=full_dataset_name,
                repo_type="dataset"
            )

            logger.info("Dataset card created and uploaded")

        except Exception as e:
            logger.warning(f"Failed to create dataset card: {e}")

    async def _dry_run_preview(
        self,
        files: List[Path],
        model: str,
        languages: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Preview what would be uploaded in dry run mode.

        Args:
            files: List of files that would be processed
            model: Model name
            languages: Optional language filter

        Returns:
            Dictionary with preview information
        """
        logger.info(f"=== DRY RUN PREVIEW ===")
        logger.info(f"Model: {model}")

        dataset_name = self.config.get_dataset_name(model)
        full_dataset_name = f"{self.config.username}/{dataset_name}"

        # Analyze files without full processing
        file_info = {}
        for file_path in files:
            info = self.builder._extract_file_info(file_path)
            lang = info["language"]
            if lang not in file_info:
                file_info[lang] = []
            file_info[lang].append(str(file_path))

        logger.info(f"Would create dataset: {full_dataset_name}")
        logger.info(f"Dataset would be {'private' if self.config.private else 'public'}")
        logger.info(f"Languages: {list(file_info.keys())}")
        logger.info(f"Files per language:")

        for lang, lang_files in file_info.items():
            logger.info(f"  {lang}: {len(lang_files)} files")
            for file_path in lang_files[:3]:  # Show first 3 files
                logger.info(f"    - {file_path}")
            if len(lang_files) > 3:
                logger.info(f"    ... and {len(lang_files) - 3} more")

        # Count conversations from files
        conv_counts = self.builder.count_conversations_from_files(files)
        logger.info(f"Total conversations: {conv_counts['total']:,}")
        for lang in file_info.keys():
            if lang in conv_counts:
                logger.info(f"  {lang}: {conv_counts[lang]:,} conversations")

        # Check if dataset exists
        exists = self.auth.check_dataset_exists(dataset_name)
        if exists:
            logger.info(f"⚠️  Dataset already exists: {full_dataset_name}")
            if self.config.overwrite:
                logger.info("✅ Would overwrite existing dataset (--overwrite specified)")
            else:
                logger.info("✅ Would merge new conversations with existing dataset")
        else:
            logger.info("✅ Would create new dataset")

        logger.info(f"Dataset URL would be: https://huggingface.co/datasets/{full_dataset_name}")

        return {model: f"DRY_RUN: https://huggingface.co/datasets/{full_dataset_name}"}

    def _generate_dataset_name(self, model: str) -> str:
        """Generate dataset name for a given model.

        Args:
            model: Model name

        Returns:
            Clean dataset name
        """
        return self.config.get_dataset_name(model)

    def _generate_description(self, model: str, languages: List[str]) -> str:
        """Generate dataset description.

        Args:
            model: Model name
            languages: List of language codes

        Returns:
            Dataset description
        """
        return self.config.get_description(model, languages)