"""
Authentication module for HuggingFace Hub operations.
"""

import logging
from typing import Dict, Any, Optional
from huggingface_hub import HfApi, HfFolder, whoami
from huggingface_hub.utils import HfHubHTTPError


logger = logging.getLogger(__name__)


class HFAuthenticator:
    """Handles HuggingFace Hub authentication and dataset operations."""

    def __init__(self, token: str, username: str):
        """Initialize authenticator with credentials.

        Args:
            token: HuggingFace API token
            username: HuggingFace username
        """
        self.token = token
        self.username = username
        self.api = HfApi(token=token)

    def validate_credentials(self) -> bool:
        """Validate HuggingFace credentials.

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            user_info = whoami(token=self.token)
            if user_info.get("name") != self.username:
                logger.warning(
                    f"Username mismatch: expected '{self.username}', "
                    f"got '{user_info.get('name')}'"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False

    def get_user_info(self) -> Dict[str, Any]:
        """Get user information from HuggingFace.

        Returns:
            User information dictionary

        Raises:
            RuntimeError: If unable to retrieve user info
        """
        try:
            return whoami(token=self.token)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve user info: {e}")

    def check_dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset already exists.

        Args:
            dataset_name: Name of the dataset to check

        Returns:
            True if dataset exists, False otherwise
        """
        try:
            full_dataset_name = f"{self.username}/{dataset_name}"
            self.api.dataset_info(full_dataset_name)
            return True
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                return False
            raise
        except Exception as e:
            logger.warning(f"Error checking dataset existence: {e}")
            return False

    def create_dataset_repo(
        self,
        dataset_name: str,
        private: bool = False,
        description: Optional[str] = None,
        license: str = "mit"
    ) -> str:
        """Create a new dataset repository on HuggingFace Hub.

        Args:
            dataset_name: Name of the dataset
            private: Whether to make the dataset private
            description: Dataset description
            license: Dataset license

        Returns:
            Full dataset name (username/dataset_name)

        Raises:
            RuntimeError: If dataset creation fails
        """
        try:
            full_dataset_name = f"{self.username}/{dataset_name}"

            self.api.create_repo(
                repo_id=full_dataset_name,
                repo_type="dataset",
                private=private
            )

            logger.info(f"Created dataset repository: {full_dataset_name}")
            return full_dataset_name

        except Exception as e:
            raise RuntimeError(f"Failed to create dataset repository: {e}")

    def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the dataset to delete

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            full_dataset_name = f"{self.username}/{dataset_name}"
            self.api.delete_repo(
                repo_id=full_dataset_name,
                repo_type="dataset"
            )
            logger.info(f"Deleted dataset: {full_dataset_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to delete dataset: {e}")

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about an existing dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information

        Raises:
            RuntimeError: If unable to retrieve dataset info
        """
        try:
            full_dataset_name = f"{self.username}/{dataset_name}"
            info = self.api.dataset_info(full_dataset_name)
            return {
                "id": info.id,
                "sha": info.sha,
                "tags": info.tags,
                "private": info.private,
                "downloads": info.downloads,
                "created_at": info.created_at,
                "last_modified": info.last_modified
            }
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve dataset info: {e}")