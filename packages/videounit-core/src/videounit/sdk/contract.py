"""Contract loading and validation utilities."""

from pathlib import Path
from typing import Union

import yaml

from videounit.sdk.models import VideoContract
from videounit.core.errors import ContractError


class ContractLoader:
    """Utility class for loading and validating VideoUnit contracts."""

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> VideoContract:
        """Load a VideoContract from a YAML file.

        Args:
            path: Path to the YAML contract file.

        Returns:
            VideoContract instance.

        Raises:
            ContractError: If the file cannot be loaded or parsed.
        """
        path = Path(path)
        if not path.exists():
            raise ContractError(f"Contract file not found: {path}")

        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise ContractError(f"Failed to parse YAML: {e}")

        return ContractLoader.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> VideoContract:
        """Create a VideoContract from a dictionary.

        Args:
            data: Dictionary containing contract data.

        Returns:
            VideoContract instance.

        Raises:
            ContractError: If the data is invalid.
        """
        try:
            return VideoContract(**data)
        except Exception as e:
            raise ContractError(f"Invalid contract data: {e}")

    @staticmethod
    def from_string(yaml_string: str) -> VideoContract:
        """Parse a VideoContract from a YAML string.

        Args:
            yaml_string: YAML string containing contract data.

        Returns:
            VideoContract instance.
        """
        try:
            data = yaml.safe_load(yaml_string)
            return ContractLoader.from_dict(data)
        except yaml.YAMLError as e:
            raise ContractError(f"Failed to parse YAML string: {e}")


def validate_contract(contract: VideoContract) -> list[str]:
    """Validate a VideoContract and return list of warnings.

    Args:
        contract: The contract to validate.

    Returns:
        List of validation warning messages (empty if valid).
    """
    warnings = []

    if not contract.test.name:
        warnings.append("Contract test name is empty")

    if not contract.input.source:
        warnings.append("Contract input source is empty")

    object_ids = {obj.id for obj in contract.objects}
    for assertion in contract.assertions:
        if assertion.object and assertion.object not in object_ids:
            warnings.append(
                f"Assertion references unknown object: {assertion.object}"
            )
        if assertion.from_ and assertion.from_ not in object_ids:
            warnings.append(
                f"Assertion references unknown source object: {assertion.from_}"
            )
        if assertion.to and assertion.to not in object_ids:
            warnings.append(
                f"Assertion references unknown target object: {assertion.to}"
            )

    if contract.scoring:
        for category in contract.scoring.categories:
            if category not in contract.scoring.weights:
                warnings.append(
                    f"Scoring category '{category}' has no weight defined"
                )

    return warnings
