"""Utility for validating PyPaCER electrode reconstruction JSON files."""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple
from datetime import datetime


class ElectrodeReconstructionValidator:
    """Validator for PyPaCER electrode reconstruction JSON files."""

    REQUIRED_FIELDS = {
        'metadata': ['timestamp', 'pypacer_version', 'ct_file', 'num_electrodes_detected'],
        'reconstruction_parameters': [],
        'seed_points': ['voxel', 'world'],
        'electrodes': []
    }

    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate an electrode reconstruction JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (is_valid, metadata, error_message)
            - is_valid: True if file is a valid electrode reconstruction
            - metadata: Dict with timestamp, pypacer_version, num_electrodes if valid
            - error_message: Error description if invalid
        """
        try:
            # Read and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate top-level structure
            for required_field in ElectrodeReconstructionValidator.REQUIRED_FIELDS.keys():
                if required_field not in data:
                    return False, None, f"Missing required field: '{required_field}'"

            # Validate metadata fields
            metadata = data.get('metadata', {})
            for field in ElectrodeReconstructionValidator.REQUIRED_FIELDS['metadata']:
                if field not in metadata:
                    return False, None, f"Missing metadata field: '{field}'"

            # Validate seed_points structure
            seed_points = data.get('seed_points', {})
            for field in ElectrodeReconstructionValidator.REQUIRED_FIELDS['seed_points']:
                if field not in seed_points:
                    return False, None, f"Missing seed_points field: '{field}'"

            # Validate electrodes is a list
            if not isinstance(data.get('electrodes'), list):
                return False, None, "Field 'electrodes' must be a list"

            # Extract metadata for display
            timestamp_str = metadata.get('timestamp', '')
            pypacer_version = metadata.get('pypacer_version', 'unknown')
            num_electrodes = metadata.get('num_electrodes_detected', 0)
            ct_file = metadata.get('ct_file', '')

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                formatted_date = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                formatted_date = timestamp_str

            result_metadata = {
                'timestamp': formatted_date,
                'timestamp_raw': timestamp_str,
                'pypacer_version': pypacer_version,
                'num_electrodes': num_electrodes,
                'ct_file': ct_file,
                'is_pypacer_reconstruction': True
            }

            return True, result_metadata, None

        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


def validate_electrode_reconstruction(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Convenience function to validate an electrode reconstruction JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (is_valid, metadata, error_message)
    """
    return ElectrodeReconstructionValidator.validate_file(file_path)
