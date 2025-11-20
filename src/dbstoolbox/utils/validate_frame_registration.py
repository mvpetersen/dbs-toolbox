"""Utility for validating frame registration JSON files."""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple
from datetime import datetime


class FrameRegistrationValidator:
    """Validator for frame registration JSON files."""

    REQUIRED_FIELDS = {
        'nifti_file': str,
        'registration_time': str,
        'frame_type': str,
        'registration': dict
    }

    REGISTRATION_FIELDS = {
        'transformation_matrix': list,
        'rmse': (int, float),
        'success': bool,
        'message': str
    }

    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a frame registration JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (is_valid, metadata, error_message)
            - is_valid: True if file is a valid frame registration
            - metadata: Dict with registration info if valid
            - error_message: Error description if invalid
        """
        try:
            # Read and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate top-level structure
            for field, expected_type in FrameRegistrationValidator.REQUIRED_FIELDS.items():
                if field not in data:
                    return False, None, f"Missing required field: '{field}'"
                if not isinstance(data[field], expected_type):
                    return False, None, f"Field '{field}' must be of type {expected_type.__name__}"

            # Validate registration structure
            registration = data.get('registration', {})
            for field, expected_type in FrameRegistrationValidator.REGISTRATION_FIELDS.items():
                if field not in registration:
                    return False, None, f"Missing registration field: '{field}'"
                if isinstance(expected_type, tuple):
                    if not isinstance(registration[field], expected_type):
                        return False, None, f"Registration field '{field}' has invalid type"
                else:
                    if not isinstance(registration[field], expected_type):
                        return False, None, f"Registration field '{field}' must be of type {expected_type.__name__}"

            # Validate transformation matrix is 4x4
            transform_matrix = registration.get('transformation_matrix', [])
            if len(transform_matrix) != 4:
                return False, None, "Transformation matrix must have 4 rows"

            for i, row in enumerate(transform_matrix):
                if not isinstance(row, list):
                    return False, None, f"Transformation matrix row {i} must be a list"
                if len(row) != 4:
                    return False, None, f"Transformation matrix row {i} must have 4 columns"
                for j, val in enumerate(row):
                    if not isinstance(val, (int, float)):
                        return False, None, f"Transformation matrix element [{i}][{j}] must be numeric"

            # Parse timestamp
            timestamp_str = data.get('registration_time', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                formatted_date = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, AttributeError):
                formatted_date = timestamp_str

            # Extract metadata for display
            result_metadata = {
                'registration_time': formatted_date,
                'registration_time_raw': timestamp_str,
                'frame_type': data.get('frame_type', 'unknown'),
                'rmse': registration.get('rmse', 0.0),
                'success': registration.get('success', False),
                'message': registration.get('message', ''),
                'is_frame_registration': True,
                'matrix_4x4': transform_matrix
            }

            # Add registration errors if present
            if 'registration_errors' in data:
                errors = data['registration_errors']
                result_metadata['errors'] = {
                    'rmse': errors.get('rmse', 0.0),
                    'mean': errors.get('mean', 0.0),
                    'std': errors.get('std', 0.0),
                    'min': errors.get('min', 0.0),
                    'max': errors.get('max', 0.0)
                }

            return True, result_metadata, None

        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


def validate_frame_registration(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Convenience function to validate a frame registration JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (is_valid, metadata, error_message)
    """
    return FrameRegistrationValidator.validate_file(file_path)
