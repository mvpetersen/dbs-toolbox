"""Utility for validating stereotactic frame fiducials JSON files."""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple
from datetime import datetime


class FrameFiducialsValidator:
    """Validator for stereotactic frame fiducials JSON files."""

    REQUIRED_FIELDS = {
        'fiducial_rods': [],
    }

    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a frame fiducials JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Tuple of (is_valid, metadata, error_message)
            - is_valid: True if file is a valid frame fiducials file
            - metadata: Dict with num_fiducials, detection_time if valid
            - error_message: Error description if invalid
        """
        try:
            # Read and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate top-level structure
            if 'fiducial_rods' not in data:
                return False, None, "Missing required field: 'fiducial_rods'"

            # Validate fiducial_rods is a list
            fiducial_rods = data.get('fiducial_rods', [])
            if not isinstance(fiducial_rods, list):
                return False, None, "Field 'fiducial_rods' must be a list"

            # Check that at least one fiducial has bottom_point and top_point
            valid_fiducials = 0
            for rod in fiducial_rods:
                if 'bottom_point' in rod and 'top_point' in rod:
                    # Validate they are lists of 3 numbers
                    bottom = rod['bottom_point']
                    top = rod['top_point']
                    if (isinstance(bottom, list) and len(bottom) == 3 and
                        isinstance(top, list) and len(top) == 3):
                        valid_fiducials += 1

            if valid_fiducials == 0:
                return False, None, "No valid fiducial rods found with bottom_point and top_point"

            # Extract metadata for display
            detection_time = data.get('detection_time', '')
            nifti_file = data.get('nifti_file', '')

            # Parse detection time
            formatted_date = 'unknown'
            if detection_time:
                try:
                    timestamp = datetime.fromisoformat(detection_time)
                    formatted_date = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, AttributeError):
                    formatted_date = detection_time

            result_metadata = {
                'num_fiducials': valid_fiducials,
                'detection_time': formatted_date,
                'detection_time_raw': detection_time,
                'nifti_file': nifti_file,
                'is_frame_fiducials': True
            }

            return True, result_metadata, None

        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


def validate_frame_fiducials(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Convenience function to validate a frame fiducials JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Tuple of (is_valid, metadata, error_message)
    """
    return FrameFiducialsValidator.validate_file(file_path)
