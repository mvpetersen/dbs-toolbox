"""Utility for validating surgical data CSV files."""

from pathlib import Path
import csv
from typing import Dict, Optional, Tuple, List


class SurgicalDataValidator:
    """Validator for surgical data CSV files."""

    REQUIRED_COLUMNS = [
        'patient_id',
        'hemisphere',
        'x',
        'y',
        'z',
        'ring',
        'arc',
        'clinical_depth',
        'clinical_track',
        'research_depth',
        'research_track',
        'research_site',
        'surgeons',
        'researchers',
        'anatomical_target',
        'notes'
    ]

    COORDINATE_COLUMNS = ['x', 'y', 'z']
    NUMERIC_COLUMNS = ['x', 'y', 'z', 'ring', 'arc', 'clinical_depth', 'research_depth', 'research_site']

    @staticmethod
    def validate_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a surgical data CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Tuple of (is_valid, metadata, error_message)
            - is_valid: True if file is a valid surgical data CSV
            - metadata: Dict with num_records, columns, coordinate_range if valid
            - error_message: Error description if invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect dialect
                sample = f.read(1024)
                f.seek(0)

                # Read CSV
                reader = csv.DictReader(f)

                # Check if all required columns are present
                if not reader.fieldnames:
                    return False, None, "No columns found in CSV file"

                missing_columns = set(SurgicalDataValidator.REQUIRED_COLUMNS) - set(reader.fieldnames)
                if missing_columns:
                    return False, None, f"Missing required columns: {', '.join(missing_columns)}"

                # Read all rows and validate
                rows = list(reader)

                if len(rows) == 0:
                    return False, None, "CSV file contains no data rows"

                # Validate numeric columns
                for idx, row in enumerate(rows, start=2):  # Start at 2 (1 for header)
                    for col in SurgicalDataValidator.NUMERIC_COLUMNS:
                        if col in row and row[col].strip():  # Only validate if not empty
                            try:
                                float(row[col])
                            except ValueError:
                                return False, None, f"Invalid numeric value in column '{col}' at row {idx}: '{row[col]}'"

                # Validate hemisphere values
                for idx, row in enumerate(rows, start=2):
                    hemisphere = row.get('hemisphere', '').strip().lower()
                    if hemisphere and hemisphere not in ['left', 'right', 'bilateral', 'l', 'r']:
                        return False, None, f"Invalid hemisphere value at row {idx}: '{row['hemisphere']}' (expected: left, right, bilateral, l, or r)"

                # Calculate coordinate ranges
                x_coords = []
                y_coords = []
                z_coords = []

                for row in rows:
                    try:
                        if row.get('x', '').strip():
                            x_coords.append(float(row['x']))
                        if row.get('y', '').strip():
                            y_coords.append(float(row['y']))
                        if row.get('z', '').strip():
                            z_coords.append(float(row['z']))
                    except ValueError:
                        continue

                # Extract metadata
                metadata = {
                    'num_records': len(rows),
                    'columns': list(reader.fieldnames),
                    'num_columns': len(reader.fieldnames),
                    'is_surgical_data': True
                }

                if x_coords and y_coords and z_coords:
                    metadata['coordinate_ranges'] = {
                        'x': {'min': min(x_coords), 'max': max(x_coords)},
                        'y': {'min': min(y_coords), 'max': max(y_coords)},
                        'z': {'min': min(z_coords), 'max': max(z_coords)}
                    }
                    metadata['num_coordinates'] = min(len(x_coords), len(y_coords), len(z_coords))

                # Count unique patients
                patient_ids = set(row.get('patient_id', '').strip() for row in rows if row.get('patient_id', '').strip())
                metadata['num_patients'] = len(patient_ids)

                return True, metadata, None

        except csv.Error as e:
            return False, None, f"CSV parsing error: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"


def validate_surgical_csv(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Convenience function to validate a surgical data CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        Tuple of (is_valid, metadata, error_message)
    """
    return SurgicalDataValidator.validate_file(file_path)
