"""Utility for validating ANTs transform files."""

from pathlib import Path
import subprocess
from typing import Dict, Optional, Tuple


class AntsTransformValidator:
    """Validator for ANTs transform files (.mat and .nii.gz warp files)."""

    @staticmethod
    def validate_mat_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate an ANTs .mat transform file using antsTransformInfo.

        Args:
            file_path: Path to the .mat file

        Returns:
            Tuple of (is_valid, metadata, error_message)
            - is_valid: True if file is a valid ANTs transform
            - metadata: Dict with transform info if valid
            - error_message: Error description if invalid
        """
        try:
            # Run antsTransformInfo to validate the file
            result = subprocess.run(
                ['antsTransformInfo', str(file_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Check if command succeeded
            if result.returncode != 0:
                return False, None, f"Invalid ANTs transform file: {result.stderr.strip()}"

            # Parse output to extract transform type
            output = result.stdout

            # Extract transform information
            transform_type = 'unknown'
            if 'AffineTransform' in output:
                transform_type = 'Affine'
            elif 'MatrixOffsetTransformBase' in output:
                transform_type = 'Affine'
            elif 'Rigid' in output:
                transform_type = 'Rigid'
            elif 'Euler' in output:
                transform_type = 'Euler'
            elif 'Similarity' in output:
                transform_type = 'Similarity'

            # Check for essential transform components
            # ANTs transforms should have Matrix and Offset/Center/Translation
            has_matrix = 'Matrix:' in output or 'Matrix' in output
            has_offset = 'Offset:' in output or 'Offset' in output
            has_center = 'Center:' in output or 'Center' in output

            if not has_matrix:
                return False, None, "Invalid ANTs transform: missing transformation matrix"

            # Extract metadata
            metadata = {
                'is_ants_transform': True,
                'transform_type': transform_type,
                'file_type': 'mat',
                'invertible': True  # .mat files are invertible
            }

            return True, metadata, None

        except subprocess.TimeoutExpired:
            return False, None, "Validation timed out"
        except FileNotFoundError:
            return False, None, "antsTransformInfo not found. Please ensure ANTs is installed."
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"

    @staticmethod
    def validate_warp_file(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Validate a NIfTI warp field file (.nii.gz).

        Args:
            file_path: Path to the .nii.gz warp file

        Returns:
            Tuple of (is_valid, metadata, error_message)
        """
        try:
            # For warp fields, we need to check it's a valid NIfTI with proper dimensions
            # Using nibabel to validate the structure
            import nibabel as nib

            img = nib.load(str(file_path))
            shape = img.shape

            # Warp fields should have 4 or 5 dimensions
            # (x, y, z, vector_components) or (x, y, z, 1, vector_components)
            if len(shape) not in [4, 5]:
                return False, None, f"Invalid warp field dimensions: {shape}. Expected 4D or 5D."

            # Last dimension should be 3 (x, y, z displacement vectors)
            vector_dim = shape[-1]
            if vector_dim != 3:
                # Could also be shape[3] if it's 5D
                if len(shape) == 5 and shape[3] == 3:
                    vector_dim = 3
                else:
                    return False, None, f"Invalid vector dimension: {vector_dim}. Expected 3 (x,y,z displacement)."

            metadata = {
                'is_ants_transform': True,
                'transform_type': 'Nonlinear Warp',
                'file_type': 'nifti_warp',
                'shape': shape,
                'invertible': False  # Warp fields cannot be simply inverted
            }

            return True, metadata, None

        except Exception as e:
            return False, None, f"Failed to validate warp file: {str(e)}"


def validate_ants_transform(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Convenience function to validate an ANTs transform file.

    Args:
        file_path: Path to the transform file (.mat or .nii.gz)

    Returns:
        Tuple of (is_valid, metadata, error_message)
    """
    if str(file_path).endswith('.mat'):
        return AntsTransformValidator.validate_mat_file(file_path)
    elif str(file_path).endswith('.nii.gz') or str(file_path).endswith('.nii'):
        return AntsTransformValidator.validate_warp_file(file_path)
    else:
        return False, None, f"Unsupported file extension. Expected .mat or .nii/.nii.gz"
