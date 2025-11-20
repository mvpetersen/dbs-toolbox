"""Utility for validating NIfTI files for visualization."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import nibabel as nib
import numpy as np


def validate_nifti(file_path: Path) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """
    Validate a NIfTI file for surface visualization.

    Expects 3D or 4D NIfTI files with data range [0, 1] for probability maps
    or segmentations.

    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz)

    Returns:
        Tuple of (is_valid, metadata, error_message)
        - is_valid: True if file is valid
        - metadata: Dict with file info if valid
        - error_message: Error description if invalid
    """
    try:
        import time
        t1 = time.time()
        # Load only the NIfTI header (fast - doesn't load data)
        img = nib.load(str(file_path))
        shape = img.shape
        ndim = len(shape)
        t2 = time.time()
        print(f"  [validate_nifti] Header load took {t2-t1:.3f}s")

        # Check dimensions
        if ndim not in [3, 4]:
            return False, None, f"Invalid dimensions: {shape}. Expected 3D or 4D NIfTI."

        # Build metadata from header only (no data loading)
        if ndim == 4:
            num_volumes = shape[3]

            metadata = {
                'shape': shape,
                'dimensions': ndim,
                'num_volumes': num_volumes,
                'data_range': None,  # Will be computed on first use if needed
                'affine': img.affine.tolist(),
                'voxel_size': img.header.get_zooms()[:3],
                'file_type': 'nifti_4d'
            }

        else:  # 3D
            metadata = {
                'shape': shape,
                'dimensions': ndim,
                'num_volumes': 1,
                'data_range': None,  # Will be computed on first use if needed
                'affine': img.affine.tolist(),
                'voxel_size': img.header.get_zooms(),
                'file_type': 'nifti_3d'
            }

        return True, metadata, None

    except FileNotFoundError:
        return False, None, f"File not found: {file_path}"
    except Exception as e:
        return False, None, f"Failed to validate NIfTI file: {str(e)}"


def load_nifti_for_visualization(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Load NIfTI file and prepare data for visualization.

    Args:
        file_path: Path to the NIfTI file

    Returns:
        Tuple of (data, affine, error_message)
        - data: 3D or 4D numpy array
        - affine: 4x4 affine transformation matrix
        - error_message: Error description if failed
    """
    try:
        import time
        t1 = time.time()
        img = nib.load(str(file_path))
        t2 = time.time()
        print(f"  [load_nifti] Header load took {t2-t1:.3f}s")

        data = img.get_fdata()
        t3 = time.time()
        print(f"  [load_nifti] Data load (get_fdata) took {t3-t2:.3f}s")

        affine = img.affine

        return data, affine, None

    except Exception as e:
        return None, None, f"Failed to load NIfTI file: {str(e)}"
