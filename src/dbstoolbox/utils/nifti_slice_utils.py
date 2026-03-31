"""Utility functions for NIfTI slice generation and manipulation."""

from typing import Optional, List, Dict, Tuple
import numpy as np
from scipy.ndimage import map_coordinates


class NiftiSliceGenerator:
    """
    Utility class for generating resampled NIfTI slices perpendicular to surgical trajectories.

    This class provides methods to resample NIfTI volumes at specific depth planes
    along a surgical trajectory, oriented perpendicular to the trajectory direction.
    """

    SLICE_SIZE_MM = 50.0  # Size of the slice in mm
    RESOLUTION_MM = 0.2   # Resolution of resampled grid in mm

    @staticmethod
    def resample_slice(
        nifti_data: np.ndarray,
        affine: np.ndarray,
        plane_point: np.ndarray,
        anterior: np.ndarray,
        lateral: np.ndarray,
        slice_size_mm: float = SLICE_SIZE_MM,
        resolution_mm: float = RESOLUTION_MM
    ) -> Optional[np.ndarray]:
        """
        Resample a slice from the NIfTI volume at the given plane.

        Args:
            nifti_data: 3D or 4D numpy array (if 4D, uses first volume)
            affine: 4x4 affine transformation matrix (RAS to voxel)
            plane_point: 3D point at the center of the plane (in RAS coordinates)
            anterior: 3D unit vector pointing anterior
            lateral: 3D unit vector pointing lateral
            slice_size_mm: Size of the slice in mm (default: 50mm)
            resolution_mm: Resolution of resampled grid in mm (default: 0.2mm)

        Returns:
            2D numpy array with resampled slice values, or None if resampling fails
        """
        # Handle 4D data by taking first volume
        if nifti_data.ndim == 4:
            nifti_data = nifti_data[..., 0]

        # Create sampling grid in plane coordinates
        n_points = int(slice_size_mm / resolution_mm)
        half_size = slice_size_mm / 2.0

        # Grid coordinates from -25mm to +25mm in both directions
        grid_coords = np.linspace(-half_size, half_size, n_points)
        lat_grid, ant_grid = np.meshgrid(grid_coords, grid_coords)

        # Convert grid to RAS coordinates
        # Each point is: plane_point + lat_offset * lateral + ant_offset * anterior
        ras_points = np.zeros((n_points, n_points, 3))
        for i in range(n_points):
            for j in range(n_points):
                lat_offset = lat_grid[i, j]
                ant_offset = ant_grid[i, j]
                ras_points[i, j] = plane_point + lat_offset * lateral + ant_offset * anterior

        # Convert RAS to voxel coordinates using inverse affine
        inv_affine = np.linalg.inv(affine)

        # Apply inverse affine to all points
        ras_homogeneous = np.concatenate([
            ras_points.reshape(-1, 3),
            np.ones((n_points * n_points, 1))
        ], axis=1)

        voxel_coords = (inv_affine @ ras_homogeneous.T).T[:, :3]
        voxel_coords = voxel_coords.reshape(n_points, n_points, 3)

        # Sample the volume at these voxel coordinates
        slice_data = map_coordinates(
            nifti_data,
            [voxel_coords[..., 0].ravel(),  # X (i dimension)
             voxel_coords[..., 1].ravel(),  # Y (j dimension)
             voxel_coords[..., 2].ravel()], # Z (k dimension)
            order=1,  # Linear interpolation
            mode='constant',
            cval=0.0
        ).reshape(n_points, n_points)

        return slice_data

    @staticmethod
    def find_plane_intersection(
        trajectory: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find where a trajectory intersects a plane.

        Args:
            trajectory: Nx3 array of trajectory points
            plane_point: 3D point on the plane
            plane_normal: 3D normal vector of the plane

        Returns:
            3D intersection point or None if no intersection found
        """
        if len(trajectory) < 2:
            return None

        # Check each line segment
        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]
            line_dir = p2 - p1

            denom = np.dot(plane_normal, line_dir)
            if abs(denom) < 1e-10:
                continue

            t = np.dot(plane_normal, plane_point - p1) / denom
            if 0 <= t <= 1:
                return p1 + t * line_dir

        # Extrapolate if no segment intersection
        p1 = trajectory[0]
        p2 = trajectory[-1]
        line_dir = p2 - p1

        denom = np.dot(plane_normal, line_dir)
        if abs(denom) < 1e-10:
            return None

        t = np.dot(plane_normal, plane_point - p1) / denom
        return p1 + t * line_dir

    @staticmethod
    def project_to_plane_coords(
        point: np.ndarray,
        plane_origin: np.ndarray,
        anterior: np.ndarray,
        lateral: np.ndarray
    ) -> Tuple[float, float]:
        """
        Project a 3D point to 2D plane coordinates.

        Args:
            point: 3D point to project
            plane_origin: Origin of the plane coordinate system
            anterior: Anterior axis unit vector
            lateral: Lateral axis unit vector

        Returns:
            Tuple of (lateral_coord, anterior_coord)
        """
        diff = point - plane_origin
        anterior_coord = float(np.dot(diff, anterior))
        lateral_coord = float(np.dot(diff, lateral))
        return lateral_coord, anterior_coord
