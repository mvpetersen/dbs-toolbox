"""
Coordinate transformation utilities for DBS Toolbox.

This module provides functions for transforming coordinates between different
coordinate systems using ANTs transformations and frame registration matrices.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import json
import csv


def ras_to_lps(points: np.ndarray) -> np.ndarray:
    """
    Convert points from RAS to LPS coordinate system.

    RAS (Right-Anterior-Superior) is used by NIfTI and neuroimaging software.
    LPS (Left-Posterior-Superior) is used by ITK and ANTs.

    Args:
        points: Nx3 array of points in RAS coordinates

    Returns:
        Nx3 array of points in LPS coordinates
    """
    points = np.atleast_2d(points)
    points_lps = points.copy()
    points_lps[:, 0] = -points[:, 0]  # R -> L (flip X)
    points_lps[:, 1] = -points[:, 1]  # A -> P (flip Y)
    # Z stays the same (S -> S)
    return points_lps


def lps_to_ras(points: np.ndarray) -> np.ndarray:
    """
    Convert points from LPS to RAS coordinate system.

    Args:
        points: Nx3 array of points in LPS coordinates

    Returns:
        Nx3 array of points in RAS coordinates
    """
    points = np.atleast_2d(points)
    points_ras = points.copy()
    points_ras[:, 0] = -points[:, 0]  # L -> R (flip X)
    points_ras[:, 1] = -points[:, 1]  # P -> A (flip Y)
    # Z stays the same (S -> S)
    return points_ras


def apply_ants_transforms_to_points(
    points: np.ndarray,
    transform_files: List[Path],
    use_inverse: Union[bool, List[bool]] = True,
    input_coordinate_system: str = "RAS"
) -> np.ndarray:
    """
    Apply ANTs transformations to points with automatic coordinate system handling.

    Args:
        points: Nx3 array of points in physical coordinates
        transform_files: List of paths to ANTs transform files (.mat or .nii.gz)
        use_inverse: Whether to use inverse transform. Can be:
            - bool: Apply same setting to all transforms (default True for points)
            - list of bool: Per-transform inverse flags (must match length of transform_files)
        input_coordinate_system: 'RAS' or 'LPS' (default 'RAS')

    Returns:
        Nx3 array of transformed points in the same coordinate system as input

    Raises:
        ImportError: If antspyx or pandas is not installed
        ValueError: If points are not Nx3 or coordinate system is invalid
    """
    try:
        import ants
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            f"Missing required package: {e}. "
            "Install with: pip install antspyx pandas"
        )

    # Validate inputs
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got shape {points.shape}")

    if input_coordinate_system not in ["RAS", "LPS"]:
        raise ValueError(
            f"Coordinate system must be 'RAS' or 'LPS', got {input_coordinate_system}"
        )

    # Convert to LPS if input is RAS (ANTs expects LPS)
    if input_coordinate_system == "RAS":
        points_lps = ras_to_lps(points)
    else:
        points_lps = points.copy()

    # Create DataFrame for ANTs
    points_df = pd.DataFrame(points_lps, columns=["x", "y", "z"])

    # Convert transform files to strings
    transform_files = [Path(f) for f in transform_files]
    transforms_str = [str(f) for f in transform_files]

    # Validate transform files exist
    for tf in transform_files:
        if not tf.exists():
            raise FileNotFoundError(f"Transform file not found: {tf}")

    # Handle use_inverse parameter - convert to list if needed
    if isinstance(use_inverse, bool):
        which_to_invert = [use_inverse] * len(transforms_str)
    elif isinstance(use_inverse, (list, tuple)):
        if len(use_inverse) != len(transforms_str):
            raise ValueError(
                f"Length of use_inverse ({len(use_inverse)}) must match "
                f"number of transforms ({len(transforms_str)})"
            )
        which_to_invert = list(use_inverse)
    else:
        raise TypeError(
            f"use_inverse must be bool or list of bool, got {type(use_inverse)}"
        )

    # Auto-detect deformation fields and disable inversion
    for i, tf in enumerate(transform_files):
        tf_lower = str(tf).lower()
        is_deformation = any(
            tf_lower.endswith(ext) for ext in [".nii", ".nii.gz"]
        )

        if is_deformation and which_to_invert[i]:
            import warnings
            warnings.warn(
                f"Transform {i} ({tf.name}) is a deformation field. "
                f"Using forward direction (deformation fields cannot be inverted).",
                UserWarning,
            )
            which_to_invert[i] = False

    # Apply transforms (applied in REVERSE order - ANTs convention)
    transformed_df = ants.apply_transforms_to_points(
        dim=3,
        points=points_df,
        transformlist=transforms_str,
        whichtoinvert=which_to_invert,
    )

    # Convert back to numpy array
    transformed_points_lps = transformed_df[["x", "y", "z"]].values

    # Convert back to original coordinate system if needed
    if input_coordinate_system == "RAS":
        transformed_points = lps_to_ras(transformed_points_lps)
    else:
        transformed_points = transformed_points_lps

    return transformed_points


def apply_4x4_matrix_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 transformation matrix to points.

    Args:
        points: Nx3 array of points
        matrix: 4x4 transformation matrix

    Returns:
        Nx3 array of transformed points
    """
    points = np.atleast_2d(points)

    # Convert to homogeneous coordinates (Nx4)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])

    # Apply transformation
    transformed_homogeneous = points_homogeneous @ matrix.T

    # Convert back to Nx3
    transformed_points = transformed_homogeneous[:, :3]

    return transformed_points


def transform_pypacer_reconstruction(
    reconstruction_data: Dict,
    transform_files: List[Path],
    invert_flags: List[bool],
    transform_types: List[str]
) -> Tuple[Dict, int]:
    """
    Transform all electrode contacts in a PyPaCER reconstruction.

    PyPaCER reconstructions are already in CT space, so only ANTs transforms
    are applied (frame registration transforms are ignored with a warning).

    Args:
        reconstruction_data: PyPaCER reconstruction dictionary
        transform_files: List of transform file paths
        invert_flags: List of booleans for each transform (True = invert, False = forward)
        transform_types: List of transform types

    Returns:
        Tuple of (updated reconstruction data, total_contacts_transformed)
    """
    # Filter out frame registration transforms (not applicable to PyPaCER data)
    ants_transforms = []
    ants_invert_flags = []

    for tf, invert, tf_type in zip(transform_files, invert_flags, transform_types):
        if tf_type == 'frame_registration':
            import warnings
            warnings.warn(
                f"Frame registration transform {tf.name} ignored for PyPaCER data "
                "(PyPaCER coordinates are already in CT space)",
                UserWarning
            )
        else:
            ants_transforms.append(tf)
            ants_invert_flags.append(invert)

    if not ants_transforms:
        raise ValueError("No valid ANTs transforms provided for PyPaCER data")

    total_contacts = 0

    for electrode in reconstruction_data.get("electrodes", []):
        # Transform contact positions
        if "contact_positions_3d" in electrode:
            contacts = np.array(electrode["contact_positions_3d"])
            transformed = apply_ants_transforms_to_points(
                contacts, ants_transforms, use_inverse=ants_invert_flags
            )

            # Store original and transformed
            electrode["contact_positions_3d_original"] = electrode["contact_positions_3d"]
            electrode["contact_positions_3d"] = transformed.tolist()
            total_contacts += len(contacts)

        # Transform tip position
        if "tip_position" in electrode:
            tip = np.array(electrode["tip_position"]).reshape(1, 3)
            transformed_tip = apply_ants_transforms_to_points(
                tip, ants_transforms, use_inverse=ants_invert_flags
            )
            electrode["tip_position_original"] = electrode["tip_position"]
            electrode["tip_position"] = transformed_tip[0].tolist()

        # Transform entry position
        if "entry_position" in electrode:
            entry = np.array(electrode["entry_position"]).reshape(1, 3)
            transformed_entry = apply_ants_transforms_to_points(
                entry, ants_transforms, use_inverse=ants_invert_flags
            )
            electrode["entry_position_original"] = electrode["entry_position"]
            electrode["entry_position"] = transformed_entry[0].tolist()

        # Transform trajectory coordinates
        if "trajectory_coordinates" in electrode:
            trajectory = np.array(electrode["trajectory_coordinates"])
            transformed_trajectory = apply_ants_transforms_to_points(
                trajectory, ants_transforms, use_inverse=ants_invert_flags
            )
            electrode["trajectory_coordinates_original"] = electrode["trajectory_coordinates"]
            electrode["trajectory_coordinates"] = transformed_trajectory.tolist()

    # Update metadata
    if "metadata" not in reconstruction_data:
        reconstruction_data["metadata"] = {}

    reconstruction_data["metadata"]["transformed"] = True
    reconstruction_data["metadata"]["transform_files"] = [str(f) for f in transform_files]
    reconstruction_data["metadata"]["transform_inverted"] = invert_flags

    return reconstruction_data, total_contacts


def calculate_direction_from_angles(ring_angle: float, arc_angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a 3D direction vector from stereotactic ring and arc angles.

    This uses the center-of-arc principle for the Leksell stereotactic frame:

    Coordinate system (origin at UPPER, POSTERIOR, RIGHT corner):
    - X-axis increases towards LEFT (positive = left)
    - Y-axis increases towards FRONT (positive = anterior)
    - Z-axis increases DOWNWARDS (positive = down)

    Process:
    1. Base direction when arc=0°: (-1, 0, 0) pointing RIGHT
    2. Arc rotation axis rotates with ring angle:
       - Starts as (0, 0, 1) pointing DOWN
       - Rotates around X-axis by ring angle
    3. Rotate base direction around the rotated arc axis by arc angle

    Examples:
    - ring=0°, arc=0° → (-1, 0, 0) RIGHT
    - ring=0°, arc=90° → (0, 1, 0) FORWARD
    - ring=90°, arc=0° → (-1, 0, 0) RIGHT (arc has no effect)
    - ring=90°, arc=90° → (0, 0, -1) UP

    Args:
        ring_angle: Ring angle in degrees
        arc_angle: Arc angle in degrees

    Returns:
        Tuple of (trajectory_direction, arc_axis) - both normalized 3D vectors
        The arc_axis represents the anterior direction in the rotated frame
    """
    # IMPORTANT: Negate angles to convert from right-hand rule to Leksell convention
    # The Leksell frame's mechanical rotation directions are opposite to the
    # mathematical right-hand rule used in standard rotation matrices.
    # Without negation:
    #   - ring=90° would rotate arc axis backward instead of forward
    #   - arc=90° would rotate base direction backward instead of forward
    ring_rad = -np.radians(ring_angle)
    arc_rad = -np.radians(arc_angle)

    # Base direction: pointing RIGHT
    base_direction = np.array([-1.0, 0.0, 0.0])

    # Arc rotation axis: starts pointing DOWN
    arc_axis = np.array([0.0, 0.0, 1.0])

    # Rotate arc axis by ring angle around X-axis
    ring_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(ring_rad), -np.sin(ring_rad)],
        [0, np.sin(ring_rad), np.cos(ring_rad)]
    ])
    arc_axis = ring_rotation @ arc_axis
    arc_axis = arc_axis / np.linalg.norm(arc_axis)

    # Rotate base_direction around arc_axis by arc_angle using Rodriguez formula
    # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    k = arc_axis
    v = base_direction

    direction = (v * np.cos(arc_rad) +
                 np.cross(k, v) * np.sin(arc_rad) +
                 k * np.dot(k, v) * (1 - np.cos(arc_rad)))

    direction = direction / np.linalg.norm(direction)

    return direction, arc_axis


def apply_frame_registration_transform(
    points: np.ndarray,
    transform_file: Path,
    invert: bool = False
) -> np.ndarray:
    """
    Apply a frame registration 4x4 transformation matrix to points.

    Args:
        points: Nx3 array of points in frame space
        transform_file: Path to frame registration JSON file
        invert: Whether to invert the transformation

    Returns:
        Nx3 array of transformed points
    """
    # Load frame registration JSON
    with open(transform_file, 'r') as f:
        data = json.load(f)

    # Extract 4x4 transformation matrix
    matrix = np.array(data['registration']['transformation_matrix'])

    # Invert if requested
    if invert:
        matrix = np.linalg.inv(matrix)

    # Apply transformation
    transformed_points = apply_4x4_matrix_transform(points, matrix)

    return transformed_points


def calculate_parallel_track_offset(
    track_type: str,
    target_x: float,
    trajectory_direction: np.ndarray,
    arc_axis: np.ndarray,
    offset_mm: float = 2.0
) -> np.ndarray:
    """
    Calculate the 3D offset for a parallel MER track perpendicular to the trajectory.

    For the Ben Gun microdrive system, parallel tracks are offset perpendicular to
    the surgical trajectory using the arc axis as the anterior reference direction.

    The arc axis rotates with the ring angle and always points "forward" relative
    to the rotating frame, making it the correct reference for anterior/posterior.
    The medial/lateral direction is calculated as the cross product of the trajectory
    and arc axis vectors.

    Coordinate system (origin at UPPER, POSTERIOR, RIGHT corner):
    - X-axis increases towards LEFT (x=100 is center)
    - Y-axis increases towards FRONT (positive = anterior)
    - Z-axis increases DOWNWARDS

    Args:
        track_type: Type of track ("central", "anterior", "posterior", "medial", "lateral")
        target_x: X coordinate of target (to determine medial/lateral hemisphere)
        trajectory_direction: Normalized trajectory direction vector (from calculate_direction_from_angles)
        arc_axis: Normalized arc axis vector (from calculate_direction_from_angles) - represents anterior
        offset_mm: Offset distance in mm (default 2.0mm)

    Returns:
        3D offset vector [dx, dy, dz] perpendicular to trajectory
    """
    track_type_lower = track_type.lower() if track_type else "central"

    if track_type_lower == "central" or not track_type:
        return np.array([0.0, 0.0, 0.0])

    # Arc axis represents the anterior direction (rotates with ring angle)
    anterior_direction = arc_axis

    # Calculate medial/lateral direction as cross product of trajectory and anterior
    # This gives us a vector perpendicular to both
    lateral_direction = np.cross(trajectory_direction, anterior_direction)
    lateral_direction = lateral_direction / np.linalg.norm(lateral_direction)

    # Determine offset direction based on track type
    if track_type_lower == "anterior":
        # Anterior: use arc axis direction
        offset_direction = anterior_direction

    elif track_type_lower == "posterior":
        # Posterior: opposite of arc axis
        offset_direction = -anterior_direction

    elif track_type_lower == "medial":
        # Medial: towards center (x=100)
        # Use lateral direction, but flip sign based on hemisphere
        if target_x < 100:
            # Right hemisphere: check if lateral_direction points towards center
            # If lateral_direction.x > 0, it points left (towards center for right side)
            offset_direction = lateral_direction if lateral_direction[0] > 0 else -lateral_direction
        else:
            # Left hemisphere: check if lateral_direction points towards center
            # If lateral_direction.x < 0, it points right (towards center for left side)
            offset_direction = lateral_direction if lateral_direction[0] < 0 else -lateral_direction

    elif track_type_lower == "lateral":
        # Lateral: away from center
        # Opposite of medial
        if target_x < 100:
            # Right hemisphere: lateral points away from center (right, -X)
            offset_direction = lateral_direction if lateral_direction[0] < 0 else -lateral_direction
        else:
            # Left hemisphere: lateral points away from center (left, +X)
            offset_direction = lateral_direction if lateral_direction[0] > 0 else -lateral_direction

    else:
        # Unknown track type - treat as central
        import warnings
        warnings.warn(f"Unknown track type '{track_type}', treating as central", UserWarning)
        return np.array([0.0, 0.0, 0.0])

    # Scale by offset distance
    perpendicular_offset = offset_direction * offset_mm

    return perpendicular_offset


def calculate_mer_track_position(
    target: np.ndarray,
    ring_angle: float,
    arc_angle: float,
    track_type: str,
    depth_mm: float,
    entry_distance_mm: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MER track target and entry positions.

    MER tracks can be parallel to the central trajectory (medial, lateral, anterior, posterior)
    and offset along the trajectory by a depth value.

    Args:
        target: Central target position [x, y, z]
        ring_angle: Ring angle in degrees
        arc_angle: Arc angle in degrees
        track_type: Type of track ("central", "anterior", "posterior", "medial", "lateral")
        depth_mm: Depth offset along trajectory (negative = towards entry, positive = beyond target)
        entry_distance_mm: Distance from target to entry point (default 100mm)

    Returns:
        Tuple of (mer_target, mer_entry) positions
    """
    # Calculate trajectory direction and arc axis from ring/arc angles
    direction, arc_axis = calculate_direction_from_angles(ring_angle, arc_angle)

    # Calculate perpendicular offset for parallel track using arc axis as anterior reference
    perpendicular_offset = calculate_parallel_track_offset(
        track_type, target[0], direction, arc_axis
    )

    # Calculate MER target position:
    # 1. Apply perpendicular offset
    # 2. Apply depth offset along trajectory direction
    # Note: negative depth = towards entry (above target), positive = below target
    # Since direction points towards entry, we subtract depth to get correct sign
    mer_target = target + perpendicular_offset - direction * depth_mm

    # Calculate MER entry position (parallel to central trajectory)
    mer_entry = mer_target + direction * entry_distance_mm

    return mer_target, mer_entry


def transform_surgical_csv(
    csv_data: List[Dict],
    transform_files: List[Path],
    invert_flags: List[bool],
    transform_types: List[str]
) -> List[Dict]:
    """
    Transform coordinates in surgical data CSV.

    Surgical data starts in frame space, so frame registration transforms
    must be applied first to get to CT space, then ANTs transforms.

    Instead of transforming ring/arc angles (which is mathematically complex),
    we calculate a virtual_entry_point from the angles, transform both target
    and entry as simple XYZ coordinates, then the trajectory is implicit in
    the two transformed points.

    Args:
        csv_data: List of dictionaries (CSV rows)
        transform_files: List of transform file paths
        invert_flags: List of booleans for each transform
        transform_types: List of transform types ('frame_registration', 'ants', etc.)

    Returns:
        Updated list of dictionaries with transformed coordinates and virtual_entry_point
    """
    # Extract coordinates, angles, and calculate virtual entry points
    targets = []
    entries = []
    clinical_targets = []
    research_targets = []

    # Track unique MER track types that appear in the data
    mer_track_types = set()

    for row in csv_data:
        try:
            x = float(row['x'])
            y = float(row['y'])
            z = float(row['z'])
            ring = float(row['ring'])
            arc = float(row['arc'])

            target = np.array([x, y, z])

            # Calculate virtual entry point from ring/arc angles
            # Direction points along trajectory from target toward entry
            direction, arc_axis = calculate_direction_from_angles(ring, arc)
            entry = target + direction * 100  # 100mm trajectory length

            targets.append(target)
            entries.append(entry)

            # Calculate clinical MER track position if available (at specified depth)
            clinical_track = row.get('clinical_track', '')
            clinical_depth = float(row.get('clinical_depth', 0)) if row.get('clinical_depth') else 0.0

            if clinical_track and clinical_track.strip():
                clinical_target, _ = calculate_mer_track_position(
                    target, ring, arc, clinical_track, clinical_depth
                )
                clinical_targets.append(clinical_target)
                mer_track_types.add(clinical_track.lower().strip())
            else:
                clinical_targets.append(None)

            # Calculate research MER track position if available (at specified depth)
            research_track = row.get('research_track', '')
            research_depth = float(row.get('research_depth', 0)) if row.get('research_depth') else 0.0

            if research_track and research_track.strip():
                research_target, _ = calculate_mer_track_position(
                    target, ring, arc, research_track, research_depth
                )
                research_targets.append(research_target)
                mer_track_types.add(research_track.lower().strip())
            else:
                research_targets.append(None)

        except (ValueError, KeyError) as e:
            # Skip rows with invalid data
            targets.append(None)
            entries.append(None)
            clinical_targets.append(None)
            research_targets.append(None)

    # Filter valid points
    valid_indices = [i for i, t in enumerate(targets) if t is not None]
    valid_targets = np.array([targets[i] for i in valid_indices])
    valid_entries = np.array([entries[i] for i in valid_indices])

    # Filter valid clinical MER tracks
    valid_clinical_indices = [i for i in valid_indices if clinical_targets[i] is not None]
    if valid_clinical_indices:
        valid_clinical_targets = np.array([clinical_targets[i] for i in valid_clinical_indices])
    else:
        valid_clinical_targets = None

    # Filter valid research MER tracks
    valid_research_indices = [i for i in valid_indices if research_targets[i] is not None]
    if valid_research_indices:
        valid_research_targets = np.array([research_targets[i] for i in valid_research_indices])
    else:
        valid_research_targets = None

    # Calculate MER track targets and entries for ALL standard track types (extended 5mm below target)
    # Generate for all tracks regardless of whether they're used in clinical/research
    all_track_types = ['anterior', 'posterior', 'medial', 'lateral', 'central']
    mer_track_positions = {}  # {track_type: {'targets': [...], 'entries': [...], 'valid_indices': [...]}}

    for track_type in all_track_types:
        track_targets = []
        track_entries = []
        track_valid_indices = []

        for i in valid_indices:
            try:
                row = csv_data[i]
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                ring = float(row['ring'])
                arc = float(row['arc'])
                target = np.array([x, y, z])

                # Calculate position 5mm below target for this track type (for visualization)
                mer_target, mer_entry = calculate_mer_track_position(
                    target, ring, arc, track_type, depth_mm=5.0
                )
                track_targets.append(mer_target)
                track_entries.append(mer_entry)
                track_valid_indices.append(i)
            except:
                continue

        if track_targets:
            mer_track_positions[track_type] = {
                'targets': np.array(track_targets),
                'entries': np.array(track_entries),
                'valid_indices': track_valid_indices
            }

    if len(valid_targets) == 0:
        return csv_data

    # Separate frame registration transforms from ANTs transforms
    frame_transforms = []
    ants_transforms = []
    ants_invert_flags = []

    for tf, invert, tf_type in zip(transform_files, invert_flags, transform_types):
        if tf_type == 'frame_registration':
            frame_transforms.append((tf, invert))
        else:
            ants_transforms.append(tf)
            ants_invert_flags.append(invert)

    # Apply frame registration transforms first (frame space -> CT space)
    current_targets = valid_targets.copy()
    current_entries = valid_entries.copy()
    current_clinical_targets = valid_clinical_targets.copy() if valid_clinical_targets is not None else None
    current_research_targets = valid_research_targets.copy() if valid_research_targets is not None else None

    # Copy MER track positions for transformation
    current_mer_track_positions = {}
    for track_type, positions in mer_track_positions.items():
        current_mer_track_positions[track_type] = {
            'targets': positions['targets'].copy(),
            'entries': positions['entries'].copy(),
            'valid_indices': positions['valid_indices']
        }

    for frame_tf, invert in frame_transforms:
        # Load the transformation matrix
        with open(frame_tf, 'r') as f:
            frame_data = json.load(f)
        matrix = np.array(frame_data['registration']['transformation_matrix'])

        # Invert if requested
        if invert:
            matrix = np.linalg.inv(matrix)

        # Transform targets and entries as simple XYZ points
        current_targets = apply_4x4_matrix_transform(current_targets, matrix)
        current_entries = apply_4x4_matrix_transform(current_entries, matrix)

        # Transform clinical MER track positions
        if current_clinical_targets is not None:
            current_clinical_targets = apply_4x4_matrix_transform(current_clinical_targets, matrix)

        # Transform research MER track positions
        if current_research_targets is not None:
            current_research_targets = apply_4x4_matrix_transform(current_research_targets, matrix)

        # Transform MER track positions for each track type
        for track_type in current_mer_track_positions:
            current_mer_track_positions[track_type]['targets'] = apply_4x4_matrix_transform(
                current_mer_track_positions[track_type]['targets'], matrix
            )
            current_mer_track_positions[track_type]['entries'] = apply_4x4_matrix_transform(
                current_mer_track_positions[track_type]['entries'], matrix
            )

    # Then apply ANTs transforms (e.g. CT space -> template space)
    if ants_transforms:
        # Transform both point sets through ANTs
        transformed_targets = apply_ants_transforms_to_points(
            current_targets, ants_transforms, use_inverse=ants_invert_flags
        )
        transformed_entries = apply_ants_transforms_to_points(
            current_entries, ants_transforms, use_inverse=ants_invert_flags
        )
        current_targets = transformed_targets
        current_entries = transformed_entries

        # Transform clinical MER track positions
        if current_clinical_targets is not None:
            current_clinical_targets = apply_ants_transforms_to_points(
                current_clinical_targets, ants_transforms, use_inverse=ants_invert_flags
            )

        # Transform research MER track positions
        if current_research_targets is not None:
            current_research_targets = apply_ants_transforms_to_points(
                current_research_targets, ants_transforms, use_inverse=ants_invert_flags
            )

        # Transform MER track positions for each track type
        for track_type in current_mer_track_positions:
            current_mer_track_positions[track_type]['targets'] = apply_ants_transforms_to_points(
                current_mer_track_positions[track_type]['targets'], ants_transforms, use_inverse=ants_invert_flags
            )
            current_mer_track_positions[track_type]['entries'] = apply_ants_transforms_to_points(
                current_mer_track_positions[track_type]['entries'], ants_transforms, use_inverse=ants_invert_flags
            )

    # Update CSV data with transformed coordinates (2 decimal places)
    for idx, valid_idx in enumerate(valid_indices):
        # Store original coordinates and angles
        csv_data[valid_idx]['x_original'] = csv_data[valid_idx]['x']
        csv_data[valid_idx]['y_original'] = csv_data[valid_idx]['y']
        csv_data[valid_idx]['z_original'] = csv_data[valid_idx]['z']
        csv_data[valid_idx]['ring_original'] = csv_data[valid_idx]['ring']
        csv_data[valid_idx]['arc_original'] = csv_data[valid_idx]['arc']

        # Update with transformed target coordinates (central target)
        csv_data[valid_idx]['x'] = f"{current_targets[idx, 0]:.2f}"
        csv_data[valid_idx]['y'] = f"{current_targets[idx, 1]:.2f}"
        csv_data[valid_idx]['z'] = f"{current_targets[idx, 2]:.2f}"

        # Add virtual entry point (for central trajectory visualization)
        csv_data[valid_idx]['entry_x'] = f"{current_entries[idx, 0]:.2f}"
        csv_data[valid_idx]['entry_y'] = f"{current_entries[idx, 1]:.2f}"
        csv_data[valid_idx]['entry_z'] = f"{current_entries[idx, 2]:.2f}"

        # Keep original ring/arc angles (no longer transformed)
        # These represent the original surgical plan

    # Add clinical MER recording positions (at specified depth on the track)
    if current_clinical_targets is not None:
        for local_idx, valid_idx in enumerate(valid_clinical_indices):
            csv_data[valid_idx]['clinical_target_x'] = f"{current_clinical_targets[local_idx, 0]:.2f}"
            csv_data[valid_idx]['clinical_target_y'] = f"{current_clinical_targets[local_idx, 1]:.2f}"
            csv_data[valid_idx]['clinical_target_z'] = f"{current_clinical_targets[local_idx, 2]:.2f}"

    # Add research MER recording positions (at specified depth on the track)
    if current_research_targets is not None:
        for local_idx, valid_idx in enumerate(valid_research_indices):
            csv_data[valid_idx]['research_target_x'] = f"{current_research_targets[local_idx, 0]:.2f}"
            csv_data[valid_idx]['research_target_y'] = f"{current_research_targets[local_idx, 1]:.2f}"
            csv_data[valid_idx]['research_target_z'] = f"{current_research_targets[local_idx, 2]:.2f}"

    # Add MER track targets and entries for each track type (extended 5mm below central target)
    for track_type, positions in current_mer_track_positions.items():
        track_targets = positions['targets']
        track_entries = positions['entries']
        track_indices = positions['valid_indices']

        for local_idx, valid_idx in enumerate(track_indices):
            # Add target position (5mm below central target on this track, for visualization)
            csv_data[valid_idx][f'mer_{track_type}_target_x'] = f"{track_targets[local_idx, 0]:.2f}"
            csv_data[valid_idx][f'mer_{track_type}_target_y'] = f"{track_targets[local_idx, 1]:.2f}"
            csv_data[valid_idx][f'mer_{track_type}_target_z'] = f"{track_targets[local_idx, 2]:.2f}"

            # Add entry position for this track
            csv_data[valid_idx][f'mer_{track_type}_entry_x'] = f"{track_entries[local_idx, 0]:.2f}"
            csv_data[valid_idx][f'mer_{track_type}_entry_y'] = f"{track_entries[local_idx, 1]:.2f}"
            csv_data[valid_idx][f'mer_{track_type}_entry_z'] = f"{track_entries[local_idx, 2]:.2f}"

    return csv_data


def convert_csv_to_json(csv_data: List[Dict]) -> Dict:
    """
    Convert surgical CSV data to JSON format.

    Args:
        csv_data: List of dictionaries (CSV rows)

    Returns:
        Dictionary with metadata and records
    """
    # Create JSON structure
    json_data = {
        "metadata": {
            "num_records": len(csv_data),
            "format": "surgical_data",
            "coordinates_transformed": any('x_original' in row for row in csv_data)
        },
        "records": csv_data
    }

    return json_data
