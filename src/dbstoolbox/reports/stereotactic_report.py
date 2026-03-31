"""HTML report generation for stereotactic targeting analysis."""

from dbstoolbox import __version__
from typing import List, Dict, Optional, Tuple, Callable, Union
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os
import json
import base64
import io
from scipy.ndimage import map_coordinates
from dbstoolbox.visualization import Plot3DGenerator
from dbstoolbox.utils.nifti_slice_utils import NiftiSliceGenerator
from dbstoolbox.reports.brain_shift_report import BrainShiftAnalyzer


class StereotacticReportGenerator:
    """
    Generates HTML reports for stereotactic targeting analysis.

    Creates comprehensive reports including:
    - 3D visualization of trajectories and electrodes
    - Polar charts showing electrode positions at each contact depth
    - Statistical tables with distance and offset measurements
    """

    # Standard MER track offset (2mm)
    MER_OFFSET_MM = 2.0

    def __init__(
        self,
        surgical_targets: List[Dict],
        electrode_trajectories: List[Dict],
        precalculated_positions: Dict[int, Dict[float, List[Dict]]],
        get_3d_figure_callback: Optional[Callable[[], go.Figure]] = None,
        nifti_files: Optional[List[Dict]] = None,
        electrode_trajectories_2: Optional[List[Dict]] = None,
        electrode_metadata_1: Optional[Dict] = None,
        electrode_metadata_2: Optional[Dict] = None,
        patient_id: str = '',
    ):
        """
        Initialize the report generator.

        Args:
            surgical_targets: List of parsed surgical target data
            electrode_trajectories: List of parsed electrode trajectory data
            precalculated_positions: Pre-calculated electrode positions {target_idx: {depth: [positions]}}
            get_3d_figure_callback: Optional callback to get the current 3D figure for export
            nifti_files: Optional list of loaded NIfTI file dictionaries with 'data' and 'affine'
            electrode_trajectories_2: Optional second set of electrode trajectories for brain shift analysis
            electrode_metadata_1: Optional metadata from first electrode reconstruction
            electrode_metadata_2: Optional metadata from second electrode reconstruction
            patient_id: Optional patient identifier (overrides value from surgical JSON)
        """
        self.surgical_targets = surgical_targets
        self.electrode_trajectories = electrode_trajectories
        self.precalculated_positions = precalculated_positions
        self.get_3d_figure_callback = get_3d_figure_callback
        self.nifti_files = nifti_files or []
        self.electrode_trajectories_2 = electrode_trajectories_2
        self.electrode_metadata_1 = electrode_metadata_1
        self.electrode_metadata_2 = electrode_metadata_2
        self.patient_id = patient_id

    # Depth range for pre-calculation
    DEPTH_MIN = -80.0
    DEPTH_MAX = 10.0
    DEPTH_STEP = 0.5

    @classmethod
    def from_json(
        cls,
        surgical_json: Union[str, Dict, List[Dict]],
        electrode_json: Union[str, Dict],
        get_3d_figure_callback: Optional[Callable[[], go.Figure]] = None,
        electrode_json_2: Optional[Union[str, Dict]] = None
    ) -> 'StereotacticReportGenerator':
        """
        Create a report generator from raw JSON data or file paths.

        Args:
            surgical_json: Either a file path to surgical JSON, or the loaded data
                          (list of surgical targets or dict containing target data)
            electrode_json: Either a file path to electrode JSON, or the loaded data
            get_3d_figure_callback: Optional callback to get the current 3D figure for export
            electrode_json_2: Optional second electrode reconstruction for brain shift analysis

        Returns:
            StereotacticReportGenerator instance

        Example:
            # From file paths
            generator = StereotacticReportGenerator.from_json(
                'surgical_plan.json',
                'electrode_reconstruction.json'
            )

            # From loaded data
            generator = StereotacticReportGenerator.from_json(
                surgical_data_list,
                electrode_data_dict
            )

            # With brain shift analysis
            generator = StereotacticReportGenerator.from_json(
                'surgical_plan.json',
                'electrode_reconstruction_1.json',
                electrode_json_2='electrode_reconstruction_2.json'
            )
        """
        # Load surgical data
        if isinstance(surgical_json, str):
            with open(surgical_json, 'r') as f:
                surgical_data = json.load(f)
            surgical_filename = os.path.basename(surgical_json)
        else:
            surgical_data = surgical_json
            surgical_filename = 'surgical_data.json'

        # Load electrode data
        if isinstance(electrode_json, str):
            with open(electrode_json, 'r') as f:
                electrode_data = json.load(f)
            electrode_filename = os.path.basename(electrode_json)
        else:
            electrode_data = electrode_json
            electrode_filename = 'electrode_data.json'

        # Parse surgical targets
        surgical_targets = []
        if isinstance(surgical_data, list):
            # List of target dictionaries
            for idx, row in enumerate(surgical_data):
                target_info = cls._parse_surgical_target(row, idx, surgical_filename)
                if target_info:
                    surgical_targets.append(target_info)
        elif isinstance(surgical_data, dict):
            # Dict containing 'targets', 'data', or 'records' key
            targets_list = surgical_data.get('targets', surgical_data.get('data', surgical_data.get('records', [])))
            for idx, row in enumerate(targets_list):
                target_info = cls._parse_surgical_target(row, idx, surgical_filename)
                if target_info:
                    surgical_targets.append(target_info)

        # Parse electrode trajectories and extract metadata
        electrode_trajectories = []
        electrodes = electrode_data.get('electrodes', [])
        is_transformed_1 = electrode_data.get('metadata', {}).get('transformed', False)
        for idx, electrode in enumerate(electrodes):
            traj_info = cls._parse_electrode_trajectory(electrode, idx, electrode_filename, is_transformed_1)
            if traj_info:
                electrode_trajectories.append(traj_info)

        # Extract metadata from first electrode reconstruction
        electrode_metadata_1 = electrode_data.get('metadata', {})

        # Parse second electrode reconstruction if provided (for brain shift analysis)
        electrode_trajectories_2 = None
        electrode_metadata_2 = None
        if electrode_json_2 is not None:
            # Load second electrode data
            if isinstance(electrode_json_2, str):
                with open(electrode_json_2, 'r') as f:
                    electrode_data_2 = json.load(f)
                electrode_filename_2 = os.path.basename(electrode_json_2)
            else:
                electrode_data_2 = electrode_json_2
                electrode_filename_2 = 'electrode_data_2.json'

            # Parse second electrode trajectories
            electrode_trajectories_2 = []
            electrodes_2 = electrode_data_2.get('electrodes', [])
            is_transformed_2 = electrode_data_2.get('metadata', {}).get('transformed', False)
            for idx, electrode in enumerate(electrodes_2):
                traj_info = cls._parse_electrode_trajectory(electrode, idx, electrode_filename_2, is_transformed_2)
                if traj_info:
                    electrode_trajectories_2.append(traj_info)

            # Extract metadata from second electrode reconstruction
            electrode_metadata_2 = electrode_data_2.get('metadata', {})

        # Pre-calculate electrode positions
        precalculated_positions = cls._precalculate_all_positions(
            surgical_targets,
            electrode_trajectories
        )

        return cls(
            surgical_targets=surgical_targets,
            electrode_trajectories=electrode_trajectories,
            precalculated_positions=precalculated_positions,
            get_3d_figure_callback=get_3d_figure_callback,
            electrode_trajectories_2=electrode_trajectories_2,
            electrode_metadata_1=electrode_metadata_1,
            electrode_metadata_2=electrode_metadata_2
        )

    @staticmethod
    def _parse_surgical_target(row: Dict, idx: int, filename: str) -> Optional[Dict]:
        """Parse a surgical target row into structured data."""
        try:
            x = float(row.get('x', 0))
            y = float(row.get('y', 0))
            z = float(row.get('z', 0))

            ring = float(row.get('ring_original', row.get('ring', 0)))
            arc = float(row.get('arc_original', row.get('arc', 0)))

            target = np.array([x, y, z])

            # Calculate trajectory direction and arc_axis from ring/arc (Leksell frame)
            orig_direction, orig_arc_axis = StereotacticReportGenerator._calculate_direction_from_angles(ring, arc)

            # Check if this is transformed data (has entry point)
            is_transformed = 'entry_x' in row and 'entry_y' in row and 'entry_z' in row

            if is_transformed:
                entry = np.array([
                    float(row['entry_x']),
                    float(row['entry_y']),
                    float(row['entry_z'])
                ])
                # Calculate actual direction from transformed entry/target
                actual_direction = entry - target
                actual_direction = actual_direction / np.linalg.norm(actual_direction)
                direction = actual_direction

                # Check if transformed MER track coordinates are available
                has_mer_coords = ('mer_anterior_target_x' in row and
                                 'mer_lateral_target_x' in row)

                if not has_mer_coords:
                    raise ValueError(
                        f"Transformed surgical data missing MER track coordinates. "
                        f"Cannot calculate anterior/lateral directions without pre-calculated MER positions."
                    )

                # Extract MER track positions
                mer_ant_target = np.array([
                    float(row['mer_anterior_target_x']),
                    float(row['mer_anterior_target_y']),
                    float(row['mer_anterior_target_z'])
                ])
                mer_lat_target = np.array([
                    float(row['mer_lateral_target_x']),
                    float(row['mer_lateral_target_y']),
                    float(row['mer_lateral_target_z'])
                ])

                mer_ant_entry = np.array([
                    float(row['mer_anterior_entry_x']),
                    float(row['mer_anterior_entry_y']),
                    float(row['mer_anterior_entry_z'])
                ])
                mer_ant_direction = mer_ant_entry - mer_ant_target
                mer_ant_direction = mer_ant_direction / np.linalg.norm(mer_ant_direction)
                mer_ant_at_depth0 = mer_ant_target + mer_ant_direction * 5.0

                mer_lat_entry = np.array([
                    float(row['mer_lateral_entry_x']),
                    float(row['mer_lateral_entry_y']),
                    float(row['mer_lateral_entry_z'])
                ])
                mer_lat_direction = mer_lat_entry - mer_lat_target
                mer_lat_direction = mer_lat_direction / np.linalg.norm(mer_lat_direction)
                mer_lat_at_depth0 = mer_lat_target + mer_lat_direction * 5.0

                # Calculate anterior and lateral from actual MER track offsets
                anterior_offset = mer_ant_at_depth0 - target
                lateral_offset = mer_lat_at_depth0 - target

                # Normalize to get unit vectors
                anterior = anterior_offset / np.linalg.norm(anterior_offset)
                lateral = lateral_offset / np.linalg.norm(lateral_offset)
            else:
                entry = target + orig_direction * 100
                direction = orig_direction
                anterior = orig_arc_axis / np.linalg.norm(orig_arc_axis)
                lateral = np.cross(direction, anterior)
                if np.linalg.norm(lateral) > 1e-10:
                    lateral = lateral / np.linalg.norm(lateral)
                else:
                    lateral = np.array([1.0, 0.0, 0.0])

            # Determine hemisphere
            hemisphere_str = row.get('hemisphere', '').lower()
            if hemisphere_str in ['left', 'l']:
                is_left_hemisphere = True
            elif hemisphere_str in ['right', 'r']:
                is_left_hemisphere = False
            else:
                if is_transformed:
                    is_left_hemisphere = x < 0
                else:
                    is_left_hemisphere = x > 100

            # Adjust lateral direction based on hemisphere
            if is_transformed:
                if is_left_hemisphere:
                    if lateral[0] < 0:
                        lateral = -lateral
                else:
                    if lateral[0] > 0:
                        lateral = -lateral
            else:
                if is_left_hemisphere:
                    if lateral[0] > 0:
                        lateral = -lateral
                else:
                    if lateral[0] < 0:
                        lateral = -lateral

            return {
                'target': target,
                'entry': entry,
                'direction': direction,
                'anterior': anterior,
                'lateral': lateral,
                'ring': ring,
                'arc': arc,
                'patient_id': row.get('patient_id', f'Target {idx+1}'),
                'hemisphere': row.get('hemisphere', ''),
                'anatomical_target': row.get('anatomical_target', ''),
                'filename': filename,
                'target_x': x,
                'is_left_hemisphere': is_left_hemisphere,
                'surgeons': row.get('surgeons', ''),
                'researchers': row.get('researchers', ''),
                'notes': row.get('notes', ''),
                'x_original': row.get('x_original', ''),
                'y_original': row.get('y_original', ''),
                'z_original': row.get('z_original', ''),
                'ring_original': row.get('ring_original', ''),
                'arc_original': row.get('arc_original', ''),
                'clinical_track': row.get('clinical_track', ''),
                'clinical_depth': row.get('clinical_depth', ''),
                'research_track': row.get('research_track', ''),
                'research_depth': row.get('research_depth', '')
            }
        except (ValueError, KeyError) as e:
            print(f"Error parsing surgical target: {e}")
            return None

    @staticmethod
    def _trajectory_from_polynomial(electrode: Dict) -> np.ndarray:
        """Generate trajectory points from polynomial coefficients."""
        poly_coeffs = np.array(electrode['polynomial'])
        t_vals = np.linspace(0, 1, 100)
        points = np.array([
            [np.polyval(poly_coeffs[:, dim], t) for dim in range(3)]
            for t in t_vals
        ])
        return points

    @staticmethod
    def _parse_electrode_trajectory(electrode: Dict, idx: int, filename: str, is_transformed: bool = False) -> Optional[Dict]:
        """Parse an electrode into trajectory data."""
        try:
            if 'trajectory_coordinates' in electrode:
                trajectory = np.array(electrode['trajectory_coordinates'])
            elif 'polynomial' in electrode and not is_transformed:
                trajectory = StereotacticReportGenerator._trajectory_from_polynomial(electrode)
            else:
                return None

            contacts = None
            if 'contact_positions_3d' in electrode:
                contacts = np.array(electrode['contact_positions_3d'])

            # Get entry and tip positions if available
            entry_position = None
            if 'entry_position' in electrode:
                entry_position = np.array(electrode['entry_position'])

            tip_position = None
            if 'tip_position' in electrode:
                tip_position = np.array(electrode['tip_position'])

            # Get orientation data for directional electrodes
            orientation = electrode.get('orientation')

            return {
                'trajectory': trajectory,
                'contacts': contacts,
                'entry_position': entry_position,
                'tip_position': tip_position,
                'orientation': orientation,
                'electrode_idx': idx,
                'filename': filename,
                'label': ' - '.join(filter(None, [f'E{idx+1}', electrode.get('electrode_type', ''), electrode.get('side', '').capitalize()]))
            }
        except (ValueError, KeyError) as e:
            print(f"Error parsing electrode trajectory: {e}")
            return None

    @staticmethod
    def _calculate_direction_from_angles(ring_angle: float, arc_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate trajectory direction and arc axis from ring/arc angles."""
        ring_rad = -np.radians(ring_angle)
        arc_rad = -np.radians(arc_angle)

        base_direction = np.array([-1.0, 0.0, 0.0])
        arc_axis = np.array([0.0, 0.0, 1.0])

        ring_rotation = np.array([
            [1, 0, 0],
            [0, np.cos(ring_rad), -np.sin(ring_rad)],
            [0, np.sin(ring_rad), np.cos(ring_rad)]
        ])
        arc_axis = ring_rotation @ arc_axis
        arc_axis = arc_axis / np.linalg.norm(arc_axis)

        k = arc_axis
        v = base_direction
        direction = (v * np.cos(arc_rad) +
                     np.cross(k, v) * np.sin(arc_rad) +
                     k * np.dot(k, v) * (1 - np.cos(arc_rad)))
        direction = direction / np.linalg.norm(direction)

        return direction, arc_axis

    @staticmethod
    def _precalculate_all_positions(
        surgical_targets: List[Dict],
        electrode_trajectories: List[Dict]
    ) -> Dict[int, Dict[float, List[Dict]]]:
        """Pre-calculate electrode positions for all depth values."""
        precalculated = {}
        depths = np.arange(
            StereotacticReportGenerator.DEPTH_MIN,
            StereotacticReportGenerator.DEPTH_MAX + StereotacticReportGenerator.DEPTH_STEP,
            StereotacticReportGenerator.DEPTH_STEP
        )

        for target_idx, surgical_target in enumerate(surgical_targets):
            precalculated[target_idx] = {}
            for depth in depths:
                depth_key = round(depth, 1)
                positions = StereotacticReportGenerator._calculate_electrode_positions(
                    surgical_target,
                    electrode_trajectories,
                    depth
                )
                precalculated[target_idx][depth_key] = positions

        return precalculated

    @staticmethod
    def _calculate_electrode_positions(
        surgical_target: Dict,
        electrode_trajectories: List[Dict],
        depth_mm: float
    ) -> List[Dict]:
        """Calculate electrode intersection positions for a given depth."""
        target = surgical_target['target']
        direction = surgical_target['direction']
        anterior = surgical_target['anterior']
        lateral = surgical_target['lateral']

        plane_point = target - direction * depth_mm

        positions = []

        for i, electrode in enumerate(electrode_trajectories):
            color_idx = i % 360
            color = f'hsl({color_idx * 60}, 70%, 50%)'
            trajectory = electrode['trajectory']

            # Find intersection
            intersection = StereotacticReportGenerator._find_plane_intersection_static(
                trajectory, plane_point, direction
            )

            if intersection is not None:
                diff = intersection - plane_point
                ant_coord = float(np.dot(diff, anterior))
                lat_coord = float(np.dot(diff, lateral))

                lat_coord_stats = lat_coord

                is_left = surgical_target.get('is_left_hemisphere', False)
                if is_left:
                    lat_coord_display = -lat_coord
                else:
                    lat_coord_display = lat_coord

                r = float(np.sqrt(lat_coord**2 + ant_coord**2))
                theta = float(-np.degrees(np.arctan2(lat_coord_display, ant_coord)))

                positions.append({
                    'r': r,
                    'theta': theta,
                    'lat_coord': lat_coord_display,
                    'lat_coord_stats': lat_coord_stats,
                    'ant_coord': ant_coord,
                    'short_label': electrode['label'],
                    'full_label': f"{electrode['filename']} {electrode['label']}",
                    'label': electrode['label'],
                    'color': color
                })

        return positions

    @staticmethod
    def _find_plane_intersection_static(
        trajectory: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray
    ) -> Optional[np.ndarray]:
        """Static version of plane intersection finder."""
        if len(trajectory) < 2:
            return None

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

        p1 = trajectory[0]
        p2 = trajectory[-1]
        line_dir = p2 - p1

        denom = np.dot(plane_normal, line_dir)
        if abs(denom) < 1e-10:
            return None

        t = np.dot(plane_normal, plane_point - p1) / denom
        return p1 + t * line_dir

    def _find_plane_intersection(
        self,
        trajectory: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find where a trajectory intersects a plane."""
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

    def _project_to_plane_coords(
        self,
        point: np.ndarray,
        plane_origin: np.ndarray,
        anterior: np.ndarray,
        lateral: np.ndarray
    ) -> Tuple[float, float]:
        """Project a 3D point to 2D plane coordinates (lateral, anterior)."""
        diff = point - plane_origin
        anterior_coord = float(np.dot(diff, anterior))
        lateral_coord = float(np.dot(diff, lateral))
        return lateral_coord, anterior_coord

    def _calculate_electrode_position_at_depth(
        self,
        surgical_target: Dict,
        electrode: Dict,
        depth_mm: float
    ) -> Optional[Dict]:
        """Calculate electrode intersection position for a specific depth."""
        target = surgical_target['target']
        direction = surgical_target['direction']
        anterior = surgical_target['anterior']
        lateral = surgical_target['lateral']

        # Calculate plane position
        plane_point = target - direction * depth_mm

        trajectory = electrode['trajectory']
        intersection = self._find_plane_intersection(trajectory, plane_point, direction)

        if intersection is None:
            return None

        lat_coord, ant_coord = self._project_to_plane_coords(
            intersection, plane_point, anterior, lateral
        )

        # Store original lat_coord for stats
        lat_coord_stats = lat_coord

        # Flip lateral coordinate for right hemisphere so medial points toward midline
        is_left = surgical_target.get('is_left_hemisphere', False)
        if is_left:
            lat_coord_display = lat_coord
        else:
            lat_coord_display = -lat_coord

        r = float(np.sqrt(lat_coord**2 + ant_coord**2))
        theta = float(-np.degrees(np.arctan2(lat_coord_display, ant_coord)))

        return {
            'r': r,
            'theta': theta,
            'lat_coord': lat_coord_display,
            'lat_coord_stats': lat_coord_stats,
            'ant_coord': ant_coord,
            'short_label': electrode['label'],
            'full_label': f"{electrode['filename']} {electrode['label']}",
            'label': electrode['label'],
            'color': 'hsl(0, 70%, 50%)',  # Default color
            'x': float(intersection[0]),
            'y': float(intersection[1]),
            'z': float(intersection[2])
        }

    def _get_electrode_positions_at_depth(self, target_idx: int, depth_mm: float) -> List[Dict]:
        """Get electrode positions at a given depth, using pre-calculated if available or calculating on-the-fly."""
        # Try pre-calculated first
        depth_key = round(depth_mm, 1)
        positions = self.precalculated_positions.get(target_idx, {}).get(depth_key, [])
        if positions:
            return positions

        # Calculate on-the-fly for non-standard depths (e.g., contact positions)
        surgical_target = self.surgical_targets[target_idx]
        positions = []

        for i, electrode in enumerate(self.electrode_trajectories):
            # Use the same color palette as 3D plot (electrodes start after surgical targets)
            color = self._get_electrode_color(i)

            pos = self._calculate_electrode_position_at_depth(surgical_target, electrode, depth_mm)
            if pos:
                pos['color'] = color
                positions.append(pos)

        return positions

    def _find_closest_electrode_for_target(self, target_idx: int) -> Optional[Dict]:
        """
        Find the electrode closest to a surgical target at depth 0.

        This method ensures each electrode is matched to the surgical target it is actually closest to,
        preventing cross-hemisphere mismatches.
        """
        surgical_target = self.surgical_targets[target_idx]
        electrode_positions = self._get_electrode_positions_at_depth(target_idx, 0.0)
        if not electrode_positions:
            return None

        # Find the electrode with smallest distance to center of THIS target
        closest_dist = float('inf')
        closest_electrode_idx = None

        for i, pos in enumerate(electrode_positions):
            electrode_label = pos.get('short_label', pos.get('label'))
            current_dist = pos['r']

            # Check if this electrode is closer to another target
            is_closest_to_this_target = True

            for other_idx in range(len(self.surgical_targets)):
                if other_idx == target_idx:
                    continue

                # Calculate distance to other target
                other_positions = self._get_electrode_positions_at_depth(other_idx, 0.0)
                for other_pos in other_positions:
                    other_label = other_pos.get('short_label', other_pos.get('label'))
                    if other_label == electrode_label:
                        # If electrode is closer to the other target, don't use it for this target
                        if other_pos['r'] < current_dist:
                            is_closest_to_this_target = False
                            break

                if not is_closest_to_this_target:
                    break

            # Update closest if this electrode is closest to this target
            if is_closest_to_this_target and current_dist < closest_dist:
                closest_dist = current_dist
                closest_electrode_idx = i

        if closest_electrode_idx is None:
            return None

        return self.electrode_trajectories[closest_electrode_idx]

    def _calculate_contact_depths(self, surgical_target: Dict, electrode: Dict) -> List[Optional[float]]:
        """
        Calculate the depth at which each contact intersects the plane perpendicular to the trajectory.

        Returns a list of 4 depths [C0, C1, C2, C3] or None if contact doesn't exist.
        C0 is the bottom contact (furthest from entry), C3 is the top contact (closest to entry).
        """
        if electrode.get('contacts') is None:
            return [None, None, None, None]

        contacts = electrode['contacts']
        target = surgical_target['target']
        direction = surgical_target['direction']

        depths = []
        for i in range(4):  # C0, C1, C2, C3
            if i >= len(contacts):
                depths.append(None)
                continue

            contact_pos = contacts[i]

            # Calculate depth: project contact position onto the trajectory direction
            diff = contact_pos - target
            depth = -float(np.dot(diff, direction))

            # Round to 2 decimal places for cleaner display
            depth = round(depth, 2)
            depths.append(depth)

        return depths

    def _create_polar_chart_for_report(
        self,
        surgical_target: Dict,
        target_idx: int,
        depth_mm: float,
        matched_electrode: Optional[Dict] = None,
        contact_label: str = None
    ) -> go.Figure:
        """
        Create a polar chart for the HTML report (standalone, no dynamic updates).

        Args:
            surgical_target: The surgical target data
            target_idx: Index of the surgical target
            depth_mm: Depth relative to target
            matched_electrode: The matched electrode to display (if None, shows all)
            contact_label: Label for the contact (e.g., "Contact C3")
        """
        fig = go.Figure()

        # MER track positions (static reference points)
        mer_positions = {
            'Central': (0, 0),
            'Anterior': (0, self.MER_OFFSET_MM),
            'Posterior': (0, -self.MER_OFFSET_MM),
            'Lateral': (self.MER_OFFSET_MM, 0),
            'Medial': (-self.MER_OFFSET_MM, 0),
        }

        is_left = surgical_target.get('is_left_hemisphere', False)
        surgical_color_idx = target_idx % 360
        surgical_color = f'hsl({surgical_color_idx * 60}, 70%, 50%)'

        for track_name, (lat, ant) in mer_positions.items():
            if track_name == 'Central':
                color = surgical_color
            else:
                color = {
                    'Anterior': 'red',
                    'Posterior': 'blue',
                    'Lateral': 'yellow',
                    'Medial': 'green'
                }.get(track_name, 'gray')

            if is_left:
                lat = -lat

            r = np.sqrt(lat**2 + ant**2)
            theta = np.degrees(np.arctan2(lat, ant))
            if r < 0.01:
                r = 0
                theta = 0

            fig.add_trace(go.Scatterpolar(
                r=[r],
                theta=[theta],
                mode='markers+text',
                marker=dict(size=12, color=color, symbol='cross'),
                text=[track_name[0]],
                textposition='top center',
                name=track_name,
                hovertemplate=f'{track_name}<br>L: {lat:.1f}mm, A: {ant:.1f}mm<extra></extra>'
            ))

        # Get electrode positions at this depth
        electrode_positions = self._get_electrode_positions_at_depth(target_idx, depth_mm)
        electrode_ring_size = 29

        # Filter to only show the matched electrode if specified
        if matched_electrode is not None:
            matched_label = matched_electrode.get('label')
            electrode_positions = [pos for pos in electrode_positions if pos.get('label') == matched_label or pos.get('short_label') == matched_label]

        for pos in electrode_positions:
            # Grey ring
            fig.add_trace(go.Scatterpolar(
                r=[pos['r']],
                theta=[pos['theta']],
                mode='markers',
                marker=dict(
                    size=electrode_ring_size,
                    color='rgba(128, 128, 128, 0.3)',
                    symbol='circle',
                    line=dict(color='grey', width=2)
                ),
                name=f"{pos['label']} electrode",
                hovertemplate=f"{pos['label']}<br>Electrode (1.27mm)<extra></extra>",
                showlegend=False
            ))

            # Colored center
            fig.add_trace(go.Scatterpolar(
                r=[pos['r']],
                theta=[pos['theta']],
                mode='markers',
                marker=dict(
                    size=8,
                    color=pos['color'],
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                name=pos['label'],
                hovertemplate=f"{pos['label']}<br>L: {pos['lat_coord']:.2f}mm, A: {pos['ant_coord']:.2f}mm<br>r: {pos['r']:.2f}mm<extra></extra>"
            ))

        # Layout
        patient_id = surgical_target['patient_id']
        hemisphere = surgical_target['hemisphere']
        anat_target = surgical_target['anatomical_target']
        base_title = f"{patient_id} {hemisphere} {anat_target}".strip()

        if contact_label:
            title_text = f'{base_title}<br><span style="font-size:12px">{contact_label} (Depth: {depth_mm:+.2f}mm)</span>'
        else:
            title_text = f'{base_title}<br><span style="font-size:12px">Depth: {depth_mm:+.2f}mm from target</span>'

        if is_left:
            angular_tickvals = [0, 90, 180, 270]
            angular_ticktext = ['Ant', 'Lat', 'Post', 'Med']
        else:
            angular_tickvals = [0, 90, 180, 270]
            angular_ticktext = ['Ant', 'Med', 'Post', 'Lat']

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['1', '2', '3', '4', '5mm'],
                ),
                angularaxis=dict(
                    visible=True,
                    direction='clockwise',
                    rotation=90,
                    tickvals=angular_tickvals,
                    ticktext=angular_ticktext,
                ),
            ),
            showlegend=False,
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=14)
            ),
            margin=dict(t=60, b=20, l=20, r=20),
            height=300,
            width=350,
        )

        return fig

    @staticmethod
    def _get_color(idx: int) -> str:
        """Get color from the 3D plot color palette (shared for surgical and electrodes)."""
        colors = [
            '#636efa',  # Blue
            '#EF553B',  # Red
            '#00cc96',  # Green
            '#ab63fa',  # Purple
            '#FFA15A',  # Orange
            '#19d3f3',  # Cyan
            '#FF6692',  # Pink
            '#B6E880',  # Light green
            '#FF97FF',  # Light purple
            '#FECB52',  # Yellow
        ]
        return colors[idx % len(colors)]

    def _get_electrode_color(self, electrode_idx: int) -> str:
        """Get electrode color matching the 3D plot color palette."""
        # In the 3D plot, electrodes start after surgical targets
        # So electrode colors are offset by the number of surgical targets
        color_offset = len(self.surgical_targets)
        return self._get_color(electrode_idx + color_offset)

    def _get_brain_image_base64(self) -> str:
        """Load and encode brain coronal image as base64."""
        # Get path to brain image
        current_dir = os.path.dirname(os.path.abspath(__file__))
        media_dir = os.path.join(os.path.dirname(current_dir), 'media')
        brain_image_path = os.path.join(media_dir, 'brain-coronal.png')

        if not os.path.exists(brain_image_path):
            return ""

        with open(brain_image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')

    def _generate_trajectory_plane_slice(
        self,
        surgical_target: Dict,
        target_idx: int
    ) -> Optional[str]:
        """
        Generate a NIfTI slice through the surgical trajectory.

        This creates a plane that contains the entire trajectory (entry to target),
        showing the full brain with the trajectory projected as a line.

        Args:
            surgical_target: Surgical target dictionary
            target_idx: Index of the surgical target (for coloring)

        Returns:
            Base64-encoded PNG image, or None if generation fails
        """
        if not self.nifti_files:
            return None

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        # Get trajectory information
        target = surgical_target['target']
        entry = surgical_target.get('entry')
        direction = surgical_target['direction']
        is_left = surgical_target.get('is_left_hemisphere', False)

        if entry is None:
            return None

        # Use first NIfTI file
        nifti_file = self.nifti_files[0]
        nifti_data = nifti_file.get('data')
        affine = nifti_file.get('affine')

        if nifti_data is None or affine is None:
            return None

        # Handle 4D data
        if nifti_data.ndim == 4:
            nifti_data = nifti_data[..., 0]

        # Define coronal plane at target position
        # Plane normal is Y-axis in RAS: [0, 1, 0]
        plane_normal = np.array([0, 1, 0])
        plane_point = target  # Plane passes through target

        # Define plane axes (in RAS coordinates)
        # X-axis (left-right): horizontal in coronal view
        # Z-axis (inferior-superior): vertical in coronal view
        axis_horizontal = np.array([1, 0, 0])  # X in RAS (left-right)
        axis_vertical = np.array([0, 0, 1])    # Z in RAS (inferior-superior)

        # Get volume extent in RAS coordinates
        volume_shape = nifti_data.shape
        corners = np.array([
            [0, 0, 0],
            [volume_shape[0], 0, 0],
            [0, volume_shape[1], 0],
            [0, 0, volume_shape[2]],
            [volume_shape[0], volume_shape[1], 0],
            [volume_shape[0], 0, volume_shape[2]],
            [0, volume_shape[1], volume_shape[2]],
            [volume_shape[0], volume_shape[1], volume_shape[2]]
        ])

        # Convert corners to RAS
        corners_homogeneous = np.hstack([corners, np.ones((8, 1))])
        corners_ras = (affine @ corners_homogeneous.T).T[:, :3]

        # Project corners onto plane axes to find extent
        proj_x = np.dot(corners_ras - plane_point, axis_horizontal)
        proj_z = np.dot(corners_ras - plane_point, axis_vertical)

        x_range = [proj_x.min(), proj_x.max()]
        z_range = [proj_z.min(), proj_z.max()]

        # Sample with 0.5mm resolution
        resolution_mm = 0.5
        n_x = int((x_range[1] - x_range[0]) / resolution_mm)
        n_z = int((z_range[1] - z_range[0]) / resolution_mm)

        # Limit to reasonable size
        n_x = min(n_x, 800)
        n_z = min(n_z, 800)

        # Create sampling grid
        grid_x = np.linspace(x_range[0], x_range[1], n_x)
        grid_z = np.linspace(z_range[0], z_range[1], n_z)

        # Generate RAS coordinates for each grid point
        ras_points = np.zeros((n_z, n_x, 3))
        for i, z in enumerate(grid_z):
            for j, x in enumerate(grid_x):
                ras_points[i, j] = plane_point + x * axis_horizontal + z * axis_vertical

        # Convert RAS to voxel coordinates
        inv_affine = np.linalg.inv(affine)
        ras_homogeneous = np.concatenate([
            ras_points.reshape(-1, 3),
            np.ones((n_z * n_x, 1))
        ], axis=1)
        voxel_coords = (inv_affine @ ras_homogeneous.T).T[:, :3]
        voxel_coords = voxel_coords.reshape(n_z, n_x, 3)

        # Sample the volume
        from scipy.ndimage import map_coordinates
        slice_data = map_coordinates(
            nifti_data,
            [voxel_coords[..., 0].ravel(),
             voxel_coords[..., 1].ravel(),
             voxel_coords[..., 2].ravel()],
            order=1,
            mode='constant',
            cval=0.0
        ).reshape(n_z, n_x)

        # Flip left-right for neurological convention (left on right side)
        slice_data = np.fliplr(slice_data)

        # Normalize intensity
        vmin = np.percentile(slice_data, 2)
        vmax = np.percentile(slice_data, 98)

        # Create figure
        aspect_ratio = (x_range[1] - x_range[0]) / (z_range[1] - z_range[0])
        fig_height = 4
        fig_width = fig_height * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=80)

        # Display the slice
        # Note: extent is [left, right, bottom, top] and we've flipped L-R
        ax.imshow(
            slice_data,
            cmap='gray',
            extent=[-x_range[1], -x_range[0], z_range[0], z_range[1]],  # Flip X for neurological view
            origin='lower',
            interpolation='bilinear',
            vmin=vmin,
            vmax=vmax
        )

        # Project entry point onto the coronal plane
        # Entry projection: just use its X and Z coordinates (Y is on the plane)
        entry_x = np.dot(entry - plane_point, axis_horizontal)
        entry_z = np.dot(entry - plane_point, axis_vertical)

        # Target is at origin in plane coordinates
        target_x = 0.0
        target_z = 0.0

        # Flip X coordinates for neurological view
        entry_x = -entry_x
        target_x = -target_x

        # Get surgical color
        surgical_color = self._get_color(target_idx)

        # Draw trajectory line
        ax.plot([entry_x, target_x], [entry_z, target_z],
                color=surgical_color, linewidth=2, linestyle='-', alpha=0.8)

        # Mark entry and target
        ax.plot(entry_x, entry_z, 'o', color=surgical_color,
                markersize=6, markeredgecolor='white', markeredgewidth=1)
        ax.plot(target_x, target_z, 'D', color=surgical_color,
                markersize=6, markeredgecolor='white', markeredgewidth=1)

        ax.set_aspect('equal')
        ax.axis('off')

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format='png',
            bbox_inches='tight',
            dpi=80,
            pad_inches=0,
            facecolor='black',
            pil_kwargs={'optimize': True, 'compress_level': 9}
        )
        plt.close(fig)
        buf.seek(0)

        # Encode as base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    def _generate_nifti_slice_images(
        self,
        surgical_target: Dict,
        depths: List[float],
        target_idx: int
    ) -> Dict[str, str]:
        """
        Generate base64-encoded PNG images of NIfTI slices at specified depths.

        Args:
            surgical_target: Surgical target dictionary with trajectory information
            depths: List of depth values (in mm) to generate slices for

        Returns:
            Dictionary mapping depth (as string with 2 decimals) to base64-encoded PNG data
        """
        if not self.nifti_files:
            return {}

        slice_images = {}

        # Get trajectory information
        target = surgical_target['target']
        direction = surgical_target['direction']
        anterior = surgical_target['anterior']
        lateral = surgical_target['lateral']
        is_left = surgical_target.get('is_left_hemisphere', False)

        # Use first NIfTI file
        nifti_file = self.nifti_files[0]
        nifti_data = nifti_file.get('data')
        affine = nifti_file.get('affine')

        if nifti_data is None or affine is None:
            return {}

        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping NIfTI slice generation")
            return {}

        for depth in depths:
            # Calculate plane position
            plane_point = target - direction * depth

            # Resample the slice
            slice_data = NiftiSliceGenerator.resample_slice(
                nifti_data, affine, plane_point, anterior, lateral
            )

            if slice_data is None:
                continue

            # Flip for left hemisphere
            if is_left:
                slice_data = np.fliplr(slice_data)

            # Normalize intensity for better contrast
            # Use percentile-based normalization to avoid outliers
            vmin = np.percentile(slice_data, 2)  # 2nd percentile
            vmax = np.percentile(slice_data, 98)  # 98th percentile

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(4, 4), dpi=80)

            # Display the slice with normalized intensity
            half_size = NiftiSliceGenerator.SLICE_SIZE_MM / 2.0
            im = ax.imshow(
                slice_data,
                cmap='gray',
                extent=[-half_size, half_size, -half_size, half_size],
                origin='lower',
                interpolation='bilinear',
                vmin=vmin,
                vmax=vmax
            )

            # Add electrode intersection markers
            for i, electrode in enumerate(self.electrode_trajectories):
                trajectory = electrode['trajectory']
                intersection = NiftiSliceGenerator.find_plane_intersection(
                    trajectory, plane_point, direction
                )

                if intersection is not None:
                    lat_coord, ant_coord = NiftiSliceGenerator.project_to_plane_coords(
                        intersection, plane_point, anterior, lateral
                    )

                    # Flip lateral coordinate for left hemisphere
                    if is_left:
                        lat_coord = -lat_coord

                    # Get color matching polar plot
                    color = self._get_electrode_color(i)

                    # Plot electrode marker (1.27mm diameter circle)
                    circle = plt.Circle(
                        (lat_coord, ant_coord),
                        0.635,  # radius in mm (diameter 1.27mm)
                        color=color,
                        fill=False,
                        linewidth=1.5
                    )
                    ax.add_patch(circle)

            # Add MER track markers (2mm offset)
            # Note: Medial/Lateral are swapped compared to polar chart because
            # the slice coordinate system has opposite lateral convention
            # Determine which track should be red (clinical track or Central if none defined)
            clinical_track = surgical_target.get('clinical_track', '')
            red_track = clinical_track if clinical_track else 'Central'

            mer_offset_mm = 2.0
            mer_tracks = [
                {'name': 'Central', 'lat': 0, 'ant': 0, 'marker': '+', 'size': 3, 'width': 0.5},
                {'name': 'Anterior', 'lat': 0, 'ant': mer_offset_mm, 'marker': 'o', 'size': 2, 'width': 0.5},
                {'name': 'Posterior', 'lat': 0, 'ant': -mer_offset_mm, 'marker': 'o', 'size': 2, 'width': 0.5},
                {'name': 'Medial', 'lat': mer_offset_mm, 'ant': 0, 'marker': 'o', 'size': 2, 'width': 0.5},
                {'name': 'Lateral', 'lat': -mer_offset_mm, 'ant': 0, 'marker': 'o', 'size': 2, 'width': 0.5}
            ]

            for track in mer_tracks:
                lat = track['lat']
                ant = track['ant']

                # Flip lateral for left hemisphere
                if is_left:
                    lat = -lat

                # Color: red for clinical track (or Central if no clinical track), gray for others
                track_color = 'red' if track['name'] == red_track else 'gray'

                ax.plot(lat, ant, track['marker'], color=track_color,
                       markersize=track['size'], markeredgewidth=track['width'], alpha=0.8)

            # Add clinical target marker if defined and at this depth
            clinical_depth_str = surgical_target.get('clinical_depth', '')
            if clinical_depth_str:
                try:
                    clin_depth = float(clinical_depth_str)
                    # Only show marker if we're at this depth (within 0.5mm)
                    if abs(depth - clin_depth) < 0.5:
                        clin_track = clinical_track if clinical_track else 'Central'
                        # Get track position
                        track_map = {
                            'Central': (0, 0),
                            'Anterior': (0, mer_offset_mm),
                            'Posterior': (0, -mer_offset_mm),
                            'Lateral': (-mer_offset_mm, 0),  # Note: swapped in slice view
                            'Medial': (mer_offset_mm, 0)     # Note: swapped in slice view
                        }
                        if clin_track in track_map:
                            lat, ant = track_map[clin_track]
                            if is_left:
                                lat = -lat
                            # Draw clinical target marker (larger red circle)
                            ax.plot(lat, ant, 'o', color='#d62728', markersize=8,
                                   markeredgewidth=2, fillstyle='none', alpha=0.9)
                except (ValueError, TypeError):
                    pass

            # Add research target markers if defined and at this depth
            research_depth_str = surgical_target.get('research_depth', '')
            research_track_str = surgical_target.get('research_track', '')
            if research_depth_str:
                depth_strs = [d.strip() for d in research_depth_str.split(',') if d.strip()]
                track_strs = [t.strip() for t in research_track_str.split(',') if t.strip()] if research_track_str else []

                for idx, depth_str in enumerate(depth_strs):
                    try:
                        res_depth = float(depth_str)
                        # Only show marker if we're at this depth (within 0.5mm)
                        if abs(depth - res_depth) < 0.5:
                            res_track = track_strs[idx] if idx < len(track_strs) else 'Central'
                            res_track = res_track if res_track else 'Central'
                            # Get track position
                            track_map = {
                                'Central': (0, 0),
                                'Anterior': (0, mer_offset_mm),
                                'Posterior': (0, -mer_offset_mm),
                                'Lateral': (-mer_offset_mm, 0),  # Note: swapped in slice view
                                'Medial': (mer_offset_mm, 0)     # Note: swapped in slice view
                            }
                            if res_track in track_map:
                                lat, ant = track_map[res_track]
                                if is_left:
                                    lat = -lat
                                # Draw research target marker (larger orange circle)
                                ax.plot(lat, ant, 'o', color='#ff7f0e', markersize=8,
                                       markeredgewidth=2, fillstyle='none', alpha=0.9)
                    except (ValueError, TypeError):
                        continue

            # Remove labels and ticks to reduce file size
            ax.set_xlim(-half_size, half_size)
            ax.set_ylim(-half_size, half_size)
            ax.set_aspect('equal')
            ax.axis('off')  # Turn off axes for smaller file size

            # Save to PNG in memory with compression
            buf = io.BytesIO()
            plt.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                dpi=80,
                pad_inches=0,
                facecolor='black',
                pil_kwargs={'optimize': True, 'compress_level': 9}
            )
            plt.close(fig)
            buf.seek(0)

            # Encode as base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            slice_images[f'{depth:.2f}'] = img_base64

        return slice_images

    def generate_html(self) -> str:
        """Generate an HTML report with 3D plot and polar charts for each contact."""
        # Generate 3D figure from data (or use callback if provided)
        fig_3d = None
        if self.get_3d_figure_callback:
            # Use callback if provided (from visualization page)
            fig_3d = self.get_3d_figure_callback()
        elif self.surgical_targets or self.electrode_trajectories:
            # Generate 3D plot directly from parsed data
            plot_generator = Plot3DGenerator(
                surgical_targets=self.surgical_targets,
                electrode_trajectories=self.electrode_trajectories
            )
            fig_3d = plot_generator.generate_figure(
                show_mer_tracks=True,
                show_trajectories=True,
                show_targets=True,
                show_contacts=True,
                dark_mode=False  # Use light mode for reports
            )

        # Build report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Collect patient info, surgeons, and researchers from surgical targets
        patient_ids = set()
        surgeons_set = set()
        researchers_set = set()
        for target in self.surgical_targets:
            patient_ids.add(target.get('patient_id', 'Unknown'))

            # Collect surgeons and researchers (comma-separated initials)
            surgeons = target.get('surgeons', '')
            if surgeons:
                surgeons_set.update([s.strip() for s in surgeons.split(',')])

            researchers = target.get('researchers', '')
            if researchers:
                researchers_set.update([r.strip() for r in researchers.split(',')])

        patient_info = self.patient_id if self.patient_id else (', '.join(patient_ids) if patient_ids else 'Unknown')
        surgeons_info = ', '.join(sorted(surgeons_set)) if surgeons_set else 'N/A'
        researchers_info = ', '.join(sorted(researchers_set)) if researchers_set else 'N/A'

        # Start building HTML
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>DBS Surgical Report</title>',
            '    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; display: flex; justify-content: center; }',
            '        .container { max-width: 1580px; width: 100%; }',
            '        .header { position: relative; text-align: center; margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .header-disclaimer { position: absolute; top: 14px; right: 18px; font-size: 16px; color: #b57a00; max-width: 330px; text-align: right; line-height: 1.4; padding: 9px 15px; background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px; }',
            '        .header-disclaimer strong { color: #e65100; }',
            '        .header h1 { margin: 0 0 10px 0; color: #333; }',
            '        .header p { margin: 5px 0; color: #666; }',
            '        .tabs { display: flex; gap: 0; margin-bottom: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }',
            '        .tab { flex: 1; padding: 15px 20px; text-align: center; cursor: pointer; background: #f5f5f5; border: none; font-size: 16px; font-weight: 600; color: #666; transition: all 0.3s; }',
            '        .tab:hover:not(.disabled) { background: #e0e0e0; }',
            '        .tab.active { background: white; color: #636efa; border-bottom: 3px solid #636efa; }',
            '        .tab.disabled { cursor: not-allowed; opacity: 0.4; }',
            '        .tab-content { display: none; }',
            '        .tab-content.active { display: block; }',
            '        .electrodes-container { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; justify-content: space-between; }',
            '        .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 0 1 auto; }',
            '        .section.full-width { flex: 1 1 100%; margin-bottom: 20px; }',
            '        .section h2 { margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }',
            '        .section > p { margin: 10px 0; color: #666; }',
            '        .target-content { display: flex; gap: 20px; }',
            '        .left-column { flex: 0 0 400px; display: flex; flex-direction: column; }',
            '        .right-column { flex: 0 0 auto; display: flex; flex-direction: column; gap: 15px; }',
            '        .brain-image-container { position: relative; margin-bottom: 20px; }',
            '        #brain_view_0 img, #brain_view_1 img, #brain_view_2 img, #brain_view_3 img { width: 100%; height: auto; border-radius: 8px; }',
            '        #slice_view_0 img, #slice_view_1 img, #slice_view_2 img, #slice_view_3 img { width: 100%; height: auto; border-radius: 8px; }',
            '        .brain-image-container svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }',
            '        .hemisphere-label { position: absolute; font-weight: bold; font-size: 24px; color: #333; bottom: 20%; }',
            '        .polar-chart { background: #fafafa; border-radius: 8px; padding: 10px; width: 300px; height: 300px; }',
            '        .contact-buttons { display: flex; flex-direction: column; gap: 8px; margin-bottom: 15px; }',
            '        .contact-button { display: flex; align-items: center; gap: 10px; padding: 10px 15px; border: 2px solid #ddd; border-radius: 6px; background: white; cursor: pointer; transition: all 0.2s; }',
            '        .contact-button:hover { background: #f8f8f8; border-color: #bbb; }',
            '        .contact-button.active { border-color: #636efa; background: #f0f4ff; }',
            '        .contact-button .label { font-weight: 600; min-width: 30px; }',
            '        .contact-button .depth { color: #666; min-width: 70px; }',
            '        .contact-button .distance { color: #888; font-size: 0.9em; }',
            '        .contact-button .tooltip-info { color: #aaa; font-size: 0.85em; }',
            '        .research-section { margin-top: 15px; padding-top: 15px; border-top: 2px solid #eee; }',
            '        .research-header { font-size: 14px; font-weight: 600; color: #333; margin-bottom: 8px; }',
            '        .research-buttons { display: flex; flex-direction: column; gap: 8px; }',
            '        .research-button { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border: 2px solid #ddd; border-radius: 6px; background: #fafafa; cursor: pointer; transition: all 0.2s; }',
            '        .research-button:hover { background: #f0f0f0; border-color: #bbb; }',
            '        .research-button.active { border-color: #ff7f0e; background: #fff4e6; }',
            '        .research-button .label { font-weight: 600; min-width: 30px; font-size: 0.9em; }',
            '        .research-button .depth { color: #666; min-width: 70px; font-size: 0.9em; }',
            '        .research-button .track { color: #888; font-size: 0.85em; }',
            '        .clinical-section { margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; }',
            '        .clinical-header { font-size: 14px; font-weight: 600; color: #d62728; margin-bottom: 8px; }',
            '        .clinical-button { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border: 2px solid #d62728; border-radius: 6px; background: #fff0f0; cursor: pointer; transition: all 0.2s; }',
            '        .clinical-button:hover { background: #ffe6e6; }',
            '        .clinical-button.active { border-color: #d62728; background: #ffcccc; }',
            '        .clinical-button .label { font-weight: 600; min-width: 30px; font-size: 0.9em; color: #d62728; }',
            '        .clinical-button .depth { color: #666; min-width: 70px; font-size: 0.9em; }',
            '        .clinical-button .track { color: #888; font-size: 0.85em; }',
            '        .coord-toggle { padding: 8px 16px; border: 2px solid #ddd; border-radius: 6px; background: white; cursor: pointer; transition: all 0.2s; font-size: 14px; font-weight: 500; }',
            '        .coord-toggle:hover { background: #f8f8f8; border-color: #bbb; }',
            '        .coord-toggle.active { border-color: #636efa; background: #f0f4ff; color: #636efa; }',
            '        .depth-slider-container { margin-top: 15px; padding: 15px; background: #f8f8f8; border-radius: 6px; }',
            '        .depth-slider-container label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }',
            '        .depth-slider { width: 100%; }',
            '        .depth-value { display: inline-block; margin-left: 10px; font-weight: 600; color: #333; }',
            '        .position-stats { margin-top: 15px; padding: 12px; background: #f8f8f8; border-radius: 6px; font-size: 14px; }',
            '        .position-stats .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; }',
            '        .position-stats .stat-row:last-child { margin-bottom: 0; }',
            '        .position-stats .stat-label { font-weight: 600; color: #555; }',
            '        .position-stats .stat-value { color: #333; }',
            '        .electrode-label { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }',
            '        .toggle-image-button { position: absolute; bottom: 8px; right: 8px; padding: 6px 12px; background: rgba(99, 110, 250, 0.9); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; transition: background 0.2s; z-index: 10; }',
            '        .toggle-image-button:hover { background: rgba(76, 82, 217, 0.95); }',
            '        .zoom-button { position: absolute; top: 8px; right: 8px; padding: 6px 12px; background: rgba(99, 110, 250, 0.9); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; transition: background 0.2s; z-index: 10; }',
            '        .zoom-button:hover { background: rgba(76, 82, 217, 0.95); }',
            '        .brain-image-container { position: relative; }',
            '        #slice_view_0 img.zoomed, #slice_view_1 img.zoomed, #slice_view_2 img.zoomed, #slice_view_3 img.zoomed { transform: scale(2); }',
            '        @media print { body { background: white; } .section { box-shadow: none; border: 1px solid #ddd; } }',
            '    </style>',
            '</head>',
            '<body>',
            '    <div class="container">',
            '        <div class="header">',
            '            <div class="header-disclaimer"><strong>&#9888; Research Use Only</strong><br>Not cleared for clinical or diagnostic use</div>',
            '            <h1>DBS Surgical Report</h1>',
            f'            <p><strong>Patient:</strong> {patient_info}</p>',
            f'            <p><strong>Surgeons:</strong> {surgeons_info}</p>',
            f'            <p><strong>Researchers:</strong> {researchers_info}</p>',
            f'            <p><strong>Generated:</strong> {report_date}</p>',
            f'            <p>Generated with <a href="https://github.com/mvpetersen/dbs-toolbox" style="color: #2196F3;">The DBS Toolbox v{__version__}</a></p>',
            '        </div>',
        ]

        # Check if brain shift data is available
        has_brain_shift = self.electrode_trajectories_2 is not None and len(self.electrode_trajectories_2) > 0
        brain_shift_class = '' if has_brain_shift else ' disabled'

        html_parts.extend([
            '        <div class="tabs">',
            '            <button class="tab active" onclick="switchTab(\'stereotactic\')">Stereotactic Targeting</button>',
            f'            <button class="tab{brain_shift_class}" onclick="switchTab(\'brainshift\')">Brain Shift</button>',
            '        </div>',
            '        <div id="stereotactic-tab" class="tab-content active">',
        ])

        # Add metadata section at the top of stereotactic tab
        html_parts.extend([
            '        <div class="section full-width">',
            '            <h2>Source Data</h2>',
            '            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">',
            '                <div>',
            '                    <h3 style="color: #2196F3;">Surgical Planning</h3>',
            '                    <table style="width: 100%; border-collapse: collapse;">',
        ])

        # Add surgical data info
        surgical_filename = self.surgical_targets[0]['filename'] if self.surgical_targets else 'N/A'
        num_targets = len(self.surgical_targets)
        html_parts.extend([
            '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>File:</strong></td>',
            f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{surgical_filename}</td></tr>',
            '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Number of Targets:</strong></td>',
            f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{num_targets}</td></tr>',
        ])

        html_parts.extend([
            '                    </table>',
            '                </div>',
            '                <div>',
            '                    <h3 style="color: #FF5722;">Electrode Reconstruction</h3>',
            '                    <table style="width: 100%; border-collapse: collapse;">',
        ])

        # Add electrode metadata
        if self.electrode_metadata_1:
            ct_file = self.electrode_metadata_1.get('ct_file', 'N/A')
            timestamp = self.electrode_metadata_1.get('timestamp', 'N/A')
            pypacer_version = self.electrode_metadata_1.get('pypacer_version', 'N/A')
            voxel_sizes = self.electrode_metadata_1.get('voxel_sizes_mm', [])
            ct_shape = self.electrode_metadata_1.get('ct_volume_shape', [])

            html_parts.extend([
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>CT File:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{ct_file}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Timestamp:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{timestamp}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>PyPaCER Version:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{pypacer_version}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Voxel Sizes (mm):</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{" × ".join(map(str, voxel_sizes)) if voxel_sizes else "N/A"}</td></tr>',
                '                        <tr><td style="padding: 5px;"><strong>CT Volume Shape:</strong></td>',
                f'                            <td style="padding: 5px;">{" × ".join(map(str, ct_shape)) if ct_shape else "N/A"}</td></tr>',
            ])
        else:
            html_parts.append('                        <tr><td colspan="2" style="padding: 10px; text-align: center; color: #999;">No metadata available</td></tr>')

        html_parts.extend([
            '                    </table>',
            '                </div>',
            '            </div>',
            '        </div>',
        ])

        # Store 3D plot HTML to add later
        plot_3d_html = None

        # Generate 3D visualization section (will be added after electrode cards)
        if fig_3d:
            # Calculate average position of all contacts for camera centering
            all_contact_positions = []
            for target_idx, target in enumerate(self.surgical_targets):
                # Find closest electrode for this target
                closest_electrode = self._find_closest_electrode_for_target(target_idx)
                if closest_electrode and closest_electrode.get('contacts') is not None:
                    contacts = closest_electrode['contacts']
                    for contact in contacts:
                        if 'position' in contact and contact['position'] is not None:
                            all_contact_positions.append(contact['position'])

            if all_contact_positions:
                all_contact_positions = np.array(all_contact_positions)
                center_x = float(np.mean(all_contact_positions[:, 0]))
                center_y = float(np.mean(all_contact_positions[:, 1]))
                center_z = float(np.mean(all_contact_positions[:, 2]))
            else:
                center_x = 0
                center_y = 0
                center_z = 0

            # Modify 3D figure for report (lighter theme)
            fig_3d_copy = go.Figure(fig_3d)

            # Get current aspect ratio to apply zoom for orthographic projection
            if fig_3d_copy.layout.scene.aspectratio:
                x_aspect = fig_3d_copy.layout.scene.aspectratio.x or 1
                y_aspect = fig_3d_copy.layout.scene.aspectratio.y or 1
                z_aspect = fig_3d_copy.layout.scene.aspectratio.z or 1
            else:
                x_aspect = 1
                y_aspect = 1
                z_aspect = 1

            # Set zoom factor
            zoom_factor = 2.5

            fig_3d_copy.update_layout(
                height=600,
                paper_bgcolor='white',
                plot_bgcolor='white',
                scene=dict(
                    bgcolor='white',
                    xaxis=dict(backgroundcolor='white', gridcolor='#ddd', showbackground=True),
                    yaxis=dict(backgroundcolor='white', gridcolor='#ddd', showbackground=True),
                    zaxis=dict(backgroundcolor='white', gridcolor='#ddd', showbackground=True),
                    camera=dict(
                        projection=dict(type='orthographic'),
                        eye=dict(x=0, y=2.5, z=0),  # Looking along y-axis from front
                        up=dict(x=0, y=0, z=1)  # Z-axis is up
                    ),
                    aspectratio=dict(
                        x=x_aspect * zoom_factor,
                        y=y_aspect * zoom_factor,
                        z=z_aspect * zoom_factor
                    ),
                    aspectmode='manual'
                ),
                font=dict(color='#333'),
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', font=dict(color='#333'))
            )

            # Store 3D plot HTML to add after electrode cards
            plot_3d_html = [
                '        <div class="section full-width">',
                '            <h2>3D Visualization</h2>',
                f'            <div id="plot3d"></div>',
                '            <script>',
                f'                var data3d = {fig_3d_copy.to_json()};',
                '                Plotly.newPlot("plot3d", data3d.data, data3d.layout, {responsive: true});',
                '            </script>',
                '        </div>',
            ]

        # Add electrode sections (each gets its own card, displayed side-by-side)
        html_parts.append('        <div class="electrodes-container">')

        # Sort targets so right hemisphere appears before left hemisphere (right on left side visually)
        sorted_target_indices = sorted(
            range(len(self.surgical_targets)),
            key=lambda idx: (
                0 if 'right' in self.surgical_targets[idx]['hemisphere'].lower() or
                     self.surgical_targets[idx]['hemisphere'].lower() == 'r'
                else 1
            )
        )

        plot_counter = 0
        for target_idx in sorted_target_indices:
            surgical_target = self.surgical_targets[target_idx]
            # Find closest electrode for this target
            closest_electrode = self._find_closest_electrode_for_target(target_idx)
            if not closest_electrode or closest_electrode.get('contacts') is None:
                continue

            # Get electrode index for color matching
            electrode_label = closest_electrode.get('label', '')
            electrode_idx = target_idx  # Default to target index
            for idx, electrode in enumerate(self.electrode_trajectories):
                if electrode.get('label') == electrode_label:
                    electrode_idx = idx
                    break

            # Get electrode color from 3D plot palette
            electrode_color = self._get_electrode_color(electrode_idx)

            # Get contact depths
            contact_depths = self._calculate_contact_depths(surgical_target, closest_electrode)

            # Target header
            patient_id = surgical_target['patient_id']
            hemisphere = surgical_target['hemisphere']
            anat_target = surgical_target['anatomical_target']
            target_title = f"{patient_id} {hemisphere} {anat_target}".strip()

            # Get brain image
            brain_image_base64 = self._get_brain_image_base64()

            # Determine hemisphere positioning for label
            hemisphere_lower = hemisphere.lower()
            if 'left' in hemisphere_lower or hemisphere_lower == 'l':
                label_position_bottom = 'right: 10%; bottom: 20%;'
                label_position_top = 'right: 10%; top: 10%;'
                label_text = 'Left'
            else:
                label_position_bottom = 'left: 10%; bottom: 20%;'
                label_position_top = 'left: 10%; top: 10%;'
                label_text = 'Right'

            # Calculate trajectory angle in frontal plane (x-z) for visualization
            # The trajectory goes from entry (top/outside) to target (center/deep)
            entry = surgical_target.get('entry')
            target = surgical_target.get('target')

            if entry is not None and target is not None:
                # Calculate trajectory direction in frontal (x-z) plane
                # x: left-right in RAS coordinates, z: inferior-superior
                dx = entry[0] - target[0]  # Positive = entry to the right in RAS
                dz = entry[2] - target[2]  # Positive = entry superior

                # Flip dx for frontal view (left hemisphere appears on right side of image)
                dx = -dx

                # Calculate starting x position based on target location
                # Target x position determines left vs right hemisphere
                # Negative x = left hemisphere (appears on right side of frontal view)
                # Positive x = right hemisphere (appears on left side of frontal view)
                target_x = target[0]

                # Offset from center: scale target x-coordinate to percentage
                # Assume ±60mm is full width of relevant brain area
                x_offset_pct = -(target_x / 60.0) * 20  # Flip and scale to ±20% max offset
                start_x = 50 + x_offset_pct
                start_y = 50  # Center vertically (approximate target depth)

                # Calculate angle and line endpoint
                # Normalize to get unit direction
                frontal_length = np.sqrt(dx**2 + dz**2)
                if frontal_length > 0.01:  # Avoid division by zero
                    # Scale the line: assume ~80mm from center to top of brain
                    # and scale based on actual trajectory length
                    line_length_pct = min(40, frontal_length * 0.5)  # 40% max length

                    # Calculate endpoint (going from start upward along trajectory)
                    end_x = start_x + (dx / frontal_length) * line_length_pct
                    end_y = start_y - (dz / frontal_length) * line_length_pct  # Negative because SVG y increases downward
                else:
                    # Vertical trajectory
                    end_x = start_x
                    end_y = start_y - 40
            else:
                # Default vertical line if no entry/target data
                start_x = 50
                start_y = 50
                end_x = 50
                end_y = 10

            # Get surgical coordinates (ring, arc, x, y, z)
            ring = surgical_target.get('ring_original', surgical_target.get('ring', 'N/A'))
            arc = surgical_target.get('arc_original', surgical_target.get('arc', 'N/A'))
            x_orig = surgical_target.get('x_original', 'N/A')
            y_orig = surgical_target.get('y_original', 'N/A')
            z_orig = surgical_target.get('z_original', 'N/A')
            notes = surgical_target.get('notes', '')

            # Start new section card for this electrode
            html_parts.extend([
                '            <div class="section">',
                f'                <h2>{target_title}</h2>',
                f'                <p>Electrode: {closest_electrode.get("label", "Unknown")}</p>',
                f'                <p><strong>Surgical Trajectory:</strong> Ring: {ring}°, Arc: {arc}°, X: {x_orig}mm, Y: {y_orig}mm, Z: {z_orig}mm</p>',
                '                <div class="target-content">',
                '                    <div class="left-column">',
            ])

            # Add brain image with trajectory - use NIfTI trajectory plane if available
            # Generate trajectory plane slice from NIfTI if available
            trajectory_slice_base64 = None
            if self.nifti_files:
                print(f"Generating trajectory plane slice for target {target_idx}...")
                trajectory_slice_base64 = self._generate_trajectory_plane_slice(surgical_target, target_idx)

            # Use NIfTI trajectory slice if available, otherwise fall back to static brain image
            if trajectory_slice_base64:
                brain_view_image = trajectory_slice_base64
                brain_view_opacity = 1.0  # Full opacity for NIfTI data
                # Use white label at top for NIfTI slice
                label_position = label_position_top
                label_color = 'white'
                use_svg_overlay = False  # NIfTI slice has trajectory baked in
            elif brain_image_base64:
                brain_view_image = brain_image_base64
                brain_view_opacity = 0.2  # Low opacity for static image
                # Use dark label at bottom for static image
                label_position = label_position_bottom
                label_color = '#333'
                use_svg_overlay = True  # Need SVG trajectory line for static image
            else:
                brain_view_image = None

            if brain_view_image:
                has_slices = bool(self.nifti_files)
                toggle_button_html = ''
                if has_slices:
                    toggle_button_html = f'<button class="toggle-image-button" id="toggle_btn_{target_idx}" onclick="toggleImageView({target_idx})">Show Probe View</button>'

                # Add zoom button for slice view if slices are available
                zoom_button_html = ''
                if has_slices:
                    zoom_button_html = f'<button class="zoom-button" id="zoom_btn_{target_idx}" onclick="toggleZoom({target_idx})">Zoom In</button>'

                # Build brain view with optional SVG overlay
                brain_view_parts = [
                    f'                    <div class="brain-image-container" id="image_container_{target_idx}">',
                    f'                        <div id="brain_view_{target_idx}" style="position: relative; width: 100%; height: 100%;">',
                    f'                            <img src="data:image/png;base64,{brain_view_image}" alt="Brain View" style="width: 100%; height: auto; opacity: {brain_view_opacity};">',
                ]

                # Add SVG trajectory line overlay for static brain image
                if use_svg_overlay:
                    brain_view_parts.extend([
                        '                            <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">',
                        f'                                <line x1="{start_x:.1f}" y1="{start_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" ',
                        f'                                      stroke="{electrode_color}" stroke-width="2" stroke-linecap="round"/>',
                        '                            </svg>',
                    ])

                brain_view_parts.extend([
                    f'                            <div class="hemisphere-label" style="{label_position} color: {label_color};">{label_text}</div>',
                    '                        </div>',
                    f'                        <div id="slice_view_{target_idx}" style="display: none; width: 100%; text-align: center; position: relative; overflow: hidden;">',
                    f'                            {zoom_button_html}',
                    f'                            <img id="nifti_img_{target_idx}" src="" alt="NIfTI Slice" style="max-width: 100%; border-radius: 6px; transition: transform 0.3s ease;">',
                    '                        </div>',
                    f'                        {toggle_button_html}',
                    '                    </div>',
                ])

                html_parts.extend(brain_view_parts)

            # Add contact buttons in left column
            html_parts.append('                    <div class="contact-buttons">')

            contact_data_list = []
            for contact_idx in [3, 2, 1, 0]:
                depth = contact_depths[contact_idx]
                if depth is None:
                    continue

                # Get electrode position at this depth (filtered to matched electrode)
                positions = self._get_electrode_positions_at_depth(target_idx, depth)
                matched_label = closest_electrode.get('label')
                matched_positions = [pos for pos in positions if pos.get('label') == matched_label or pos.get('short_label') == matched_label]

                if matched_positions:
                    pos = matched_positions[0]
                    euclidean_dist = pos['r']
                    ap_offset = pos['ant_coord']
                    ml_offset = pos.get('lat_coord_stats', pos['lat_coord'])

                    ap_dir = 'Ant' if ap_offset >= 0 else 'Post'
                    ml_dir = 'Med' if ml_offset >= 0 else 'Lat'

                    tooltip_info = f'A-P: {ap_offset:+.2f}mm ({ap_dir}) | M-L: {ml_offset:+.2f}mm ({ml_dir})'

                    active_class = ' active' if contact_idx == 3 else ''  # C3 active by default

                    html_parts.append(
                        f'                        <div class="contact-button{active_class}" onclick="setDepthForTarget({target_idx}, {depth:.2f})">'
                        f'<span class="label">C{contact_idx}</span>'
                        f'<span class="depth">{depth:+.2f}mm</span>'
                        f'<span class="distance">{euclidean_dist:.2f}mm</span>'
                        f'<span class="tooltip-info">{tooltip_info}</span>'
                        f'</div>'
                    )

                    contact_data_list.append({
                        'contact_idx': contact_idx,
                        'depth': depth
                    })

            html_parts.append('                    </div>')  # Close contact-buttons

            # Add clinical target button if defined
            clinical_depth = surgical_target.get('clinical_depth', '')
            clinical_track = surgical_target.get('clinical_track', '')

            if clinical_depth:
                try:
                    clin_depth = float(clinical_depth)
                    clin_track_display = clinical_track if clinical_track else 'Central'

                    html_parts.extend([
                        '                    <div class="clinical-section">',
                        '                        <div class="clinical-header">Clinical Target</div>',
                        f'                        <div class="clinical-button" onclick="setDepthForTarget({target_idx}, {clin_depth:.2f})">'
                        f'<span class="label">CLIN</span>'
                        f'<span class="depth">{clin_depth:+.2f}mm</span>'
                        f'<span class="track">{clin_track_display}</span>'
                        f'</div>',
                        '                    </div>'
                    ])
                except (ValueError, TypeError):
                    pass

            # Add research target buttons if defined
            research_depth = surgical_target.get('research_depth', '')
            research_track = surgical_target.get('research_track', '')

            if research_depth:
                # Parse comma-separated depths
                depth_strs = [d.strip() for d in research_depth.split(',') if d.strip()]
                # Parse comma-separated tracks (or empty list if not provided)
                track_strs = [t.strip() for t in research_track.split(',') if t.strip()] if research_track else []

                if depth_strs:
                    html_parts.extend([
                        '                    <div class="research-section">',
                        '                        <div class="research-header">Research Targets</div>',
                        '                        <div class="research-buttons">'
                    ])

                    for idx, depth_str in enumerate(depth_strs):
                        try:
                            res_depth = float(depth_str)
                            # Get corresponding track or default to Central
                            res_track = track_strs[idx] if idx < len(track_strs) else 'Central'
                            res_track_display = res_track if res_track else 'Central'

                            html_parts.append(
                                f'                            <div class="research-button" onclick="setDepthForTarget({target_idx}, {res_depth:.2f})">'
                                f'<span class="label">R{idx+1}</span>'
                                f'<span class="depth">{res_depth:+.2f}mm</span>'
                                f'<span class="track">{res_track_display}</span>'
                                f'</div>'
                            )
                        except (ValueError, TypeError):
                            continue

                    html_parts.extend([
                        '                        </div>',
                        '                    </div>'
                    ])

            html_parts.extend([
                '                </div>',  # Close left-column
            ])

            html_parts.append('                <div class="right-column">')

            # Create single polar chart with compact data for dynamic generation
            # Calculate depth range based on electrode trajectory
            trajectory = closest_electrode.get('trajectory')
            target = surgical_target.get('target')
            direction = surgical_target.get('direction')

            if trajectory is not None and target is not None and direction is not None:
                # Calculate depth for each trajectory point
                depths = []
                for point in trajectory:
                    diff = point - target
                    depth = -float(np.dot(diff, direction))
                    depths.append(depth)
                depth_min = round(min(depths), 1)
                depth_max = round(max(depths), 1)
            else:
                # Fallback to default range
                depth_min = -80.0
                depth_max = 10.0
            depth_step = 0.5

            # Store compact position data for all depths
            position_data = {}

            # First, add exact contact depths to ensure we have precise data for those positions
            contact_depths_to_calculate = set()
            for contact_idx in [3, 2, 1, 0]:
                if contact_depths[contact_idx] is not None:
                    contact_depths_to_calculate.add(contact_depths[contact_idx])

            # Then add the regular 0.5mm increment depths
            depths_range = np.arange(depth_min, depth_max + depth_step, depth_step)
            all_depths = sorted(set(depths_range.tolist()) | contact_depths_to_calculate)

            for depth in all_depths:
                # Force recalculation instead of using precalculated positions
                # This ensures the new flipping logic is applied
                positions = []
                for i, electrode in enumerate(self.electrode_trajectories):
                    # Use the same color palette as 3D plot
                    color = self._get_electrode_color(i)
                    pos = self._calculate_electrode_position_at_depth(surgical_target, electrode, depth)
                    if pos:
                        pos['color'] = color
                        positions.append(pos)

                matched_label = closest_electrode.get('label')
                matched_positions = [pos for pos in positions if pos.get('label') == matched_label or pos.get('short_label') == matched_label]

                # Only store if we have position data
                if matched_positions:

                    # Store compact position data
                    position_data[f'{depth:.2f}'] = [{
                        'theta': float(pos['theta']),
                        'r': float(pos['r']),
                        'label': pos.get('label', ''),
                        'short_label': pos.get('short_label', ''),
                        'color': pos['color'],
                        'ant': float(pos['ant_coord']),
                        'lat': float(pos.get('lat_coord_stats', pos['lat_coord'])),
                        'x': float(pos.get('x', 0)),
                        'y': float(pos.get('y', 0)),
                        'z': float(pos.get('z', 0))
                    } for pos in matched_positions]

            # Generate NIfTI slice images if available
            slice_images = {}
            if self.nifti_files:
                print(f"Generating NIfTI slices for target {target_idx}...")
                slice_images = self._generate_nifti_slice_images(surgical_target, all_depths, target_idx)
                print(f"Generated {len(slice_images)} slice images")

            # Store MER track configuration
            is_left = surgical_target.get('is_left_hemisphere', False)
            # Use the same color as 3D plot for surgical trajectory
            surgical_color = self._get_color(target_idx)
            clinical_track = surgical_target.get('clinical_track', '')
            clinical_depth = surgical_target.get('clinical_depth', '')
            research_depth = surgical_target.get('research_depth', '')
            research_track = surgical_target.get('research_track', '')

            # Parse clinical target
            clinical_targets = []
            if clinical_depth:
                try:
                    clin_depth = float(clinical_depth)
                    clin_track = clinical_track if clinical_track else 'Central'
                    clinical_targets.append({'depth': clin_depth, 'track': clin_track})
                except (ValueError, TypeError):
                    pass

            # Parse research targets
            research_targets = []
            if research_depth:
                depth_strs = [d.strip() for d in research_depth.split(',') if d.strip()]
                track_strs = [t.strip() for t in research_track.split(',') if t.strip()] if research_track else []
                for idx, depth_str in enumerate(depth_strs):
                    try:
                        res_depth = float(depth_str)
                        res_track = track_strs[idx] if idx < len(track_strs) else 'Central'
                        res_track = res_track if res_track else 'Central'
                        research_targets.append({'depth': res_depth, 'track': res_track})
                    except (ValueError, TypeError):
                        continue

            mer_config = {
                'is_left': is_left,
                'surgical_color': surgical_color,
                'offset_mm': 2.0,
                'clinical_track': clinical_track,
                'clinical_targets': clinical_targets,
                'research_targets': research_targets
            }

            # Default depth (C3 contact)
            default_depth = contact_depths[3] if contact_depths[3] is not None else 0.0

            div_id = f'polar_{target_idx}'
            slider_id = f'depth_slider_{target_idx}'

            html_parts.extend([
                f'                    <div class="polar-chart">',
                f'                        <div id="{div_id}"></div>',
                '                    </div>',
                '                    <div class="depth-slider-container">',
                f'                        <label for="{slider_id}">Depth: <span id="{slider_id}_value" class="depth-value">{default_depth:+.2f} mm</span></label>',
                f'                        <input type="range" id="{slider_id}" class="depth-slider" ',
                f'                               min="{depth_min}" max="{depth_max}" step="{depth_step}" value="{default_depth:.2f}" ',
                f'                               oninput="onSliderInput({target_idx}, this.value)">',
                '                    </div>',
                f'                    <div class="position-stats" id="stats_{target_idx}">',
                '                        <div class="stat-row">',
                '                            <span class="stat-label">Distance:</span>',
                f'                            <span class="stat-value" id="stats_{target_idx}_distance">--</span>',
                '                        </div>',
                '                        <div class="stat-row">',
                '                            <span class="stat-label">A-P Offset:</span>',
                f'                            <span class="stat-value" id="stats_{target_idx}_ap">--</span>',
                '                        </div>',
                '                        <div class="stat-row">',
                '                            <span class="stat-label">M-L Offset:</span>',
                f'                            <span class="stat-value" id="stats_{target_idx}_ml">--</span>',
                '                        </div>',
                '                        <div class="stat-row">',
                '                            <span class="stat-label">Position (X,Y,Z):</span>',
                f'                            <span class="stat-value" id="stats_{target_idx}_xyz">--</span>',
                '                        </div>',
                '                    </div>',
            ])

            html_parts.extend([
                '                    <script>',
                f'                        // Store position data and MER config for target {target_idx}',
                f'                        window.positionDataTarget{target_idx} = {json.dumps(position_data)};',
                f'                        window.merConfigTarget{target_idx} = {json.dumps(mer_config)};',
            ])

            # Store slice images if available
            if slice_images:
                html_parts.append(f'                        window.sliceImagesTarget{target_idx} = {json.dumps(slice_images)};')

            html_parts.extend([
                f'                        // Store initialization params for later',
                f'                        if (!window.chartsToInit) window.chartsToInit = [];',
                f'                        window.chartsToInit.push({{targetIdx: {target_idx}, divId: "{div_id}", depth: {default_depth:.2f}}});',
                '                    </script>',
            ])

            html_parts.extend([
                '                    </div>',  # Close right-column
                '                </div>',  # Close target-content
            ])

            # Add notes at the bottom if present
            if notes:
                html_parts.append(f'                <p style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;"><strong>Notes:</strong> {notes}</p>')

            html_parts.append('            </div>')  # Close section

        html_parts.extend([
            '        </div>',  # Close electrodes-container
        ])

        # Add 3D plot after electrode cards
        if plot_3d_html:
            html_parts.extend(plot_3d_html)

        html_parts.extend([
            '        </div>',  # Close stereotactic-tab
            '        <div id="brainshift-tab" class="tab-content">',
        ])

        # Generate brain shift tab content
        if has_brain_shift:
            analyzer = BrainShiftAnalyzer(
                self.electrode_trajectories,
                self.electrode_trajectories_2,
                metadata_1=self.electrode_metadata_1,
                metadata_2=self.electrode_metadata_2
            )
            brain_shift_html = analyzer.generate_html_section()
            html_parts.append(brain_shift_html)
        else:
            html_parts.extend([
                '            <div class="section full-width">',
                '                <p style="text-align: center; color: #999;">No brain shift data available. Load two electrode reconstructions to enable this analysis.</p>',
                '            </div>',
            ])

        html_parts.extend([
            '        </div>',  # Close brainshift-tab
            '    </div>',  # Close container
            '    <script>',
            '        // Tab switching function',
            '        function switchTab(tabName) {',
            '            // Only allow switching to enabled tabs',
            '            const targetTab = document.querySelector(`.tab[onclick*="${tabName}"]`);',
            '            if (targetTab && targetTab.classList.contains("disabled")) {',
            '                return;',
            '            }',
            '            ',
            '            // Hide all tab contents',
            '            document.querySelectorAll(".tab-content").forEach(content => {',
            '                content.classList.remove("active");',
            '            });',
            '            ',
            '            // Deactivate all tabs',
            '            document.querySelectorAll(".tab").forEach(tab => {',
            '                tab.classList.remove("active");',
            '            });',
            '            ',
            '            // Show selected tab content',
            '            const tabContent = document.getElementById(tabName + "-tab");',
            '            if (tabContent) {',
            '                tabContent.classList.add("active");',
            '            }',
            '            ',
            '            // Activate selected tab button',
            '            if (targetTab) {',
            '                targetTab.classList.add("active");',
            '            }',
            '            ',
            '            // Resize Plotly charts when switching to brain shift tab',
            '            if (tabName === "brainshift") {',
            '                setTimeout(function() {',
            '                    const plot = document.getElementById("plot3d_brainshift");',
            '                    if (plot && window.Plotly) {',
            '                        Plotly.Plots.resize(plot);',
            '                    }',
            '                }, 100);',
            '            }',
            '        }',
            '        ',
            '        // Function to create polar chart traces from position data',
            '        function createPolarChartTraces(positions, merConfig) {',
            '            const traces = [];',
            '            ',
            '            // Determine which track should be red',
            '            // If clinical_track is defined, use it; otherwise default to "Central"',
            '            const redTrack = merConfig.clinical_track || "Central";',
            '            ',
            '            // Add MER track markers',
            '            const merTracks = [',
            '                {name: "Central", lat: 0, ant: 0, symbol: "cross"},',
            '                {name: "Anterior", lat: 0, ant: merConfig.offset_mm, symbol: "cross"},',
            '                {name: "Posterior", lat: 0, ant: -merConfig.offset_mm, symbol: "cross"},',
            '                {name: "Lateral", lat: merConfig.offset_mm, ant: 0, symbol: "cross"},',
            '                {name: "Medial", lat: -merConfig.offset_mm, ant: 0, symbol: "cross"}',
            '            ];',
            '            ',
            '            merTracks.forEach(track => {',
            '                let lat = track.lat;',
            '                let ant = track.ant;',
            '                ',
            '                // Flip lateral for right hemisphere so medial points toward midline',
            '                if (!merConfig.is_left) {',
            '                    lat = -lat;',
            '                }',
            '                ',
            '                const r = Math.sqrt(lat * lat + ant * ant);',
            '                const theta = Math.atan2(lat, ant) * 180 / Math.PI;',
            '                ',
            '                // Determine color: red for clinical track (or Central if no clinical track), gray for others',
            '                const trackColor = (track.name === redTrack) ? "red" : "gray";',
            '                ',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [r],',
            '                    theta: [theta],',
            '                    mode: "markers",',
            '                    marker: {size: 10, color: trackColor, symbol: track.symbol},',
            '                    name: track.name,',
            '                    hovertemplate: track.name + "<br>L: " + lat.toFixed(1) + "mm, A: " + ant.toFixed(1) + "mm<extra></extra>",',
            '                    showlegend: false',
            '                });',
            '            });',
            '            ',
            '            // Add electrode positions',
            '            positions.forEach(pos => {',
            '                // Grey ring',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [pos.r],',
            '                    theta: [pos.theta],',
            '                    mode: "markers",',
            '                    marker: {',
            '                        size: 29,',
            '                        color: "rgba(128, 128, 128, 0.3)",',
            '                        symbol: "circle",',
            '                        line: {color: "grey", width: 2}',
            '                    },',
            '                    hovertemplate: pos.label + "<br>Electrode (1.27mm)<extra></extra>",',
            '                    showlegend: false',
            '                });',
            '                ',
            '                // Colored center',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [pos.r],',
            '                    theta: [pos.theta],',
            '                    mode: "markers",',
            '                    marker: {size: 8, color: pos.color, symbol: "circle"},',
            '                    hovertemplate: pos.label + "<br>Distance: " + pos.r.toFixed(2) + "mm<br>" +',
            '                                   "A-P: " + (pos.ant >= 0 ? "+" : "") + pos.ant.toFixed(2) + "mm<br>" +',
            '                                   "M-L: " + (pos.lat >= 0 ? "+" : "") + pos.lat.toFixed(2) + "mm<extra></extra>",',
            '                    showlegend: false',
            '                });',
            '                ',
            '                // Label',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [pos.r],',
            '                    theta: [pos.theta],',
            '                    mode: "text",',
            '                    text: [pos.short_label || pos.label],',
            '                    textposition: "top center",',
            '                    textfont: {size: 10, color: "#333"},',
            '                    hoverinfo: "skip",',
            '                    showlegend: false',
            '                });',
            '            });',
            '            ',
            '            return traces;',
            '        }',
            '        ',
            '        // Helper function to get track position',
            '        function getTrackPosition(trackName, offsetMm, isLeft) {',
            '            const trackMap = {',
            '                "Central": {lat: 0, ant: 0},',
            '                "Anterior": {lat: 0, ant: offsetMm},',
            '                "Posterior": {lat: 0, ant: -offsetMm},',
            '                "Lateral": {lat: offsetMm, ant: 0},',
            '                "Medial": {lat: -offsetMm, ant: 0}',
            '            };',
            '            ',
            '            let pos = trackMap[trackName] || trackMap["Central"];',
            '            let lat = pos.lat;',
            '            let ant = pos.ant;',
            '            ',
            '            // Flip lateral for right hemisphere',
            '            if (!isLeft) {',
            '                lat = -lat;',
            '            }',
            '            ',
            '            return {lat: lat, ant: ant};',
            '        }',
            '        ',
            '        // Function to add target markers (clinical/research) to traces',
            '        function addTargetMarkers(traces, targets, currentDepth, merConfig, color, label) {',
            '            targets.forEach((target, idx) => {',
            '                // Only show marker if we\'re at this target\'s depth (within 0.5mm)',
            '                if (Math.abs(currentDepth - target.depth) < 0.5) {',
            '                    const pos = getTrackPosition(target.track, merConfig.offset_mm, merConfig.is_left);',
            '                    const r = Math.sqrt(pos.lat * pos.lat + pos.ant * pos.ant);',
            '                    const theta = Math.atan2(pos.lat, pos.ant) * 180 / Math.PI;',
            '                    ',
            '                    // Add a larger circle marker',
            '                    traces.push({',
            '                        type: "scatterpolar",',
            '                        r: [r],',
            '                        theta: [theta],',
            '                        mode: "markers",',
            '                        marker: {size: 20, color: color, symbol: "circle", line: {color: color, width: 3}},',
            '                        name: label + (targets.length > 1 ? " " + (idx + 1) : ""),',
            '                        hovertemplate: label + (targets.length > 1 ? " " + (idx + 1) : "") + "<br>" + target.track + "<br>Depth: " + target.depth.toFixed(2) + "mm<extra></extra>",',
            '                        showlegend: false',
            '                    });',
            '                }',
            '            });',
            '        }',
            '        ',
            '        // Function to update position stats display',
            '        function updatePositionStats(targetIdx, positions) {',
            '            const distanceEl = document.getElementById("stats_" + targetIdx + "_distance");',
            '            const apEl = document.getElementById("stats_" + targetIdx + "_ap");',
            '            const mlEl = document.getElementById("stats_" + targetIdx + "_ml");',
            '            const xyzEl = document.getElementById("stats_" + targetIdx + "_xyz");',
            '            ',
            '            if (!distanceEl || !apEl || !mlEl || !xyzEl) return;',
            '            ',
            '            if (!positions || positions.length === 0) {',
            '                distanceEl.textContent = "--";',
            '                apEl.textContent = "--";',
            '                mlEl.textContent = "--";',
            '                xyzEl.textContent = "--";',
            '                return;',
            '            }',
            '            ',
            '            // Use first position (should only be one matched electrode)',
            '            const pos = positions[0];',
            '            const antDir = pos.ant >= 0 ? "Ant" : "Post";',
            '            const latDir = pos.lat >= 0 ? "Med" : "Lat";',
            '            ',
            '            distanceEl.textContent = pos.r.toFixed(2) + " mm";',
            '            apEl.textContent = (pos.ant >= 0 ? "+" : "") + pos.ant.toFixed(2) + " mm (" + antDir + ")";',
            '            mlEl.textContent = (pos.lat >= 0 ? "+" : "") + pos.lat.toFixed(2) + " mm (" + latDir + ")";',
            '            xyzEl.textContent = "(" + pos.x.toFixed(1) + ", " + pos.y.toFixed(1) + ", " + pos.z.toFixed(1) + ")";',
            '        }',
            '        ',
            '        // Function to find closest available depth key',
            '        function findClosestDepthKey(positionData, targetDepth) {',
            '            const availableDepths = Object.keys(positionData).map(parseFloat);',
            '            if (availableDepths.length === 0) return null;',
            '            ',
            '            let closest = availableDepths[0];',
            '            let minDiff = Math.abs(targetDepth - closest);',
            '            ',
            '            for (let i = 1; i < availableDepths.length; i++) {',
            '                const diff = Math.abs(targetDepth - availableDepths[i]);',
            '                if (diff < minDiff) {',
            '                    minDiff = diff;',
            '                    closest = availableDepths[i];',
            '                }',
            '            }',
            '            ',
            '            return closest.toFixed(2);',
            '        }',
            '        ',
            '        // Function to initialize polar chart',
            '        function initializePolarChart(targetIdx, divId, depth) {',
            '            const positionData = window["positionDataTarget" + targetIdx];',
            '            const merConfig = window["merConfigTarget" + targetIdx];',
            '            ',
            '            if (!positionData || !merConfig) {',
            '                console.error("Missing data for target", targetIdx);',
            '                return;',
            '            }',
            '            ',
            '            // Find closest available depth',
            '            const depthKey = findClosestDepthKey(positionData, depth);',
            '            if (!depthKey) {',
            '                console.error("No position data available");',
            '                return;',
            '            }',
            '            ',
            '            const positions = positionData[depthKey] || [];',
            '            const traces = createPolarChartTraces(positions, merConfig);',
            '            ',
            '            // Add clinical and research target markers at current depth',
            '            if (merConfig.clinical_targets) {',
            '                addTargetMarkers(traces, merConfig.clinical_targets, depth, merConfig, "#d62728", "Clinical");',
            '            }',
            '            if (merConfig.research_targets) {',
            '                addTargetMarkers(traces, merConfig.research_targets, depth, merConfig, "#ff7f0e", "Research");',
            '            }',
            '            ',
            '            // Set Lat/Med labels based on hemisphere (Med towards midline)',
            '            const ticktext = merConfig.is_left ? ["Ant", "Lat", "Post", "Med"] : ["Ant", "Med", "Post", "Lat"];',
            '            ',
            '            const layout = {',
            '                title: "Depth: " + (depth >= 0 ? "+" : "") + depth.toFixed(2) + "mm",',
            '                polar: {',
            '                    radialaxis: {range: [0, 4], showticklabels: true, tick0: 0, dtick: 1, title: ""},',
            '                    angularaxis: {',
            '                        direction: "clockwise",',
            '                        rotation: 90,',
            '                        tickmode: "array",',
            '                        tickvals: [0, 90, 180, 270],',
            '                        ticktext: ticktext',
            '                    }',
            '                },',
            '                showlegend: false,',
            '                width: 300,',
            '                height: 300,',
            '                margin: {l: 50, r: 50, t: 50, b: 50}',
            '            };',
            '            ',
            '            Plotly.newPlot(divId, traces, layout, {responsive: false});',
            '            ',
            '            // Update position stats',
            '            updatePositionStats(targetIdx, positions);',
            '            ',
            '            // Update NIfTI slice if available',
            '            updateNiftiSlice(targetIdx, depthKey);',
            '        }',
            '        ',
            '        // Function to update polar chart for a given target',
            '        function updatePolarChart(targetIdx, depth) {',
            '            depth = parseFloat(depth);',
            '            ',
            '            const positionData = window["positionDataTarget" + targetIdx];',
            '            const merConfig = window["merConfigTarget" + targetIdx];',
            '            ',
            '            if (!positionData || !merConfig) return;',
            '            ',
            '            // Find closest available depth',
            '            const depthKey = findClosestDepthKey(positionData, depth);',
            '            if (!depthKey) return;',
            '            ',
            '            const positions = positionData[depthKey] || [];',
            '            const traces = createPolarChartTraces(positions, merConfig);',
            '            ',
            '            // Add clinical and research target markers at current depth',
            '            if (merConfig.clinical_targets) {',
            '                addTargetMarkers(traces, merConfig.clinical_targets, depth, merConfig, "#d62728", "Clinical");',
            '            }',
            '            if (merConfig.research_targets) {',
            '                addTargetMarkers(traces, merConfig.research_targets, depth, merConfig, "#ff7f0e", "Research");',
            '            }',
            '            ',
            '            // Set Lat/Med labels based on hemisphere (Med towards midline)',
            '            const ticktext = merConfig.is_left ? ["Ant", "Lat", "Post", "Med"] : ["Ant", "Med", "Post", "Lat"];',
            '            ',
            '            const layout = {',
            '                title: "Depth: " + (depth >= 0 ? "+" : "") + depth.toFixed(2) + "mm",',
            '                polar: {',
            '                    radialaxis: {range: [0, 4], showticklabels: true, tick0: 0, dtick: 1, title: ""},',
            '                    angularaxis: {',
            '                        direction: "clockwise",',
            '                        rotation: 90,',
            '                        tickmode: "array",',
            '                        tickvals: [0, 90, 180, 270],',
            '                        ticktext: ticktext',
            '                    }',
            '                },',
            '                showlegend: false,',
            '                width: 300,',
            '                height: 300,',
            '                margin: {l: 50, r: 50, t: 50, b: 50}',
            '            };',
            '            ',
            '            // Update the chart',
            '            Plotly.react("polar_" + targetIdx, traces, layout);',
            '            ',
            '            // Update depth value display',
            '            const valueSpan = document.getElementById("depth_slider_" + targetIdx + "_value");',
            '            if (valueSpan) {',
            '                valueSpan.textContent = (depth >= 0 ? "+" : "") + depth.toFixed(2) + " mm";',
            '            }',
            '            ',
            '            // Update position stats',
            '            updatePositionStats(targetIdx, positions);',
            '            ',
            '            // Update NIfTI slice if available',
            '            updateNiftiSlice(targetIdx, depthKey);',
            '        }',
            '        ',
            '        // Function to update NIfTI slice image',
            '        function updateNiftiSlice(targetIdx, depthKey) {',
            '            const sliceImages = window["sliceImagesTarget" + targetIdx];',
            '            if (!sliceImages) return;',
            '            ',
            '            const imgElement = document.getElementById("nifti_img_" + targetIdx);',
            '            if (!imgElement) return;',
            '            ',
            '            const imageData = sliceImages[depthKey];',
            '            if (imageData) {',
            '                imgElement.src = "data:image/png;base64," + imageData;',
            '                imgElement.style.display = "block";',
            '            } else {',
            '                imgElement.style.display = "none";',
            '            }',
            '        }',
            '        ',
            '        // Function to toggle between brain image and NIfTI slice',
            '        window.imageViewState = {};  // Track view state per target (true = slice, false = brain)',
            '        function toggleImageView(targetIdx) {',
            '            const brainView = document.getElementById("brain_view_" + targetIdx);',
            '            const sliceView = document.getElementById("slice_view_" + targetIdx);',
            '            const toggleBtn = document.getElementById("toggle_btn_" + targetIdx);',
            '            ',
            '            if (!brainView || !sliceView || !toggleBtn) return;',
            '            ',
            '            // Toggle state',
            '            const showingSlice = window.imageViewState[targetIdx] || false;',
            '            ',
            '            if (showingSlice) {',
            '                // Switch to brain view',
            '                brainView.style.display = "block";',
            '                sliceView.style.display = "none";',
            '                toggleBtn.textContent = "Show Probe View";',
            '                window.imageViewState[targetIdx] = false;',
            '            } else {',
            '                // Switch to slice view',
            '                brainView.style.display = "none";',
            '                sliceView.style.display = "block";',
            '                toggleBtn.textContent = "Hide Probe View";',
            '                window.imageViewState[targetIdx] = true;',
            '            }',
            '        }',
            '        ',
            '        // Function to toggle zoom on NIfTI slice',
            '        window.zoomState = {};  // Track zoom state per target (true = zoomed, false = normal)',
            '        function toggleZoom(targetIdx) {',
            '            const sliceImg = document.getElementById("nifti_img_" + targetIdx);',
            '            const zoomBtn = document.getElementById("zoom_btn_" + targetIdx);',
            '            ',
            '            if (!sliceImg || !zoomBtn) return;',
            '            ',
            '            const isZoomed = window.zoomState[targetIdx] || false;',
            '            ',
            '            if (isZoomed) {',
            '                // Zoom out',
            '                sliceImg.classList.remove("zoomed");',
            '                zoomBtn.textContent = "Zoom In";',
            '                window.zoomState[targetIdx] = false;',
            '            } else {',
            '                // Zoom in',
            '                sliceImg.classList.add("zoomed");',
            '                zoomBtn.textContent = "Zoom Out";',
            '                window.zoomState[targetIdx] = true;',
            '            }',
            '        }',
            '        ',
            '        // Track which contact button was last clicked for each target',
            '        window.lastClickedContact = {};',
            '        ',
            '        // Function to set depth when contact button is clicked',
            '        function setDepthForTarget(targetIdx, depth, clickedButton) {',
            '            // Store that this button was clicked',
            '            window.lastClickedContact[targetIdx] = depth;',
            '            ',
            '            const slider = document.getElementById("depth_slider_" + targetIdx);',
            '            if (slider) {',
            '                slider.value = depth;',
            '                updatePolarChart(targetIdx, depth);',
            '                updateActiveContactButton(targetIdx);',
            '            }',
            '        }',
            '        ',
            '        // Function to update which contact button is active',
            '        function updateActiveContactButton(targetIdx) {',
            '            // Find all buttons (contact, research, clinical) for this target',
            '            const section = document.getElementById("polar_" + targetIdx);',
            '            if (!section) return;',
            '            const targetSection = section.closest(".section");',
            '            if (!targetSection) return;',
            '            ',
            '            const lastClicked = window.lastClickedContact[targetIdx];',
            '            ',
            '            const buttons = targetSection.querySelectorAll(".contact-button, .research-button, .clinical-button");',
            '            buttons.forEach(btn => {',
            '                const onclick = btn.getAttribute("onclick");',
            '                if (!onclick) return;',
            '                ',
            '                // Extract depth from onclick',
            '                const match = onclick.match(/setDepthForTarget\\([^,]+,\\s*([+-]?\\d+\\.?\\d*)\\)/);',
            '                if (match) {',
            '                    const btnDepth = parseFloat(match[1]);',
            '                    // Highlight if this is the last clicked button',
            '                    if (lastClicked !== undefined && Math.abs(btnDepth - lastClicked) < 0.01) {',
            '                        btn.classList.add("active");',
            '                    } else {',
            '                        btn.classList.remove("active");',
            '                    }',
            '                }',
            '            });',
            '        }',
            '        ',
            '        // Clear last clicked when slider is moved manually',
            '        function onSliderInput(targetIdx, depth) {',
            '            // Clear the last clicked contact for this target',
            '            delete window.lastClickedContact[targetIdx];',
            '            // Clear all button highlights',
            '            updateActiveContactButton(targetIdx);',
            '            // Update chart',
            '            updatePolarChart(targetIdx, depth);',
            '        }',
            '        ',
            '        // ===== Brain Shift Chart Functions =====',
            '        ',
            '        // Track which contact button was last clicked for brain shift',
            '        window.lastClickedBrainShift = {};',
            '        ',
            '        function updateBrainShiftPolarChart(pairIdx, depth) {',
            '            depth = parseFloat(depth);',
            '            ',
            '            // Determine which coordinate system is active',
            '            const activeCoord = window["brainShiftActiveCoordPair" + pairIdx] || "perp";',
            '            const positionData = window["brainShiftData" + (activeCoord === "perp" ? "Perp" : "Axial") + "Pair" + pairIdx];',
            '            const colorRecon1 = window["brainShiftColorRecon1Pair" + pairIdx] || "#FF4444";',
            '            const colorRecon2 = window["brainShiftColorRecon2Pair" + pairIdx] || "#4477FF";',
            '            const rangeMax = window["brainShiftRangeMaxPair" + pairIdx] || 4;',
            '            const dtick = window["brainShiftDtickPair" + pairIdx] || 1;',
            '            ',
            '            if (!positionData) return;',
            '            ',
            '            // Find closest available depth',
            '            const depthKey = findClosestDepthKey(positionData, depth);',
            '            if (!depthKey) return;',
            '            ',
            '            const data = positionData[depthKey];',
            '            if (!data) return;',
            '            ',
            '            // Calculate marker size for 1.27mm diameter based on scale',
            '            // Size 29 represents 1.27mm on a scale of [0,4]',
            '            // Scale proportionally for different ranges',
            '            const electrodeCircleSize = 29 * (4 / rangeMax);',
            '            const centerDotSize = 6;',
            '            ',
            '            // Create traces for both electrodes (only if data exists)',
            '            const traces = [];',
            '            ',
            '            // Add recon 1 traces if position exists',
            '            if (data.pos_1) {',
            '                // 1.27mm circle for recon 1',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [data.pos_1.r],',
            '                    theta: [data.pos_1.theta],',
            '                    mode: "markers",',
            '                    marker: {',
            '                        size: electrodeCircleSize,',
            '                        color: "rgba(255, 68, 68, 0.2)",',
            '                        symbol: "circle",',
            '                        line: {color: colorRecon1, width: 2}',
            '                    },',
            '                    hovertemplate: "Recon 1<br>Electrode (1.27mm)<extra></extra>",',
            '                    showlegend: false',
            '                });',
            '                // Center dot for recon 1',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [data.pos_1.r],',
            '                    theta: [data.pos_1.theta],',
            '                    mode: "markers",',
            '                    marker: {color: colorRecon1, size: centerDotSize, symbol: "circle"},',
            '                    name: "Recon 1",',
            '                    showlegend: true',
            '                });',
            '            }',
            '            ',
            '            // Add recon 2 traces if position exists',
            '            if (data.pos_2) {',
            '                // 1.27mm circle for recon 2',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [data.pos_2.r],',
            '                    theta: [data.pos_2.theta],',
            '                    mode: "markers",',
            '                    marker: {',
            '                        size: electrodeCircleSize,',
            '                        color: "rgba(68, 119, 255, 0.2)",',
            '                        symbol: "circle",',
            '                        line: {color: colorRecon2, width: 2}',
            '                    },',
            '                    hovertemplate: "Recon 2<br>Electrode (1.27mm)<extra></extra>",',
            '                    showlegend: false',
            '                });',
            '                // Center dot for recon 2',
            '                traces.push({',
            '                    type: "scatterpolar",',
            '                    r: [data.pos_2.r],',
            '                    theta: [data.pos_2.theta],',
            '                    mode: "markers",',
            '                    marker: {color: colorRecon2, size: centerDotSize, symbol: "circle"},',
            '                    name: "Recon 2",',
            '                    showlegend: true',
            '                });',
            '            }',
            '            ',
            '            // Determine medial/lateral labels based on X-direction relative to center',
            '            const isPosXMedial = window["brainShiftIsPositiveXMedialPair" + pairIdx] || false;',
            '            const lat90Label = isPosXMedial ? "Med" : "Lat";',
            '            const lat270Label = isPosXMedial ? "Lat" : "Med";',
            '            ',
            '            const layout = {',
            '                title: "Depth: " + (depth >= 0 ? "+" : "") + depth.toFixed(2) + "mm",',
            '                polar: {',
            '                    radialaxis: {range: [0, rangeMax], showticklabels: true, tick0: 0, dtick: dtick, title: ""},',
            '                    angularaxis: {',
            '                        direction: "clockwise",',
            '                        rotation: 90,',
            '                        tickmode: "array",',
            '                        tickvals: [0, 90, 180, 270],',
            '                        ticktext: ["Ant", lat90Label, "Post", lat270Label]',
            '                    }',
            '                },',
            '                showlegend: true,',
            '                legend: {x: 0.7, y: 1},',
            '                width: 300,',
            '                height: 300,',
            '                margin: {l: 50, r: 50, t: 50, b: 50}',
            '            };',
            '            ',
            '            Plotly.newPlot("polar_shift_" + pairIdx, traces, layout, {responsive: false});',
            '            ',
            '            // Update stats with 3D coordinates from intersection points (handle null values)',
            '            if (data.pos_1 && data.pos_2) {',
            '                // Calculate 3D displacement between intersection points',
            '                const dx = data.pos_2.x - data.pos_1.x;',
            '                const dy = data.pos_2.y - data.pos_1.y;',
            '                const dz = data.pos_2.z - data.pos_1.z;',
            '                const euclideanDist = Math.sqrt(dx * dx + dy * dy + dz * dz);',
            '                ',
            '                document.getElementById("stats_shift_" + pairIdx + "_distance").textContent = euclideanDist.toFixed(2) + " mm";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dx").textContent = (dx >= 0 ? "+" : "") + dx.toFixed(2) + " mm";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dy").textContent = (dy >= 0 ? "+" : "") + dy.toFixed(2) + " mm";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dz").textContent = (dz >= 0 ? "+" : "") + dz.toFixed(2) + " mm";',
            '                document.getElementById("stats_shift_" + pairIdx + "_xyz1").textContent = "(" + data.pos_1.x.toFixed(1) + ", " + data.pos_1.y.toFixed(1) + ", " + data.pos_1.z.toFixed(1) + ")";',
            '                document.getElementById("stats_shift_" + pairIdx + "_xyz2").textContent = "(" + data.pos_2.x.toFixed(1) + ", " + data.pos_2.y.toFixed(1) + ", " + data.pos_2.z.toFixed(1) + ")";',
            '            } else {',
            '                document.getElementById("stats_shift_" + pairIdx + "_distance").textContent = "--";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dx").textContent = "--";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dy").textContent = "--";',
            '                document.getElementById("stats_shift_" + pairIdx + "_dz").textContent = "--";',
            '                ',
            '                if (data.pos_1) {',
            '                    document.getElementById("stats_shift_" + pairIdx + "_xyz1").textContent = "(" + data.pos_1.x.toFixed(1) + ", " + data.pos_1.y.toFixed(1) + ", " + data.pos_1.z.toFixed(1) + ")";',
            '                } else {',
            '                    document.getElementById("stats_shift_" + pairIdx + "_xyz1").textContent = "--";',
            '                }',
            '                ',
            '                if (data.pos_2) {',
            '                    document.getElementById("stats_shift_" + pairIdx + "_xyz2").textContent = "(" + data.pos_2.x.toFixed(1) + ", " + data.pos_2.y.toFixed(1) + ", " + data.pos_2.z.toFixed(1) + ")";',
            '                } else {',
            '                    document.getElementById("stats_shift_" + pairIdx + "_xyz2").textContent = "--";',
            '                }',
            '            }',
            '            ',
            '            // Update slider label',
            '            document.getElementById("depth_slider_shift_" + pairIdx + "_value").textContent = (depth >= 0 ? "+" : "") + depth.toFixed(2) + " mm";',
            '        }',
            '        ',
            '        function setDepthForBrainShift(pairIdx, depth) {',
            '            window.lastClickedBrainShift[pairIdx] = depth;',
            '            const slider = document.getElementById("depth_slider_shift_" + pairIdx);',
            '            if (slider) {',
            '                slider.value = depth;',
            '                updateBrainShiftPolarChart(pairIdx, depth);',
            '                updateActiveBrainShiftButton(pairIdx);',
            '            }',
            '        }',
            '        ',
            '        function switchBrainShiftCoords(pairIdx, coordType) {',
            '            // Update active coordinate system',
            '            window["brainShiftActiveCoordPair" + pairIdx] = coordType;',
            '            ',
            '            // Update button states',
            '            const perpBtn = document.getElementById("toggle_perp_" + pairIdx);',
            '            const axialBtn = document.getElementById("toggle_axial_" + pairIdx);',
            '            ',
            '            if (coordType === "perp") {',
            '                perpBtn.classList.add("active");',
            '                axialBtn.classList.remove("active");',
            '            } else {',
            '                perpBtn.classList.remove("active");',
            '                axialBtn.classList.add("active");',
            '            }',
            '            ',
            '            // Refresh the chart with current depth',
            '            const slider = document.getElementById("depth_slider_shift_" + pairIdx);',
            '            if (slider) {',
            '                updateBrainShiftPolarChart(pairIdx, parseFloat(slider.value));',
            '            }',
            '        }',
            '        ',
            '        function switchContactView(pairIdx, viewType) {',
            '            // Update button states',
            '            const sagBtn = document.getElementById("toggle_sag_" + pairIdx);',
            '            const corBtn = document.getElementById("toggle_cor_" + pairIdx);',
            '            ',
            '            // Update image visibility',
            '            const sagImg = document.getElementById("contact_view_img_" + pairIdx + "_sagittal");',
            '            const corImg = document.getElementById("contact_view_img_" + pairIdx + "_coronal");',
            '            ',
            '            if (viewType === "sagittal") {',
            '                sagBtn.classList.add("active");',
            '                corBtn.classList.remove("active");',
            '                sagImg.style.display = "block";',
            '                corImg.style.display = "none";',
            '            } else {',
            '                sagBtn.classList.remove("active");',
            '                corBtn.classList.add("active");',
            '                sagImg.style.display = "none";',
            '                corImg.style.display = "block";',
            '            }',
            '        }',
            '        ',
            '        function updateActiveBrainShiftButton(pairIdx) {',
            '            const section = document.getElementById("polar_shift_" + pairIdx);',
            '            if (!section) return;',
            '            const targetSection = section.closest(".section");',
            '            if (!targetSection) return;',
            '            ',
            '            const lastClicked = window.lastClickedBrainShift[pairIdx];',
            '            const buttons = targetSection.querySelectorAll(".contact-button");',
            '            buttons.forEach(btn => {',
            '                const onclick = btn.getAttribute("onclick");',
            '                if (!onclick) return;',
            '                const match = onclick.match(/setDepthForBrainShift\\([^,]+,\\s*([+-]?\\d+\\.?\\d*)\\)/);',
            '                if (match) {',
            '                    const btnDepth = parseFloat(match[1]);',
            '                    if (lastClicked !== undefined && Math.abs(btnDepth - lastClicked) < 0.01) {',
            '                        btn.classList.add("active");',
            '                    } else {',
            '                        btn.classList.remove("active");',
            '                    }',
            '                }',
            '            });',
            '        }',
            '        ',
            '        function onBrainShiftSliderInput(pairIdx, depth) {',
            '            delete window.lastClickedBrainShift[pairIdx];',
            '            updateActiveBrainShiftButton(pairIdx);',
            '            updateBrainShiftPolarChart(pairIdx, depth);',
            '        }',
            '        ',
            '        // Initialize all charts after functions are defined',
            '        if (window.chartsToInit) {',
            '            window.chartsToInit.forEach(function(params) {',
            '                // Store initial depth as last clicked for C3',
            '                window.lastClickedContact[params.targetIdx] = params.depth;',
            '                // Initialize chart',
            '                initializePolarChart(params.targetIdx, params.divId, params.depth);',
            '                // Initialize the active contact button',
            '                updateActiveContactButton(params.targetIdx);',
            '            });',
            '        }',
            '        ',
            '        // Initialize brain shift charts',
            '        if (window.brainShiftChartsToInit) {',
            '            window.brainShiftChartsToInit.forEach(function(params) {',
            '                window.lastClickedBrainShift[params.pairIdx] = params.depth;',
            '                updateBrainShiftPolarChart(params.pairIdx, params.depth);',
            '                updateActiveBrainShiftButton(params.pairIdx);',
            '            });',
            '        }',
            '    </script>',
            '</body>',
            '</html>',
        ])

        return '\n'.join(html_parts)

    def save_and_download(self, filename_prefix: str = "stereotactic_report") -> Tuple[str, str]:
        """
        Generate HTML report and save to temporary file.

        Args:
            filename_prefix: Prefix for the output filename

        Returns:
            Tuple of (temp_file_path, filename)
        """
        html_content = self.generate_html()

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get patient ID for filename
        if self.patient_id:
            patient_str = self.patient_id
        else:
            patient_ids = set()
            for target in self.surgical_targets:
                pid = target.get('patient_id', '')
                if pid:
                    patient_ids.add(pid)
            patient_str = '_'.join(patient_ids) if patient_ids else 'report'
        # Clean filename
        patient_str = ''.join(c if c.isalnum() or c in '-_' else '_' for c in patient_str)

        filename = f'{filename_prefix}_{patient_str}_{timestamp}.html'

        # Write to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return temp_path, filename
