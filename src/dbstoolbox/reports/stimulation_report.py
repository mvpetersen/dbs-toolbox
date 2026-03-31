"""Stimulation targeting report generator.

Generates a standalone HTML report with:
- Per-electrode cards showing axial NIfTI slices at each contact position (3D)
- Optional 3D isosurface plot from 4D probability maps with electrode overlays
"""

import base64
import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from dbstoolbox import __version__

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import map_coordinates, gaussian_filter


class StimulationReportGenerator:
    """Generates HTML stimulation targeting reports."""

    AXIAL_SLICE_SIZE_MM = 50.0  # 50x50mm region around contact
    AXIAL_RESOLUTION_MM = 0.1   # Resolution for resampled grid
    ELECTRODE_DIAMETER_MM = 1.27

    # Color palette for 4D volume isosurfaces
    VOLUME_COLORS = [
        'rgb(31,119,180)',   # blue
        'rgb(255,127,14)',   # orange
        'rgb(44,160,44)',    # green
        'rgb(214,39,40)',    # red
        'rgb(148,103,189)',  # purple
        'rgb(140,86,75)',    # brown
        'rgb(227,119,194)',  # pink
        'rgb(127,127,127)',  # gray
        'rgb(188,189,34)',   # olive
        'rgb(23,190,207)',   # cyan
    ]

    VOLUME_COLORS_HEX = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    def __init__(
        self,
        electrodes: List[Dict],
        nifti_3d_data: Optional[np.ndarray] = None,
        nifti_3d_affine: Optional[np.ndarray] = None,
        nifti_3d_filename: str = '',
        nifti_4d_data: Optional[np.ndarray] = None,
        nifti_4d_affine: Optional[np.ndarray] = None,
        nifti_4d_filename: str = '',
        volume_labels: Optional[List[str]] = None,
        volume_colors: Optional[List[Optional[str]]] = None,
        volume_visible: Optional[List[bool]] = None,
        electrode_metadata: Optional[Dict] = None,
        threshold: float = 0.5,
        acpc_landmarks: Optional[Dict[str, np.ndarray]] = None,
        patient_id: str = '',
    ):
        self.electrodes = electrodes
        self.threshold = threshold
        self.acpc_landmarks = acpc_landmarks  # {'AC': array, 'PC': array, 'midline': array}
        self.patient_id = patient_id
        self.nifti_3d_data = nifti_3d_data
        self.nifti_3d_affine = nifti_3d_affine
        self.nifti_3d_filename = nifti_3d_filename
        self.nifti_4d_data = nifti_4d_data
        self.nifti_4d_affine = nifti_4d_affine
        self.nifti_4d_filename = nifti_4d_filename
        self.electrode_metadata = electrode_metadata or {}

        # 4D volume info
        if nifti_4d_data is not None and nifti_4d_data.ndim == 4:
            self.num_volumes = nifti_4d_data.shape[3]
        else:
            self.num_volumes = 0

        if volume_labels:
            self.volume_labels = volume_labels
        elif self.num_volumes > 0:
            self.volume_labels = [f'Volume {i}' for i in range(self.num_volumes)]
        else:
            self.volume_labels = []

        # Per-volume colors (None entries fall back to default palette)
        self.volume_colors = volume_colors or [None] * self.num_volumes
        # Per-volume visibility (True = include in isosurface and contours)
        self.volume_visible = volume_visible or [True] * self.num_volumes

        # Compute geometry for each electrode
        for elec in self.electrodes:
            self._compute_electrode_geometry(elec)

    def _get_volume_color(self, vol_idx: int) -> str:
        """Get the hex color for a volume, using label override or default palette."""
        if vol_idx < len(self.volume_colors) and self.volume_colors[vol_idx]:
            return self.volume_colors[vol_idx]
        return self.VOLUME_COLORS_HEX[vol_idx % len(self.VOLUME_COLORS_HEX)]

    def _is_volume_visible(self, vol_idx: int) -> bool:
        """Check whether a volume should be rendered."""
        if vol_idx < len(self.volume_visible):
            return self.volume_visible[vol_idx]
        return True

    @classmethod
    def from_json(
        cls,
        electrode_json: Union[str, Dict],
        nifti_3d_path: Optional[str] = None,
        nifti_4d_path: Optional[str] = None,
        label_path: Optional[str] = None,
        acpc_path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> 'StimulationReportGenerator':
        """Create from file paths or loaded data."""
        import nibabel as nib

        # Load electrode data
        if isinstance(electrode_json, str):
            with open(electrode_json, 'r') as f:
                electrode_data = json.load(f)
            electrode_filename = Path(electrode_json).name
        else:
            electrode_data = electrode_json
            electrode_filename = 'electrode_data'

        metadata = electrode_data.get('metadata', {})
        is_transformed = metadata.get('transformed', False)

        raw_electrodes = electrode_data.get('electrodes', [])
        electrodes = []
        for idx, elec in enumerate(raw_electrodes):
            parsed = cls._parse_electrode(elec, idx, electrode_filename, is_transformed)
            if parsed:
                electrodes.append(parsed)

        if not electrodes:
            raise ValueError('No valid electrodes found in the reconstruction data')

        # Load 3D NIfTI
        nifti_3d_data = None
        nifti_3d_affine = None
        nifti_3d_filename = ''
        if nifti_3d_path:
            img = nib.load(nifti_3d_path)
            nifti_3d_data = img.get_fdata()
            nifti_3d_affine = img.affine
            nifti_3d_filename = Path(nifti_3d_path).name
            # Handle 4D with single volume as 3D
            if nifti_3d_data.ndim == 4 and nifti_3d_data.shape[3] == 1:
                nifti_3d_data = nifti_3d_data[..., 0]

        # Load 4D NIfTI
        nifti_4d_data = None
        nifti_4d_affine = None
        nifti_4d_filename = ''
        if nifti_4d_path:
            img = nib.load(nifti_4d_path)
            nifti_4d_data = img.get_fdata()
            nifti_4d_affine = img.affine
            nifti_4d_filename = Path(nifti_4d_path).name

        # Load labels (and optional colors / visibility)
        volume_labels = None
        volume_colors = None
        volume_visible = None
        if label_path and nifti_4d_data is not None:
            parsed = cls._parse_label_file(label_path, nifti_4d_data)
            if parsed:
                volume_labels = parsed['labels']
                volume_colors = parsed['colors']
                volume_visible = parsed['visible']

        # Load AC-PC landmarks
        acpc_landmarks = None
        if acpc_path:
            acpc_landmarks = cls._parse_acpc_csv(acpc_path)

        return cls(
            electrodes=electrodes,
            nifti_3d_data=nifti_3d_data,
            nifti_3d_affine=nifti_3d_affine,
            nifti_3d_filename=nifti_3d_filename,
            nifti_4d_data=nifti_4d_data,
            nifti_4d_affine=nifti_4d_affine,
            nifti_4d_filename=nifti_4d_filename,
            volume_labels=volume_labels,
            volume_colors=volume_colors,
            volume_visible=volume_visible,
            electrode_metadata=metadata,
            threshold=threshold,
            acpc_landmarks=acpc_landmarks,
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_electrode(
        electrode: Dict, idx: int, filename: str, is_transformed: bool = False
    ) -> Optional[Dict]:
        """Parse an electrode entry into trajectory data."""
        try:
            if 'trajectory_coordinates' in electrode:
                trajectory = np.array(electrode['trajectory_coordinates'])
            elif 'polynomial' in electrode and not is_transformed:
                poly_coeffs = np.array(electrode['polynomial'])
                t_vals = np.linspace(0, 1, 100)
                trajectory = np.array([
                    [np.polyval(poly_coeffs[:, dim], t) for dim in range(3)]
                    for t in t_vals
                ])
            else:
                return None

            contacts = None
            if 'contact_positions_3d' in electrode:
                contacts = np.array(electrode['contact_positions_3d'])

            # Parse orientation data for directional electrodes
            orientation = electrode.get('orientation')

            return {
                'trajectory': trajectory,
                'contacts': contacts,
                'orientation': orientation,
                'electrode_idx': idx,
                'filename': filename,
                'label': ' - '.join(filter(None, [
                    f'E{idx+1}',
                    electrode.get('electrode_type', ''),
                    electrode.get('side', '').capitalize(),
                ])),
                'side': electrode.get('side', ''),
            }
        except (ValueError, KeyError) as e:
            print(f"Error parsing electrode trajectory: {e}")
            return None

    @staticmethod
    def _parse_label_file(
        label_path: str, nifti_data: np.ndarray
    ) -> Optional[Dict[str, list]]:
        """Parse a label TXT file.

        Supports formats:
            index  label
            index  label  #hexcolor
            index  label  #hexcolor  visible(0/1)

        Returns dict with keys 'labels', 'colors', 'visible' (each a list indexed by volume),
        or None on failure.
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            entries: Dict[int, Dict] = {}
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue

                label = parts[1]
                color = None
                visible = True

                if len(parts) >= 3 and parts[2].startswith('#'):
                    color = parts[2]
                if len(parts) >= 4:
                    try:
                        visible = parts[3] != '0'
                    except (ValueError, IndexError):
                        pass

                entries[idx] = {'label': label, 'color': color, 'visible': visible}

            if not entries:
                return None

            max_idx = max(entries.keys())
            num_volumes = nifti_data.shape[3] if nifti_data.ndim == 4 else 1
            count = max(max_idx + 1, num_volumes)

            labels = []
            colors = []
            visible = []
            for i in range(count):
                entry = entries.get(i, {})
                labels.append(entry.get('label', f'Volume {i}'))
                colors.append(entry.get('color'))  # None means use default
                visible.append(entry.get('visible', True))

            return {'labels': labels, 'colors': colors, 'visible': visible}
        except Exception as e:
            print(f"Warning: Failed to parse label file: {e}")
            return None

    @staticmethod
    def _compute_electrode_geometry(electrode: Dict):
        """Compute tip, entry, direction and trajectory length."""
        trajectory = electrode['trajectory']
        tip = trajectory[-1]
        entry = trajectory[0]
        direction = entry - tip
        length = np.linalg.norm(direction)
        direction = direction / length

        electrode['tip'] = tip
        electrode['entry'] = entry
        electrode['direction'] = direction
        electrode['trajectory_length'] = length

    @staticmethod
    def _parse_acpc_csv(csv_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Parse a CSV file with AC, PC, and midline (VSPS) landmark coordinates.

        Expected format:
            x,y,z,t,label
            -1.81,..,...,0,AC
            96.40,..,...,0,PC
            96.70,..,...,0,VSPS

        Returns dict with 'AC', 'PC', 'midline' keys mapping to 3D arrays, or None.
        """
        try:
            import csv
            landmarks = {}
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get('label', '').strip().upper()
                    if label in ('AC', 'PC', 'VSPS', 'MIDLINE'):
                        pos = np.array([
                            float(row['x']), float(row['y']), float(row['z'])
                        ])
                        if label in ('VSPS', 'MIDLINE'):
                            landmarks['midline'] = pos
                        else:
                            landmarks[label] = pos

            if 'AC' not in landmarks or 'PC' not in landmarks:
                print(f"Warning: AC-PC CSV missing required landmarks (found: {list(landmarks.keys())})")
                return None

            if 'midline' not in landmarks:
                print("Warning: AC-PC CSV missing midline/VSPS landmark")
                return None

            print(f"AC-PC landmarks: AC={landmarks['AC']}, PC={landmarks['PC']}, midline={landmarks['midline']}")
            return landmarks
        except Exception as e:
            print(f"Warning: Failed to parse AC-PC CSV: {e}")
            return None

    def _get_acpc_axes(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the AC-PC coordinate axes. Returns (x_axis, y_axis, z_axis) or None.

        Coordinate system (enforced via NIfTI affine):
            Y-axis: AC → PC (posterior positive)
            X-axis: left-to-right positive, perpendicular to Y and midline vector
            Z-axis: inferior-to-superior positive (the axial plane normal)
        """
        if self.acpc_landmarks is None:
            return None

        ac = self.acpc_landmarks['AC']
        pc = self.acpc_landmarks['PC']
        midline = self.acpc_landmarks['midline']

        # Y: PC → AC (anterior positive)
        y_axis = ac - pc
        y_axis = y_axis / np.linalg.norm(y_axis)

        # The midline point and AC-PC line define the midsagittal plane.
        # The plane normal (X-axis) is perpendicular to both Y and the
        # AC→midline vector. However, the midline point may be above or
        # below the AC-PC plane, so we cannot rely on cross(Y, mid_vec)
        # alone to determine left/right or superior/inferior.
        #
        # Strategy: compute X from the midline geometry, then enforce
        # Z+ = superior and X+ = right using RAS world-space conventions,
        # re-deriving dependent axes to maintain right-handedness.
        mid_vec = midline - ac

        # Initial X from cross product (defines the midsagittal plane)
        x_axis = np.cross(y_axis, mid_vec)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Z from X × Y
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Enforce Z+ = superior (RAS Z+ is superior).
        # If Z points inferior, flip it and re-derive X = Y × Z.
        if z_axis[2] < 0:
            z_axis = -z_axis

        # Re-derive X from Y × Z to guarantee right-handedness with
        # the corrected Z direction.
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        return x_axis, y_axis, z_axis

    def _world_to_acpc(self, point: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Transform a world-space point to AC-PC coordinates.

        Returns (coords, y_norm) where:
            coords: [x, y, z] in mm relative to AC
            y_norm: y normalised to AC-PC distance (AC=0, PC=1, MCP=0.5)
        Or None if no ACPC landmarks.
        """
        axes = self._get_acpc_axes()
        if axes is None:
            return None

        x_axis, y_axis, z_axis = axes
        ac = self.acpc_landmarks['AC']
        pc = self.acpc_landmarks['PC']
        acpc_dist = float(np.linalg.norm(pc - ac))

        diff = point - ac
        coords = np.array([
            float(np.dot(diff, x_axis)),
            float(np.dot(diff, y_axis)),
            float(np.dot(diff, z_axis)),
        ])
        # Normalise along AC-PC: AC=0, MCP=0.5, PC=1.0
        # Y+ is anterior, so PC has negative Y. Negate to get AC=0 → PC=1.
        y_norm = -coords[1] / acpc_dist if acpc_dist > 1e-6 else 0.0
        return coords, y_norm

    def _calculate_contact_depths(self, electrode: Dict) -> List[Optional[float]]:
        """Calculate depth of each contact from the tip."""
        contacts = electrode.get('contacts')
        if contacts is None:
            return [None, None, None, None]
        tip = electrode['tip']
        direction = electrode['direction']
        depths = []
        for i in range(min(4, len(contacts))):
            diff = contacts[i] - tip
            depths.append(round(float(np.dot(diff, direction)), 2))
        while len(depths) < 4:
            depths.append(None)
        return depths

    MARKER_COLORS = {'A': '#FFA500', 'B': '#1E90FF', 'C': '#32CD32'}

    @staticmethod
    def _get_orientation_markers(electrode: Dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extract orientation markers (A, B, computed C) from electrode data.

        Returns dict of marker_key -> (position_xyz, direction_vector).
        """
        orientation = electrode.get('orientation')
        if not orientation or not orientation.get('has_markers') or 'markers' not in orientation:
            return {}

        markers = {}
        for mk, md in orientation['markers'].items():
            p = md.get('position_xyz')
            d = md.get('direction_vector')
            if p is not None and d is not None:
                markers[mk] = (np.array(p, dtype=float), np.array(d, dtype=float))

        # Compute virtual C marker: position = midpoint(A,B), direction = -(A+B) normalized
        if 'A' in markers and 'B' in markers:
            pos_a, dir_a = markers['A']
            pos_b, dir_b = markers['B']
            pos_c = (pos_a + pos_b) / 2.0
            dir_c = -(dir_a + dir_b)
            norm_c = np.linalg.norm(dir_c)
            if norm_c > 1e-10:
                dir_c = dir_c / norm_c
            markers['C'] = (pos_c, dir_c)

        return markers

    # ------------------------------------------------------------------
    # 3D NIfTI: axial slice rendering at contact positions
    # ------------------------------------------------------------------

    @staticmethod
    def _voxel_axial_normal_to_world(affine: np.ndarray) -> np.ndarray:
        """Get the world-space normal corresponding to the voxel k-axis (axial plane).

        This is the default slice plane normal. To align with AC-PC space,
        replace this with a custom normal vector.
        """
        # The k-axis in voxel space is [0, 0, 1]. Transform its direction
        # to world space using the rotation/scaling part of the affine.
        k_world = affine[:3, 2]  # third column = world direction of k-axis
        return k_world / np.linalg.norm(k_world)

    def _build_slice_grid(
        self,
        center: np.ndarray,
        plane_normal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build an mm-spaced sampling grid on an arbitrary plane in world space.

        Args:
            center: 3D world-space point at the centre of the slice.
            plane_normal: Unit normal defining the slice plane orientation.

        Returns:
            (ras_points, u_range, v_range, u_axis, v_axis):
                ras_points: (N*N, 4) homogeneous world coordinates for sampling
                u_range: 1D array of in-plane u offsets in mm (image X axis)
                v_range: 1D array of in-plane v offsets in mm (image Y axis)
                u_axis: unit vector for the u direction
                v_axis: unit vector for the v direction (≈ anterior)
        """
        half = self.AXIAL_SLICE_SIZE_MM / 2.0
        n_points = int(self.AXIAL_SLICE_SIZE_MM / self.AXIAL_RESOLUTION_MM)

        # Build an orthonormal basis for the slice plane.
        # v_axis should point towards anterior (RAS Y+) as closely as possible
        # so that "up" in the image ≈ anterior.
        ras_anterior = np.array([0.0, 1.0, 0.0])
        v_axis = ras_anterior - np.dot(ras_anterior, plane_normal) * plane_normal
        if np.linalg.norm(v_axis) < 1e-6:
            # Normal is nearly parallel to anterior — fall back to superior
            ras_superior = np.array([0.0, 0.0, 1.0])
            v_axis = ras_superior - np.dot(ras_superior, plane_normal) * plane_normal
        v_axis = v_axis / np.linalg.norm(v_axis)

        u_axis = np.cross(v_axis, plane_normal)
        u_axis = u_axis / np.linalg.norm(u_axis)

        u_range = np.linspace(-half, half, n_points)
        v_range = np.linspace(-half, half, n_points)
        uu, vv = np.meshgrid(u_range, v_range)

        ras_points = np.zeros((n_points * n_points, 4))
        ras_points[:, :3] = (
            center
            + uu.ravel()[:, None] * u_axis
            + vv.ravel()[:, None] * v_axis
        )
        ras_points[:, 3] = 1.0

        return ras_points, u_range, v_range, u_axis, v_axis

    @staticmethod
    @staticmethod
    def _render_figure_to_base64(fig, facecolor='black') -> str:
        """Render a matplotlib figure to base64 PNG and close it."""
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(
            buf, format='png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0,
            facecolor=facecolor, transparent=(facecolor == 'none'),
            pil_kwargs={'optimize': True, 'compress_level': 9},
        )
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _render_axial_contact_image(
        self,
        contact_pos: np.ndarray,
        electrode: Optional[Dict] = None,
        plane_normal: Optional[np.ndarray] = None,
        draw_arrows: bool = False,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], List[int]]:
        """Render a slice centred on a contact position.

        Returns:
            (base_b64, seg_overlay_b64, arrow_overlay_b64, acpc_overlay_b64, intersecting_volumes)
        """
        has_3d = self.nifti_3d_data is not None and self.nifti_3d_affine is not None
        has_4d = self.nifti_4d_data is not None and self.nifti_4d_affine is not None

        if not has_3d and not has_4d:
            return None, None, None, None, []

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Determine plane normal from whichever NIfTI is available
        if plane_normal is None:
            ref_affine = self.nifti_3d_affine if has_3d else self.nifti_4d_affine
            plane_normal = self._voxel_axial_normal_to_world(ref_affine)

        half = self.AXIAL_SLICE_SIZE_MM / 2.0
        n_points = int(self.AXIAL_SLICE_SIZE_MM / self.AXIAL_RESOLUTION_MM)

        ras_points, u_range, v_range, u_axis, v_axis = self._build_slice_grid(contact_pos, plane_normal)

        # --- Base image ---
        fig, ax = plt.subplots(figsize=(2, 2), dpi=120)

        if has_3d:
            # Grayscale anatomical background
            nifti_data = self.nifti_3d_data
            if nifti_data.ndim == 4:
                nifti_data = nifti_data[..., 0]

            inv_affine = np.linalg.inv(self.nifti_3d_affine)
            voxel_coords = (inv_affine @ ras_points.T).T[:, :3]

            slice_data = map_coordinates(
                nifti_data,
                [voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]],
                order=1, mode='constant', cval=0.0,
            ).reshape(n_points, n_points)

            vmin = np.percentile(slice_data, 2)
            vmax = np.percentile(slice_data, 98)
            if vmax <= vmin:
                vmax = vmin + 1

            ax.imshow(
                slice_data, cmap='gray',
                extent=[-half, half, -half, half],
                origin='lower', interpolation='bilinear',
                vmin=vmin, vmax=vmax,
            )

        # Electrode marker
        circle = plt.Circle(
            (0, 0), self.ELECTRODE_DIAMETER_MM / 2.0,
            color='#ff4444', fill=False, linewidth=1.5,
        )
        ax.add_patch(circle)
        ax.plot(0, 0, '+', color='#ff4444', markersize=4, markeredgewidth=0.8)

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
        ax.axis('off')

        # If 4D only (no 3D background), draw contours directly on the base image
        overlay_b64 = None
        intersecting_volumes = []

        if has_4d:
            inv_affine_4d = np.linalg.inv(self.nifti_4d_affine)
            voxel_coords_4d = (inv_affine_4d @ ras_points.T).T[:, :3]

            # Choose where to draw contours: on the base if no 3D, else on a separate overlay
            if has_3d:
                # Separate transparent overlay
                fig_ov, ax_ov = plt.subplots(figsize=(2, 2), dpi=120)
                ax_ov.set_xlim(-half, half)
                ax_ov.set_ylim(-half, half)
                ax_ov.set_aspect('equal')
                ax_ov.axis('off')
                fig_ov.patch.set_alpha(0.0)
                ax_ov.patch.set_alpha(0.0)
                contour_ax = ax_ov
            else:
                # Draw directly on the base (black background)
                contour_ax = ax

            has_contours = False
            for vol_idx in range(self.num_volumes):
                if not self._is_volume_visible(vol_idx):
                    continue

                vol_data = self.nifti_4d_data[..., vol_idx]

                vol_slice = map_coordinates(
                    vol_data,
                    [voxel_coords_4d[:, 0], voxel_coords_4d[:, 1], voxel_coords_4d[:, 2]],
                    order=1, mode='constant', cval=0.0,
                ).reshape(n_points, n_points)

                if np.max(vol_slice) < 0.1:
                    continue

                vol_slice_smooth = gaussian_filter(vol_slice, sigma=2.0)
                color = self._get_volume_color(vol_idx)
                contour_ax.contour(
                    u_range, v_range, vol_slice_smooth,
                    levels=[self.threshold], colors=[color], linewidths=[0.8],
                )
                has_contours = True
                intersecting_volumes.append(vol_idx)

            if has_3d:
                if has_contours:
                    overlay_b64 = self._render_figure_to_base64(fig_ov, facecolor='none')
                else:
                    plt.close(fig_ov)

        base_b64 = self._render_figure_to_base64(fig, facecolor='black')

        # --- Arrow overlay: orientation markers projected onto slice plane ---
        arrow_b64 = None
        if draw_arrows and electrode is not None:
            markers = self._get_orientation_markers(electrode)
            if markers:
                fig_arr, ax_arr = plt.subplots(figsize=(2, 2), dpi=120)
                ax_arr.set_xlim(-half, half)
                ax_arr.set_ylim(-half, half)
                ax_arr.set_aspect('equal')
                ax_arr.axis('off')
                fig_arr.patch.set_alpha(0.0)
                ax_arr.patch.set_alpha(0.0)

                has_arrows = False
                arrow_length_mm = 8.0
                arrow_start_mm = 2.0
                for mk, (pos, m_dir) in markers.items():
                    du = float(np.dot(m_dir, u_axis))
                    dv = float(np.dot(m_dir, v_axis))
                    proj_len = np.sqrt(du * du + dv * dv)
                    if proj_len < 1e-6:
                        continue
                    du_n = du / proj_len
                    dv_n = dv / proj_len

                    m_color = self.MARKER_COLORS.get(mk, '#888')
                    x0 = du_n * arrow_start_mm
                    y0 = dv_n * arrow_start_mm
                    x1 = du_n * arrow_length_mm
                    y1 = dv_n * arrow_length_mm

                    ax_arr.annotate(
                        '', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=m_color, lw=1.2, mutation_scale=10),
                    )
                    has_arrows = True

                if has_arrows:
                    arrow_b64 = self._render_figure_to_base64(fig_arr, facecolor='none')
                else:
                    plt.close(fig_arr)

        # --- ACPC overlay: AC, PC, MCP markers projected onto slice plane ---
        acpc_ov_b64 = None
        if self.acpc_landmarks is not None:
            ac = self.acpc_landmarks['AC']
            pc = self.acpc_landmarks['PC']
            mcp = (ac + pc) / 2.0  # Midcommissural point

            fig_ac, ax_ac = plt.subplots(figsize=(2, 2), dpi=120)
            ax_ac.set_xlim(-half, half)
            ax_ac.set_ylim(-half, half)
            ax_ac.set_aspect('equal')
            ax_ac.axis('off')
            fig_ac.patch.set_alpha(0.0)
            ax_ac.patch.set_alpha(0.0)

            # Project AC, PC, MCP onto the slice plane (relative to contact_pos centre)
            color_ac = '#ffd700'   # gold
            color_pc = '#daa520'   # goldenrod
            color_mcp = '#ffe44d'  # light yellow
            color_line = '#ccaa00' # dark gold for connecting line
            marker_radius = 0.8
            mcp_half_width = 1.5

            colors_map = {'AC': color_ac, 'PC': color_pc, 'MCP': color_mcp}

            for pt, label in [(ac, 'AC'), (pc, 'PC'), (mcp, 'MCP')]:
                diff = pt - contact_pos
                u_coord = float(np.dot(diff, u_axis))
                v_coord = float(np.dot(diff, v_axis))
                c = colors_map[label]

                if label == 'MCP':
                    ax_ac.plot(
                        [u_coord - mcp_half_width, u_coord + mcp_half_width],
                        [v_coord, v_coord],
                        color=c, linewidth=1.2,
                    )
                else:
                    circle = plt.Circle(
                        (u_coord, v_coord), marker_radius,
                        color=c, fill=False, linewidth=1.2,
                    )
                    ax_ac.add_patch(circle)

            # Line between AC and PC
            ac_u = float(np.dot(ac - contact_pos, u_axis))
            ac_v = float(np.dot(ac - contact_pos, v_axis))
            pc_u = float(np.dot(pc - contact_pos, u_axis))
            pc_v = float(np.dot(pc - contact_pos, v_axis))
            ax_ac.plot([ac_u, pc_u], [ac_v, pc_v], color=color_line, linewidth=0.8, linestyle='--')

            acpc_ov_b64 = self._render_figure_to_base64(fig_ac, facecolor='none')

        return base_b64, overlay_b64, arrow_b64, acpc_ov_b64, intersecting_volumes

    # ------------------------------------------------------------------
    # Coronal reference plot
    # ------------------------------------------------------------------

    def _render_coronal_reference(self, electrode: Dict) -> Optional[str]:
        """Render a coronal reference plot showing the electrode trajectory and contacts.

        Views from posterior along the AC-PC Y-axis (coronal plane).
        Axes: X = lateral (right+), Y = superior (up).
        If no AC-PC data, uses world X and Z.

        Returns base64 PNG or None.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        contacts = electrode.get('contacts')
        trajectory = electrode['trajectory']
        if contacts is None or len(contacts) == 0:
            return None

        acpc_axes = self._get_acpc_axes()

        if acpc_axes is not None:
            x_axis, y_axis, z_axis = acpc_axes
            origin = self.acpc_landmarks['AC']

            # Project trajectory and contacts onto coronal plane (X-Z in ACPC)
            def project(pt):
                diff = pt - origin
                return float(np.dot(diff, x_axis)), float(np.dot(diff, z_axis))

            origin_x, origin_z = 0.0, 0.0
            show_origin = True
        else:
            # No ACPC: use world X and Z directly
            def project(pt):
                return float(pt[0]), float(pt[2])

            show_origin = False

        # Project all points
        traj_proj = [project(p) for p in trajectory]
        contact_proj = [(ci, project(contacts[ci])) for ci in range(min(4, len(contacts)))]

        # Compute bounds from contacts (and origin if available)
        all_x = [p[1] for p in contact_proj]
        all_z = [q[1] for ci, q in contact_proj]
        if show_origin:
            all_x.append(origin_x)
            all_z.append(origin_z)

        # Wait — all_x should be the x-coordinates, all_z the z-coordinates
        all_x = [p[0] for _, p in contact_proj]
        all_z = [p[1] for _, p in contact_proj]
        if show_origin:
            all_x.append(origin_x)
            all_z.append(origin_z)

        pad = 5.0
        x_min, x_max = min(all_x) - pad, max(all_x) + pad
        z_min, z_max = min(all_z) - pad, max(all_z) + pad

        # Make it square to match the slice image aspect
        x_range = x_max - x_min
        z_range = z_max - z_min
        if x_range > z_range:
            mid_z = (z_min + z_max) / 2
            z_min = mid_z - x_range / 2
            z_max = mid_z + x_range / 2
        else:
            mid_x = (x_min + x_max) / 2
            x_min = mid_x - z_range / 2
            x_max = mid_x + z_range / 2

        fig, ax = plt.subplots(figsize=(2, 2), dpi=240)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Trajectory line
        traj_x = [p[0] for p in traj_proj]
        traj_z = [p[1] for p in traj_proj]
        # Clip to view range
        ax.plot(traj_x, traj_z, color='grey', linewidth=1.5, solid_capstyle='round')

        # Contact markers and labels
        for ci, (cx, cz) in contact_proj:
            ax.plot(cx, cz, 'o', color='red', markersize=5, zorder=5)
            ax.text(cx + (x_max - x_min) * 0.05, cz, f'C{ci}',
                    color='#333', fontsize=7, fontweight='bold',
                    va='center', ha='left', zorder=6)

        # Origin marker
        if show_origin:
            ax.plot(origin_x, origin_z, '+', color='#daa520', markersize=8,
                    markeredgewidth=1.5, zorder=5)
            ax.text(origin_x, origin_z + (z_max - z_min) * 0.05, 'AC',
                    color='#daa520', fontsize=6, fontweight='bold',
                    ha='center', va='bottom', zorder=6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect('equal')
        ax.axis('off')

        return self._render_figure_to_base64(fig, facecolor='white')

    # ------------------------------------------------------------------
    # 4D NIfTI: isosurface extraction and 3D Plotly figure
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_isosurface(
        volume: np.ndarray, affine: np.ndarray,
        threshold: float = 0.5, smoothness: float = 0.5,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extract surface mesh using marching cubes. Returns (vertices, faces) or None."""
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            print("Warning: scikit-image not available, skipping isosurface extraction")
            return None

        try:
            if smoothness > 0:
                volume = gaussian_filter(volume, sigma=smoothness)

            if np.max(volume) < threshold:
                return None

            verts, faces, _, _ = marching_cubes(volume, level=threshold, allow_degenerate=False)

            # Transform to world coordinates
            verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
            verts_world = (affine @ verts_h.T).T[:, :3]

            return verts_world, faces
        except Exception as e:
            print(f"Warning: isosurface extraction failed: {e}")
            return None

    def _generate_3d_isosurface_figure(self) -> Optional[str]:
        """Generate a 3D Plotly figure with isosurfaces and electrode trajectories. Returns JSON."""
        if self.nifti_4d_data is None or self.nifti_4d_affine is None:
            return None

        fig = go.Figure()
        mesh_count = 0

        # Extract and add isosurface for each volume
        for vol_idx in range(self.num_volumes):
            if not self._is_volume_visible(vol_idx):
                print(f"  Volume {vol_idx}: skipped (hidden by label file)")
                continue

            vol_data = self.nifti_4d_data[..., vol_idx]
            vol_max = float(np.max(vol_data))
            vol_min = float(np.min(vol_data))
            print(f"  Volume {vol_idx}: range [{vol_min:.4f}, {vol_max:.4f}]")

            if vol_max < 1e-10:
                print(f"  Volume {vol_idx}: skipped (empty)")
                continue

            threshold = self.threshold

            print(f"  Volume {vol_idx}: extracting isosurface at threshold {threshold:.4f}...")
            result = self._extract_isosurface(
                vol_data, self.nifti_4d_affine,
                threshold=threshold, smoothness=0.5,
            )
            if result is None:
                print(f"  Volume {vol_idx}: isosurface extraction returned None")
                continue

            verts, faces = result
            print(f"  Volume {vol_idx}: {len(verts)} vertices, {len(faces)} faces")

            color = self._get_volume_color(vol_idx)
            label = self.volume_labels[vol_idx] if vol_idx < len(self.volume_labels) else f'Volume {vol_idx}'

            fig.add_trace(go.Mesh3d(
                x=verts[:, 0].tolist(),
                y=verts[:, 1].tolist(),
                z=verts[:, 2].tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
                color=color,
                opacity=0.5,
                name=label,
                showlegend=True,
            ))
            mesh_count += 1

        if mesh_count == 0:
            print("  No isosurfaces extracted from any volume")

        # Add electrode cylinders and contact cylinders
        for electrode in self.electrodes:
            trajectory = electrode['trajectory']
            contacts = electrode.get('contacts')
            label = electrode['label']
            side = electrode.get('side', '')

            # --- Trajectory cylinder mesh ---
            if len(trajectory) > 1:
                radius = 0.625  # 1.25mm diameter
                n_sides = 8
                all_verts = []
                all_faces = []
                v_offset = 0

                for seg_idx in range(len(trajectory) - 1):
                    p1, p2 = trajectory[seg_idx], trajectory[seg_idx + 1]
                    direction = p2 - p1
                    length = np.linalg.norm(direction)
                    if length < 1e-6:
                        continue
                    direction = direction / length

                    perp1 = np.array([1, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1, 0])
                    perp1 = perp1 - np.dot(perp1, direction) * direction
                    perp1 = perp1 / np.linalg.norm(perp1)
                    perp2 = np.cross(direction, perp1)

                    for point in [p1, p2]:
                        for i in range(n_sides):
                            angle = 2 * np.pi * i / n_sides
                            all_verts.append(point + radius * np.cos(angle) * perp1 + radius * np.sin(angle) * perp2)

                    for i in range(n_sides):
                        ni = (i + 1) % n_sides
                        v1, v2 = v_offset + i, v_offset + ni
                        v3, v4 = v_offset + n_sides + i, v_offset + n_sides + ni
                        all_faces.append([v1, v3, v2])
                        all_faces.append([v2, v3, v4])
                    v_offset += 2 * n_sides

                if all_verts:
                    verts = np.array(all_verts)
                    faces = np.array(all_faces)
                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
                        i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
                        color='grey', opacity=1.0,
                        name=f'{label}', showlegend=True, hoverinfo='skip',
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                        flatshading=False,
                    ))

            # --- Contact cylinders ---
            if contacts is not None and len(trajectory) > 1:
                contact_radius = 0.635
                contact_height = 1.5
                c_n_sides = 24

                for ci, contact_pos in enumerate(contacts):
                    distances = np.linalg.norm(trajectory - contact_pos, axis=1)
                    closest_idx = np.argmin(distances)
                    if closest_idx < len(trajectory) - 1:
                        c_dir = trajectory[closest_idx + 1] - trajectory[closest_idx]
                    else:
                        c_dir = trajectory[closest_idx] - trajectory[closest_idx - 1]
                    c_dir = c_dir / np.linalg.norm(c_dir)

                    perp1 = np.array([1, 0, 0]) if abs(c_dir[0]) < 0.9 else np.array([0, 1, 0])
                    perp1 = perp1 - np.dot(perp1, c_dir) * c_dir
                    perp1 = perp1 / np.linalg.norm(perp1)
                    perp2 = np.cross(c_dir, perp1)

                    bottom = contact_pos - c_dir * (contact_height / 2)
                    top = contact_pos + c_dir * (contact_height / 2)

                    c_verts = []
                    for center in [bottom, top]:
                        for i in range(c_n_sides):
                            angle = 2 * np.pi * i / c_n_sides
                            c_verts.append(center + contact_radius * np.cos(angle) * perp1 + contact_radius * np.sin(angle) * perp2)

                    c_faces = []
                    for i in range(c_n_sides):
                        ni = (i + 1) % c_n_sides
                        c_faces.append([i, c_n_sides + i, ni])
                        c_faces.append([ni, c_n_sides + i, c_n_sides + ni])

                    c_verts = np.array(c_verts)
                    c_faces = np.array(c_faces)
                    fig.add_trace(go.Mesh3d(
                        x=c_verts[:, 0].tolist(), y=c_verts[:, 1].tolist(), z=c_verts[:, 2].tolist(),
                        i=c_faces[:, 0].tolist(), j=c_faces[:, 1].tolist(), k=c_faces[:, 2].tolist(),
                        color='red', opacity=1.0,
                        name=f'C{ci}', showlegend=False, hoverinfo='skip',
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                        flatshading=False,
                    ))

            # --- Orientation marker vectors ---
            markers = self._get_orientation_markers(electrode)
            vector_length_mm = 5.0
            for mk, (pos, m_dir) in markers.items():
                end = pos + m_dir * vector_length_mm
                m_color = self.MARKER_COLORS.get(mk, '#888')

                # Vector line
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
                    mode='lines',
                    line=dict(color=m_color, width=3),
                    name=f'{label} Marker {mk}',
                    showlegend=False, hoverinfo='skip',
                ))
                # Arrowhead cone
                fig.add_trace(go.Cone(
                    x=[end[0]], y=[end[1]], z=[end[2]],
                    u=[m_dir[0]], v=[m_dir[1]], w=[m_dir[2]],
                    sizemode='absolute', sizeref=1.0, showscale=False,
                    colorscale=[[0, m_color], [1, m_color]],
                    showlegend=False, hoverinfo='skip',
                ))
                # Floating label
                fig.add_trace(go.Scatter3d(
                    x=[end[0]], y=[end[1]], z=[end[2]],
                    mode='text', text=[mk],
                    textposition='top center',
                    textfont=dict(size=12, color=m_color),
                    showlegend=False, hoverinfo='skip',
                ))

            # --- Hemisphere label below the tip ---
            tip = electrode['tip']
            direction = electrode['direction']
            # Offset label below the tip (opposite to entry direction)
            label_pos = tip - direction * 3.0
            hemisphere = 'Right' if side.lower() == 'right' else ('Left' if side.lower() == 'left' else '')
            if hemisphere:
                fig.add_trace(go.Scatter3d(
                    x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
                    mode='text',
                    text=[hemisphere],
                    textfont=dict(size=12, color='black'),
                    showlegend=False, hoverinfo='skip',
                ))

        # --- AC-PC landmarks in 3D ---
        if self.acpc_landmarks is not None:
            ac = self.acpc_landmarks['AC']
            pc = self.acpc_landmarks['PC']
            mcp = (ac + pc) / 2.0
            acpc_color = '#e6e600'

            # Line AC → PC
            fig.add_trace(go.Scatter3d(
                x=[ac[0], pc[0]], y=[ac[1], pc[1]], z=[ac[2], pc[2]],
                mode='lines',
                line=dict(color=acpc_color, width=3, dash='dash'),
                name='AC-PC', showlegend=True, hoverinfo='skip',
            ))
            # Markers: AC, PC, MCP
            fig.add_trace(go.Scatter3d(
                x=[ac[0], pc[0], mcp[0]],
                y=[ac[1], pc[1], mcp[1]],
                z=[ac[2], pc[2], mcp[2]],
                mode='markers+text',
                marker=dict(size=4, color=acpc_color, symbol='diamond'),
                text=['AC', 'PC', 'MCP'],
                textposition='top center',
                textfont=dict(size=10, color=acpc_color),
                showlegend=False, hoverinfo='text',
            ))

        if len(fig.data) == 0:
            return None

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                bgcolor='white',
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#333'),
            legend=dict(bgcolor='rgba(255,255,255,0.8)'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=600,
        )

        return fig.to_plotly_json()

    # ------------------------------------------------------------------
    # HTML report generation
    # ------------------------------------------------------------------

    def generate_html(self) -> str:
        """Generate the full HTML report."""
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ct_file = self.electrode_metadata.get('ct_file', '')
        ct_filename = Path(ct_file).name if ct_file else 'N/A'
        pypacer_version = self.electrode_metadata.get('pypacer_version', 'N/A')

        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>DBS Clinical Report</title>',
            '    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; display: flex; justify-content: center; }',
            '        .container { max-width: 1200px; width: 100%; }',
            '        .header { position: relative; text-align: center; margin-bottom: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '        .header-disclaimer { position: absolute; top: 14px; right: 18px; font-size: 16px; color: #b57a00; max-width: 330px; text-align: right; line-height: 1.4; padding: 9px 15px; background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px; }',
            '        .header-disclaimer strong { color: #e65100; }',
            '        .header h1 { margin: 0 0 10px 0; color: #333; }',
            '        .header p { margin: 5px 0; color: #666; }',
            '        .source-data { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }',
            '        .source-data h2 { margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }',
            '        .source-data table { width: 100%; border-collapse: collapse; }',
            '        .source-data td { padding: 6px 12px; color: #555; }',
            '        .source-data td:first-child { font-weight: 600; color: #333; width: 180px; }',
            '        .section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }',
            '        .section h2 { margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }',
            '        .contact-strip { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }',
            '        .contact-card { text-align: center; }',
            '        .contact-card .contact-label { font-weight: 600; font-size: 14px; color: #333; margin-bottom: 4px; }',
            '        .contact-card .contact-ras { font-size: 11px; color: #999; margin-bottom: 6px; }',
            '        .contact-card .slice-stack { position: relative; width: 200px; height: 200px; }',
            '        .contact-card .slice-stack img { position: absolute; top: 0; left: 0; width: 200px; height: 200px; border-radius: 6px; border: 2px solid #ddd; }',
            '        .contact-card .slice-stack .overlay-img { pointer-events: none; }',
            '        .overlay-toggle { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 12px; color: #555; margin-top: 8px; }',
            '        .overlay-toggle:hover { background: #f0f0f0; }',
            '        .overlay-toggle.active { background: #e8f5e9; border-color: #4caf50; color: #2e7d32; }',
            '        .no-data { color: #999; font-style: italic; padding: 20px; }',
            '        .plot-3d-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }',
            '        .plot-3d-section h2 { margin-top: 0; color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }',
            '        .volume-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }',
            '        .volume-legend-item { display: flex; align-items: center; gap: 5px; font-size: 13px; color: #555; }',
            '        .volume-legend-swatch { width: 14px; height: 14px; border-radius: 3px; }',
            '        .section-header { display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 12px; }',
            '        .section-header h2 { margin: 0; color: #333; }',
            '        .maximize-btn { padding: 4px 10px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 18px; color: #555; line-height: 1; }',
            '        .maximize-btn:hover { background: #f0f0f0; }',
            '        .plot-3d-section.maximized { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: 9999; margin: 0; border-radius: 0; padding: 10px; box-sizing: border-box; }',
            '        .plot-3d-section.maximized #plot3d { height: calc(100vh - 60px) !important; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <div class="container">',
            '',
            '    <!-- Header -->',
            '    <div class="header">',
            '        <div class="header-disclaimer"><strong>&#9888; Research Use Only</strong><br>Not cleared for clinical or diagnostic use</div>',
            '        <h1>DBS Clinical Report</h1>',
            *([ f'        <p><strong>Patient:</strong> {self.patient_id}</p>'] if self.patient_id else []),
            f'        <p>Generated: {report_date}</p>',
            f'        <p>Generated with <a href="https://github.com/mvpetersen/dbs-toolbox" style="color: #2196F3;">The DBS Toolbox v{__version__}</a></p>',
            '    </div>',
            '',
            '    <!-- Source Data -->',
            '    <div class="source-data">',
            '        <h2>Source Data</h2>',
            '        <div style="display: flex; gap: 20px; flex-wrap: wrap;">',
            '',
            '        <!-- Left column -->',
            '        <div style="flex: 1; min-width: 300px;">',
            '        <table>',
        ]

        if self.nifti_3d_filename:
            html_parts.append(f'            <tr><td>3D NIfTI</td><td>{self.nifti_3d_filename}</td></tr>')
        if self.nifti_4d_filename:
            html_parts.append(f'            <tr><td>4D NIfTI</td><td>{self.nifti_4d_filename} ({self.num_volumes} volumes)</td></tr>')
        html_parts.extend([
            f'            <tr><td>CT File</td><td>{ct_filename}</td></tr>',
            f'            <tr><td>Electrodes</td><td>{len(self.electrodes)}</td></tr>',
            f'            <tr><td>PyPaCER Version</td><td>{pypacer_version}</td></tr>',
            f'            <tr><td>Threshold</td><td>{self.threshold:.2f}</td></tr>' if self.num_volumes > 0 else '',
            '        </table>',
            '        </div>',
        ])

        # Right column: AC-PC info
        if self.acpc_landmarks:
            ac = self.acpc_landmarks['AC']
            pc = self.acpc_landmarks['PC']
            mid = self.acpc_landmarks['midline']
            acpc_dist = float(np.linalg.norm(pc - ac))

            # Build export data
            acpc_export = {
                'landmarks_world': {
                    'AC': [round(float(v), 4) for v in ac],
                    'PC': [round(float(v), 4) for v in pc],
                    'midline': [round(float(v), 4) for v in mid],
                },
                'landmarks_acpc': {
                    'AC': [0.0, 0.0, 0.0],
                    'PC': [0.0, round(-acpc_dist, 2), 0.0],
                    'MCP': [0.0, round(-acpc_dist / 2, 2), 0.0],
                },
                'ac_pc_distance_mm': round(acpc_dist, 2),
                'electrodes': {},
            }
            for electrode in self.electrodes:
                contacts = electrode.get('contacts')
                if contacts is None:
                    continue
                elec_contacts = {}
                for ci in range(min(4, len(contacts))):
                    result = self._world_to_acpc(contacts[ci])
                    if result is not None:
                        coords, y_norm = result
                        elec_contacts[f'C{ci}'] = {
                            'world': [round(float(v), 4) for v in contacts[ci]],
                            'acpc': [round(float(v), 2) for v in coords],
                            'y_norm': round(float(y_norm), 4),
                        }
                acpc_export['electrodes'][electrode['label']] = elec_contacts

            acpc_json_str = json.dumps(acpc_export, indent=2)

            html_parts.extend([
                '',
                '        <!-- Right column: AC-PC -->',
                '        <div style="flex: 1; min-width: 300px;">',
                '        <table>',
                f'            <tr><td>AC</td><td>[{ac[0]:.2f}, {ac[1]:.2f}, {ac[2]:.2f}]</td></tr>',
                f'            <tr><td>PC</td><td>[{pc[0]:.2f}, {pc[1]:.2f}, {pc[2]:.2f}]</td></tr>',
                f'            <tr><td>Midline</td><td>[{mid[0]:.2f}, {mid[1]:.2f}, {mid[2]:.2f}]</td></tr>',
                f'            <tr><td>AC-PC Distance</td><td>{acpc_dist:.2f} mm</td></tr>',
                '        </table>',
                '        <button onclick="downloadAcpcJson()" '
                'style="margin-top: 10px; padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; '
                'background: white; cursor: pointer; font-size: 12px; color: #555;">'
                '&#x2B07; Export AC-PC data (JSON)</button>',
                '        <script>',
                '        function downloadAcpcJson() {',
                f'            var data = {acpc_json_str};',
                '            var blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});',
                '            var url = URL.createObjectURL(blob);',
                '            var a = document.createElement("a");',
                '            a.href = url;',
                '            a.download = "acpc_coordinates.json";',
                '            a.click();',
                '            URL.revokeObjectURL(url);',
                '        }',
                '        </script>',
                '        </div>',
            ])

        html_parts.extend([
            '',
            '        </div>',  # flex container
            '    </div>',  # source-data
            '',
        ])

        # Electrode cards with axial slices (3D NIfTI)
        for elec_idx, electrode in enumerate(self.electrodes):
            contacts = electrode.get('contacts')
            contact_depths = self._calculate_contact_depths(electrode)
            tip = electrode['tip']
            direction = electrode['direction']

            # Pre-render all contact images to know which volumes intersect
            has_seg_overlay = False
            has_arrow_overlay = False
            has_acpc_overlay = False
            contact_images = []  # list of (ci, ras_str, acpc_str, acpc_ynorm_str, base_b64, seg_b64, arrow_b64, acpc_ov_b64)
            elec_intersecting_vols = set()

            has_any_nifti = self.nifti_3d_data is not None or self.nifti_4d_data is not None

            # Use AC-PC Z-axis as slice plane normal if available
            acpc_axes = self._get_acpc_axes()
            slice_normal = acpc_axes[2] if acpc_axes is not None else None

            if contacts is not None and has_any_nifti:
                for ci in reversed(range(min(4, len(contacts)))):
                    contact_pos = contacts[ci]
                    ras_str = f'{contact_pos[0]:.1f}, {contact_pos[1]:.1f}, {contact_pos[2]:.1f}'

                    # AC-PC coordinates
                    acpc_str = None
                    acpc_ynorm_str = None
                    acpc_result = self._world_to_acpc(contact_pos)
                    if acpc_result is not None:
                        acpc_coords, y_norm = acpc_result
                        acpc_str = f'{acpc_coords[0]:.1f}, {acpc_coords[1]:.1f}, {acpc_coords[2]:.1f}'
                        acpc_ynorm_str = f'{y_norm:.2f}'

                    # Only C1 and C2 are directional contacts
                    is_directional = ci in (1, 2)

                    print(f"Rendering axial slice for {electrode['label']} C{ci}...")
                    base_b64, seg_b64, arrow_b64, acpc_ov_b64, vol_indices = self._render_axial_contact_image(
                        contact_pos, electrode=electrode, draw_arrows=is_directional,
                        plane_normal=slice_normal,
                    )
                    if seg_b64:
                        has_seg_overlay = True
                    if arrow_b64:
                        has_arrow_overlay = True
                    if acpc_ov_b64:
                        has_acpc_overlay = True
                    elec_intersecting_vols.update(vol_indices)
                    contact_images.append((ci, ras_str, acpc_str, acpc_ynorm_str, base_b64, seg_b64, arrow_b64, acpc_ov_b64))

            # Card header
            html_parts.append(f'    <div class="section">')
            html_parts.append(f'        <h2>{electrode["label"]}</h2>')

            if contact_images:
                html_parts.append('        <div class="contact-strip">')

                # Coronal reference plot (first in strip)
                print(f"Rendering coronal reference for {electrode['label']}...")
                coronal_b64 = self._render_coronal_reference(electrode)
                if coronal_b64:
                    html_parts.append('            <div class="contact-card" style="align-self: flex-end;">')
                    html_parts.append(f'                <img src="data:image/png;base64,{coronal_b64}" '
                                      f'width="200" height="200" style="border-radius: 6px; border: 2px solid #ddd;" />')
                    html_parts.append('            </div>')

                for ci, ras_str, acpc_str, acpc_ynorm_str, base_b64, seg_b64, arrow_b64, acpc_ov_b64 in contact_images:
                    html_parts.append('            <div class="contact-card">')
                    html_parts.append(f'                <div class="contact-label">C{ci}</div>')
                    html_parts.append(f'                <div class="contact-ras">[{ras_str}]</div>')
                    if acpc_str:
                        html_parts.append(f'                <div class="contact-ras" style="color: #6a5acd;">ACPC [{acpc_str}] <span style="color: #888;">y&#x0302;={acpc_ynorm_str}</span></div>')
                    if base_b64:
                        html_parts.append(f'                <div class="slice-stack">')
                        html_parts.append(f'                    <img src="data:image/png;base64,{base_b64}" />')
                        if seg_b64:
                            html_parts.append(
                                f'                    <img class="overlay-img seg-e{elec_idx}" '
                                f'src="data:image/png;base64,{seg_b64}" />'
                            )
                        if arrow_b64:
                            html_parts.append(
                                f'                    <img class="overlay-img arrow-e{elec_idx}" '
                                f'src="data:image/png;base64,{arrow_b64}" />'
                            )
                        if acpc_ov_b64:
                            html_parts.append(
                                f'                    <img class="overlay-img acpc-e{elec_idx}" '
                                f'src="data:image/png;base64,{acpc_ov_b64}" />'
                            )
                        html_parts.append(f'                </div>')
                    else:
                        html_parts.append('                <div class="no-data">No image</div>')
                    html_parts.append('            </div>')

                html_parts.append('        </div>')

                # Controls and legend row below contact strip
                html_parts.append('        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 8px;">')

                # Segmentation toggle + legend
                if has_seg_overlay:
                    html_parts.append(
                        f'            <button class="overlay-toggle active" '
                        f'onclick="toggleSegOverlay({elec_idx})" id="seg_btn_{elec_idx}">'
                        f'Segmentation</button>'
                    )

                # Arrow toggle
                if has_arrow_overlay:
                    html_parts.append(
                        f'            <button class="overlay-toggle active" '
                        f'onclick="toggleArrowOverlay({elec_idx})" id="arrow_btn_{elec_idx}">'
                        f'Directional markers</button>'
                    )

                # AC-PC toggle
                if has_acpc_overlay:
                    html_parts.append(
                        f'            <button class="overlay-toggle active" '
                        f'onclick="toggleAcpcOverlay({elec_idx})" id="acpc_btn_{elec_idx}">'
                        f'AC-PC</button>'
                    )

                html_parts.append('        </div>')

                # Combined legend row
                legend_sections = [elec_intersecting_vols, has_arrow_overlay, has_acpc_overlay]
                if any(legend_sections):
                    html_parts.append('        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 6px;">')
                    divider_needed = False

                    # Volume legend (hidden when seg overlay is hidden)
                    if elec_intersecting_vols:
                        html_parts.append(
                            f'            <div class="volume-legend" id="vol_legend_{elec_idx}" style="margin: 0;">'
                        )
                        for vi in sorted(elec_intersecting_vols):
                            color = self._get_volume_color(vi)
                            lbl = self.volume_labels[vi] if vi < len(self.volume_labels) else f'Volume {vi}'
                            html_parts.append(
                                f'                <div class="volume-legend-item">'
                                f'<div class="volume-legend-swatch" style="background: {color};"></div>'
                                f'{lbl}</div>'
                            )
                        html_parts.append('            </div>')
                        divider_needed = True

                    # Arrow legend (hidden when arrow overlay is hidden)
                    if has_arrow_overlay:
                        if divider_needed:
                            html_parts.append('            <div style="width: 1px; height: 20px; background: #ddd;"></div>')
                        html_parts.append(
                            f'            <div class="volume-legend" id="arrow_legend_{elec_idx}" style="margin: 0;">'
                        )
                        for mk in ('A', 'B', 'C'):
                            color = self.MARKER_COLORS.get(mk, '#888')
                            html_parts.append(
                                f'                <div class="volume-legend-item">'
                                f'<div class="volume-legend-swatch" style="background: {color};"></div>'
                                f'Group {mk}</div>'
                            )
                        html_parts.append('            </div>')
                        divider_needed = True

                    # ACPC legend (hidden when acpc overlay is hidden)
                    if has_acpc_overlay:
                        if divider_needed:
                            html_parts.append('            <div style="width: 1px; height: 20px; background: #ddd;"></div>')
                        html_parts.append(
                            f'            <div class="volume-legend" id="acpc_legend_{elec_idx}" style="margin: 0;">'
                        )
                        acpc_legend_colors = {'AC': '#ffd700', 'PC': '#daa520', 'MCP': '#ffe44d'}
                        for lbl, clr in acpc_legend_colors.items():
                            html_parts.append(
                                f'                <div class="volume-legend-item">'
                                f'<div class="volume-legend-swatch" style="background: {clr};"></div>'
                                f'{lbl}</div>'
                            )
                        html_parts.append('            </div>')

                    html_parts.append('        </div>')
            elif not has_any_nifti and contacts is not None:
                # No NIfTI at all - just show contact info
                html_parts.append('        <div class="contact-strip">')
                for ci in reversed(range(min(4, len(contacts)))):
                    contact_pos = contacts[ci]
                    ras_str = f'{contact_pos[0]:.1f}, {contact_pos[1]:.1f}, {contact_pos[2]:.1f}'
                    html_parts.append('            <div class="contact-card">')
                    html_parts.append(f'                <div class="contact-label">C{ci}</div>')
                    html_parts.append(f'                <div class="contact-ras">[{ras_str}]</div>')
                    acpc_result = self._world_to_acpc(contact_pos)
                    if acpc_result is not None:
                        acpc_coords, y_norm = acpc_result
                        acpc_str = f'{acpc_coords[0]:.1f}, {acpc_coords[1]:.1f}, {acpc_coords[2]:.1f}'
                        acpc_ynorm_str = f'{y_norm:.2f}'
                        html_parts.append(f'                <div class="contact-ras" style="color: #6a5acd;">ACPC [{acpc_str}] <span style="color: #888;">y&#x0302;={acpc_ynorm_str}</span></div>')
                    html_parts.append(f'                <div class="no-data">No NIfTI loaded</div>')
                    html_parts.append('            </div>')
                html_parts.append('        </div>')
            else:
                html_parts.append('        <p class="no-data">No contact positions available</p>')

            html_parts.extend([
                '    </div>',
                '',
            ])

        # 3D isosurface plot (4D NIfTI)
        if self.nifti_4d_data is not None:
            print("Generating 3D isosurface figure...")
            fig_dict = self._generate_3d_isosurface_figure()

            if fig_dict:
                fig_json_str = json.dumps(fig_dict)
                html_parts.extend([
                    '    <div class="plot-3d-section" id="plot3d_section">',
                    '        <div class="section-header">',
                    '            <h2>3D View</h2>',
                    '            <button class="maximize-btn" id="maximize_btn" onclick="toggleMaximize()" title="Maximize">&#x26F6;</button>',
                    '        </div>',
                    '        <div id="plot3d" style="width: 100%; height: 600px;"></div>',
                    '        <script>',
                    f'            var figData = {fig_json_str};',
                    '            Plotly.newPlot("plot3d", figData.data, figData.layout, {responsive: true});',
                    '        </script>',
                    '    </div>',
                    '',
                ])

        # JavaScript
        html_parts.extend([
            '    <script>',
            '        function toggleSegOverlay(elecIdx) {',
            '            var btn = document.getElementById("seg_btn_" + elecIdx);',
            '            var overlays = document.querySelectorAll(".seg-e" + elecIdx);',
            '            var legend = document.getElementById("vol_legend_" + elecIdx);',
            '            var visible = btn.classList.contains("active");',
            '            overlays.forEach(function(img) {',
            '                img.style.display = visible ? "none" : "block";',
            '            });',
            '            if (legend) legend.style.display = visible ? "none" : "flex";',
            '            if (visible) { btn.classList.remove("active"); }',
            '            else { btn.classList.add("active"); }',
            '        }',
            '',
            '        function toggleArrowOverlay(elecIdx) {',
            '            var btn = document.getElementById("arrow_btn_" + elecIdx);',
            '            var overlays = document.querySelectorAll(".arrow-e" + elecIdx);',
            '            var legend = document.getElementById("arrow_legend_" + elecIdx);',
            '            var visible = btn.classList.contains("active");',
            '            overlays.forEach(function(img) {',
            '                img.style.display = visible ? "none" : "block";',
            '            });',
            '            if (legend) legend.style.display = visible ? "none" : "flex";',
            '            if (visible) { btn.classList.remove("active"); }',
            '            else { btn.classList.add("active"); }',
            '        }',
            '',
            '        function toggleAcpcOverlay(elecIdx) {',
            '            var btn = document.getElementById("acpc_btn_" + elecIdx);',
            '            var overlays = document.querySelectorAll(".acpc-e" + elecIdx);',
            '            var legend = document.getElementById("acpc_legend_" + elecIdx);',
            '            var visible = btn.classList.contains("active");',
            '            overlays.forEach(function(img) {',
            '                img.style.display = visible ? "none" : "block";',
            '            });',
            '            if (legend) legend.style.display = visible ? "none" : "flex";',
            '            if (visible) { btn.classList.remove("active"); }',
            '            else { btn.classList.add("active"); }',
            '        }',
            '',
            '        function toggleMaximize() {',
            '            var section = document.getElementById("plot3d_section");',
            '            var btn = document.getElementById("maximize_btn");',
            '            if (!section) return;',
            '            section.classList.toggle("maximized");',
            '            var isMax = section.classList.contains("maximized");',
            '            btn.innerHTML = isMax ? "&#x2716;" : "&#x26F6;";',
            '            btn.title = isMax ? "Minimize" : "Maximize";',
            '            // Resize plotly to fit new container',
            '            setTimeout(function() { Plotly.Plots.resize("plot3d"); }, 100);',
            '        }',
            '    </script>',
            '',
            '    </div>',  # container
            '</body>',
            '</html>',
        ])

        return '\n'.join(html_parts)

    def save_and_download(self, filename_prefix: str = "stimulation_report") -> Tuple[str, str]:
        """Generate HTML report and save to a temporary file."""
        html_content = self.generate_html()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build a name from available info
        name_parts = []
        if self.patient_id:
            name_parts.append(self.patient_id)
        if self.nifti_3d_filename:
            name_parts.append(Path(self.nifti_3d_filename).stem.replace('.nii', ''))
        elif self.nifti_4d_filename:
            name_parts.append(Path(self.nifti_4d_filename).stem.replace('.nii', ''))
        if not name_parts:
            name_parts.append('report')
        clean_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in '_'.join(name_parts))

        filename = f'{filename_prefix}_{clean_name}_{timestamp}.html'

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return temp_path, filename
