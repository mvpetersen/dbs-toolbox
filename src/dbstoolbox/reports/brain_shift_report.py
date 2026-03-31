"""Brain shift analysis report generation."""

from typing import List, Dict, Optional, Tuple
import json
import numpy as np
import plotly.graph_objects as go


class BrainShiftAnalyzer:
    """
    Analyzes brain shift by comparing two electrode reconstructions.

    This class takes two sets of electrode trajectories (e.g., pre-op and post-op,
    or intra-op at different time points) and calculates displacement between
    corresponding contacts, generates polar charts relative to a virtual trajectory,
    and produces HTML visualization of the brain shift.
    """

    def __init__(
        self,
        electrode_trajectories_1: List[Dict],
        electrode_trajectories_2: List[Dict],
        metadata_1: Optional[Dict] = None,
        metadata_2: Optional[Dict] = None
    ):
        """
        Initialize the brain shift analyzer.

        Args:
            electrode_trajectories_1: First set of electrode trajectories (e.g., pre-op)
            electrode_trajectories_2: Second set of electrode trajectories (e.g., post-op)
            metadata_1: Metadata from first reconstruction
            metadata_2: Metadata from second reconstruction
        """
        self.electrode_trajectories_1 = electrode_trajectories_1
        self.electrode_trajectories_2 = electrode_trajectories_2
        self.metadata_1 = metadata_1 or {}
        self.metadata_2 = metadata_2 or {}
        self.matched_pairs = self._match_electrodes()

        # Define color palettes for each reconstruction (red shades for recon 1, blue shades for recon 2)
        self.colors_recon_1 = [
            '#FF4444',  # Bright red
            '#CC0000',  # Dark red
            '#FF6B6B',  # Light red
            '#DC143C',  # Crimson
            '#8B0000',  # Dark red
        ]
        self.colors_recon_2 = [
            '#4477FF',  # Bright blue
            '#0044CC',  # Dark blue
            '#6B9BFF',  # Light blue
            '#1E90FF',  # Dodger blue
            '#00008B',  # Dark blue
        ]

    def _match_electrodes(self) -> List[Tuple[Dict, Dict]]:
        """
        Match electrodes between two reconstructions by label.

        Returns:
            List of tuples (electrode_1, electrode_2) for matched pairs
        """
        matched_pairs = []

        # Create label lookup for second set
        electrode_2_map = {
            electrode.get('label', ''): electrode
            for electrode in self.electrode_trajectories_2
        }

        # Match by label
        for electrode_1 in self.electrode_trajectories_1:
            label = electrode_1.get('label', '')
            if label and label in electrode_2_map:
                electrode_2 = electrode_2_map[label]
                matched_pairs.append((electrode_1, electrode_2))

        return matched_pairs

    def _calculate_virtual_trajectory(
        self,
        electrode_1: Dict,
        electrode_2: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate virtual trajectory as average of two electrode trajectories.

        Args:
            electrode_1: First electrode dictionary
            electrode_2: Second electrode dictionary

        Returns:
            Tuple of (entry_virtual, target_virtual, direction_virtual,
                     anterior_perp, lateral_perp,
                     anterior_axial, lateral_axial, plane_warning)
        """
        traj_1 = electrode_1.get('trajectory')
        traj_2 = electrode_2.get('trajectory')

        if traj_1 is None or traj_2 is None or len(traj_1) == 0 or len(traj_2) == 0:
            return None, None, None, None, None

        # Use explicit entry and tip positions if available, otherwise infer from trajectory
        entry_1 = electrode_1.get('entry_position')
        tip_1 = electrode_1.get('tip_position')
        entry_2 = electrode_2.get('entry_position')
        tip_2 = electrode_2.get('tip_position')

        # Fallback to trajectory endpoints if entry/tip not available
        if entry_1 is None or tip_1 is None:
            if traj_1[0][2] < traj_1[-1][2]:
                # First point has lower Z (deeper), so it's the tip
                entry_1 = traj_1[-1]
                tip_1 = traj_1[0]
            else:
                # First point has higher Z (superior), so it's the entry
                entry_1 = traj_1[0]
                tip_1 = traj_1[-1]

        if entry_2 is None or tip_2 is None:
            if traj_2[0][2] < traj_2[-1][2]:
                # First point has lower Z (deeper), so it's the tip
                entry_2 = traj_2[-1]
                tip_2 = traj_2[0]
            else:
                # First point has higher Z (superior), so it's the entry
                entry_2 = traj_2[0]
                tip_2 = traj_2[-1]

        # Calculate virtual entry and target (tip)
        entry_virtual = (entry_1 + entry_2) / 2.0
        target_virtual = (tip_1 + tip_2) / 2.0  # Tip is the target/deepest point

        # Calculate virtual trajectory direction (from target/tip to entry)
        # This is used for depth calculation where depth=0 at tip, positive towards entry
        direction_virtual = entry_virtual - target_virtual
        direction_virtual = direction_virtual / np.linalg.norm(direction_virtual)

        # Calculate both coordinate systems and return both
        # We'll pass both to the HTML and let JavaScript toggle between them

        # METHOD 1: Perpendicular plane (trajectory-aligned)
        # Anterior = intersection of perpendicular plane with sagittal plane (X=0)
        # This is the line perpendicular to both direction_virtual and [1,0,0]
        sagittal_normal = np.array([1.0, 0.0, 0.0])

        # Check if trajectory is parallel to sagittal plane normal (purely lateral)
        if abs(np.dot(direction_virtual, sagittal_normal)) > 0.95:
            # Trajectory is too lateral, use coronal plane instead
            anterior_perp = np.cross(direction_virtual, np.array([0.0, 1.0, 0.0]))
            anterior_perp = anterior_perp / np.linalg.norm(anterior_perp)
        else:
            # Normal case: anterior is the intersection line
            anterior_perp = np.cross(direction_virtual, sagittal_normal)
            anterior_perp = anterior_perp / np.linalg.norm(anterior_perp)

        # Ensure anterior_perp points more toward [0,1,0] than [0,-1,0]
        # (i.e., Y-component should be positive)
        if anterior_perp[1] < 0:
            anterior_perp = -anterior_perp

        # Lateral is perpendicular to both direction and anterior
        # Use cross(anterior, direction) to match right-hand convention with axial plane
        lateral_perp = np.cross(anterior_perp, direction_virtual)
        lateral_perp = lateral_perp / np.linalg.norm(lateral_perp)

        # METHOD 2: Axial plane (pure image axes)
        # Always use EXACTLY the axial plane of the image data
        anterior_axial = np.array([0.0, 1.0, 0.0])  # Pure anterior (Y-axis)
        lateral_axial = np.array([1.0, 0.0, 0.0])   # Pure lateral (X-axis)
        plane_warning = None

        return (entry_virtual, target_virtual, direction_virtual,
                anterior_perp, lateral_perp,
                anterior_axial, lateral_axial, plane_warning)

    def _calculate_contact_displacements(
        self,
        electrode_1: Dict,
        electrode_2: Dict
    ) -> List[Dict]:
        """
        Calculate displacement between corresponding contacts.

        Args:
            electrode_1: First electrode dictionary with contacts
            electrode_2: Second electrode dictionary with contacts

        Returns:
            List of displacement dictionaries for each contact
        """
        contacts_1 = electrode_1.get('contacts')
        contacts_2 = electrode_2.get('contacts')

        displacements = []

        # Check if contacts exist for both electrodes
        if contacts_1 is None or contacts_2 is None:
            return displacements

        # Match contacts by index (assuming same contact numbering)
        for i in range(min(len(contacts_1), len(contacts_2))):
            pos_1 = contacts_1[i]
            pos_2 = contacts_2[i]

            # Calculate displacement vector
            displacement_vector = pos_2 - pos_1
            euclidean_distance = float(np.linalg.norm(displacement_vector))

            displacements.append({
                'contact_index': i,
                'contact_label': f'C{i}',
                'position_1': pos_1,
                'position_2': pos_2,
                'displacement_vector': displacement_vector,
                'euclidean_distance': euclidean_distance,
                'dx': float(displacement_vector[0]),
                'dy': float(displacement_vector[1]),
                'dz': float(displacement_vector[2])
            })

        return displacements

    def _calculate_position_at_depth(
        self,
        electrode: Dict,
        entry_virtual: np.ndarray,
        target_virtual: np.ndarray,
        direction_virtual: np.ndarray,
        anterior_virtual: np.ndarray,
        lateral_virtual: np.ndarray,
        depth_mm: float,
        plane_normal: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Calculate electrode position at a given depth along virtual trajectory.

        Similar to stereotactic targeting, but relative to virtual trajectory.

        Args:
            electrode: Electrode dictionary
            entry_virtual: Virtual trajectory entry point
            target_virtual: Virtual trajectory target point (tip)
            direction_virtual: Virtual trajectory direction (from target to entry)
            anterior_virtual: Anterior vector for plane
            lateral_virtual: Lateral vector for plane
            depth_mm: Depth along virtual trajectory in mm (0 at target, positive towards entry)
            plane_normal: Normal vector for the plane (defaults to direction_virtual)

        Returns:
            Dictionary with position info (r, theta, lat, ant, etc.) or None
        """
        trajectory = electrode.get('trajectory')
        if trajectory is None or len(trajectory) == 0:
            return None

        # Use provided plane normal or default to direction_virtual
        if plane_normal is None:
            plane_normal = direction_virtual

        # Define plane at depth along virtual trajectory
        # depth_mm=0 is at target, positive goes towards entry
        plane_point = target_virtual + direction_virtual * depth_mm

        # Find intersection of electrode trajectory with this plane
        intersection = self._find_trajectory_plane_intersection(
            trajectory, plane_point, plane_normal
        )

        if intersection is None:
            return None

        # Project to plane coordinates
        lat_coord, ant_coord = self._project_to_plane_coords(
            intersection, plane_point, anterior_virtual, lateral_virtual
        )

        # Calculate polar coordinates
        # arctan2(ant, lat) gives angle where ant points up (0°) and lat points right (90°)
        # Then negate to get clockwise rotation
        r = float(np.sqrt(lat_coord**2 + ant_coord**2))
        theta = float(-np.degrees(np.arctan2(lat_coord, ant_coord)))

        return {
            'r': r,
            'theta': theta,
            'lat_coord': lat_coord,
            'ant_coord': ant_coord,
            'x': float(intersection[0]),
            'y': float(intersection[1]),
            'z': float(intersection[2]),
            'intersection': intersection
        }

    def _find_trajectory_plane_intersection(
        self,
        trajectory: np.ndarray,
        plane_point: np.ndarray,
        plane_normal: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find where trajectory intersects a plane."""
        if len(trajectory) < 2:
            return None

        # Find the two trajectory points that bracket the plane
        distances = np.dot(trajectory - plane_point, plane_normal)

        # Check if trajectory crosses the plane
        sign_changes = np.where(np.diff(np.sign(distances)))[0]

        if len(sign_changes) == 0:
            # No intersection, return closest point
            closest_idx = np.argmin(np.abs(distances))
            return trajectory[closest_idx]

        # Use first intersection
        idx = sign_changes[0]
        p1 = trajectory[idx]
        p2 = trajectory[idx + 1]
        d1 = distances[idx]
        d2 = distances[idx + 1]

        # Linear interpolation
        t = -d1 / (d2 - d1)
        intersection = p1 + t * (p2 - p1)

        return intersection

    def _project_to_plane_coords(
        self,
        point: np.ndarray,
        plane_point: np.ndarray,
        anterior: np.ndarray,
        lateral: np.ndarray
    ) -> Tuple[float, float]:
        """Project 3D point onto 2D plane coordinates."""
        vec = point - plane_point
        lat_coord = float(np.dot(vec, lateral))
        ant_coord = float(np.dot(vec, anterior))
        return lat_coord, ant_coord

    def _get_color_recon_1(self, idx: int) -> str:
        """Get color for electrode in reconstruction 1 (red shades)."""
        return self.colors_recon_1[idx % len(self.colors_recon_1)]

    def _get_color_recon_2(self, idx: int) -> str:
        """Get color for electrode in reconstruction 2 (blue shades)."""
        return self.colors_recon_2[idx % len(self.colors_recon_2)]

    @staticmethod
    def _get_color(idx: int) -> str:
        """Get color from the 3D plot color palette (for compatibility)."""
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

    def has_data(self) -> bool:
        """Check if there is brain shift data to analyze."""
        return len(self.matched_pairs) > 0

    def generate_contact_view_images(self, electrode_1: Dict, electrode_2: Dict,
                                     color_1: str, color_2: str,
                                     grid_spacing: float = 2.0) -> tuple[Optional[str], Optional[str]]:
        """
        Generate sagittal and coronal view images of the contact region showing both reconstructions.

        Args:
            electrode_1: First electrode reconstruction
            electrode_2: Second electrode reconstruction
            color_1: Color for first reconstruction
            color_2: Color for second reconstruction
            grid_spacing: Fixed grid spacing in mm for consistent grid across all electrodes

        Returns:
            Tuple of (sagittal_image, coronal_image) as base64-encoded PNG strings or None
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        from io import BytesIO
        import base64

        contacts_1 = electrode_1.get('contacts')
        contacts_2 = electrode_2.get('contacts')

        if contacts_1 is None or contacts_2 is None:
            return None, None

        # Get trajectory segments near contacts
        traj_1 = electrode_1.get('trajectory')
        traj_2 = electrode_2.get('trajectory')

        if traj_1 is None or traj_2 is None:
            return None, None

        # Find bounding box of all contacts with padding
        all_contacts = np.vstack([contacts_1, contacts_2])
        x_min, y_min, z_min = all_contacts.min(axis=0) - 5  # 5mm padding
        x_max, y_max, z_max = all_contacts.max(axis=0) + 5

        # Filter trajectory points within the contact region
        def filter_near_contacts(traj, x_min, x_max, y_min, y_max, z_min, z_max):
            mask = (
                (traj[:, 0] >= x_min) & (traj[:, 0] <= x_max) &
                (traj[:, 1] >= y_min) & (traj[:, 1] <= y_max) &
                (traj[:, 2] >= z_min) & (traj[:, 2] <= z_max)
            )
            return traj[mask]

        traj_1_filtered = filter_near_contacts(traj_1, x_min, x_max, y_min, y_max, z_min, z_max)
        traj_2_filtered = filter_near_contacts(traj_2, x_min, x_max, y_min, y_max, z_min, z_max)

        # Create sagittal view (Y-Z plane, looking from the side)
        fig_sag, ax_sag = plt.subplots(figsize=(2.5, 3.5), dpi=100)

        # Plot trajectories
        if len(traj_1_filtered) > 0:
            ax_sag.plot(traj_1_filtered[:, 1], traj_1_filtered[:, 2],
                       color=color_1, linewidth=2, solid_capstyle='round')
        if len(traj_2_filtered) > 0:
            ax_sag.plot(traj_2_filtered[:, 1], traj_2_filtered[:, 2],
                       color=color_2, linewidth=2, linestyle='--', solid_capstyle='round')

        # Plot contacts
        ax_sag.scatter(contacts_1[:, 1], contacts_1[:, 2],
                      color=color_1, s=80, zorder=5, edgecolors='white', linewidths=1)
        ax_sag.scatter(contacts_2[:, 1], contacts_2[:, 2],
                      color=color_2, s=80, zorder=5, edgecolors='white', linewidths=1, marker='s')

        # Remove spines and enable grid
        ax_sag.spines['top'].set_visible(False)
        ax_sag.spines['right'].set_visible(False)
        ax_sag.spines['left'].set_visible(False)
        ax_sag.spines['bottom'].set_visible(False)

        # Set fixed grid spacing
        from matplotlib.ticker import MultipleLocator
        ax_sag.xaxis.set_major_locator(MultipleLocator(grid_spacing))
        ax_sag.yaxis.set_major_locator(MultipleLocator(grid_spacing))

        # Enable grid with fixed spacing, then hide tick labels
        ax_sag.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.5)
        ax_sag.tick_params(axis='both', which='both', length=0, labelsize=0)
        ax_sag.set_aspect('equal', adjustable='box')

        # Convert sagittal to base64 PNG
        buffer_sag = BytesIO()
        plt.savefig(buffer_sag, format='png', bbox_inches='tight', dpi=100, pad_inches=0.1)
        plt.close(fig_sag)
        buffer_sag.seek(0)
        sagittal_base64 = base64.b64encode(buffer_sag.read()).decode('utf-8')

        # Create coronal view (X-Z plane, looking from the front)
        fig_cor, ax_cor = plt.subplots(figsize=(2.5, 3.5), dpi=100)

        # Plot trajectories
        if len(traj_1_filtered) > 0:
            ax_cor.plot(traj_1_filtered[:, 0], traj_1_filtered[:, 2],
                       color=color_1, linewidth=2, solid_capstyle='round')
        if len(traj_2_filtered) > 0:
            ax_cor.plot(traj_2_filtered[:, 0], traj_2_filtered[:, 2],
                       color=color_2, linewidth=2, linestyle='--', solid_capstyle='round')

        # Plot contacts
        ax_cor.scatter(contacts_1[:, 0], contacts_1[:, 2],
                      color=color_1, s=80, zorder=5, edgecolors='white', linewidths=1)
        ax_cor.scatter(contacts_2[:, 0], contacts_2[:, 2],
                      color=color_2, s=80, zorder=5, edgecolors='white', linewidths=1, marker='s')

        # Remove spines and enable grid
        ax_cor.spines['top'].set_visible(False)
        ax_cor.spines['right'].set_visible(False)
        ax_cor.spines['left'].set_visible(False)
        ax_cor.spines['bottom'].set_visible(False)

        # Set fixed grid spacing
        ax_cor.xaxis.set_major_locator(MultipleLocator(grid_spacing))
        ax_cor.yaxis.set_major_locator(MultipleLocator(grid_spacing))

        # Enable grid with fixed spacing, then hide tick labels
        ax_cor.grid(True, which='major', alpha=0.4, linestyle='--', linewidth=0.5)
        ax_cor.tick_params(axis='both', which='both', length=0, labelsize=0)
        ax_cor.set_aspect('equal', adjustable='box')

        # Convert coronal to base64 PNG
        buffer_cor = BytesIO()
        plt.savefig(buffer_cor, format='png', bbox_inches='tight', dpi=100, pad_inches=0.1)
        plt.close(fig_cor)
        buffer_cor.seek(0)
        coronal_base64 = base64.b64encode(buffer_cor.read()).decode('utf-8')

        return (f'data:image/png;base64,{sagittal_base64}',
                f'data:image/png;base64,{coronal_base64}')

    def generate_3d_figure(self) -> Optional[go.Figure]:
        """
        Generate 3D visualization showing both electrode reconstructions.

        Returns:
            Plotly Figure with both reconstructions, or None if no data
        """
        if not self.has_data():
            return None

        fig = go.Figure()

        # Add virtual trajectories (average of both reconstructions) in grey
        for i, (electrode_1, electrode_2) in enumerate(self.matched_pairs):
            result = self._calculate_virtual_trajectory(electrode_1, electrode_2)
            if result[0] is not None:
                entry_virtual, target_virtual = result[0], result[1]
                # Create virtual trajectory line from target to entry
                fig.add_trace(go.Scatter3d(
                    x=[target_virtual[0], entry_virtual[0]],
                    y=[target_virtual[1], entry_virtual[1]],
                    z=[target_virtual[2], entry_virtual[2]],
                    mode='lines',
                    line=dict(color='#888888', width=2, dash='dot'),
                    name=f'E{i+1} Virtual',
                    legendgroup=f'electrode_{i}_virtual',
                    showlegend=True
                ))

        # Add first reconstruction (solid lines, red shades)
        for i, (electrode_1, electrode_2) in enumerate(self.matched_pairs):
            label = electrode_1.get('label', f'E{i+1}')
            color_1 = self._get_color_recon_1(i)
            color_2 = self._get_color_recon_2(i)

            # Electrode 1 trajectory
            traj_1 = electrode_1.get('trajectory')
            if traj_1 is not None and len(traj_1) > 0:
                fig.add_trace(go.Scatter3d(
                    x=traj_1[:, 0].tolist(),
                    y=traj_1[:, 1].tolist(),
                    z=traj_1[:, 2].tolist(),
                    mode='lines',
                    line=dict(color=color_1, width=4),
                    name=f'{label} Recon 1',
                    legendgroup=f'electrode_{i}_recon1',
                    showlegend=True
                ))

            # Electrode 1 contacts
            contacts_1 = electrode_1.get('contacts')
            if contacts_1 is not None and len(contacts_1) > 0:
                fig.add_trace(go.Scatter3d(
                    x=contacts_1[:, 0].tolist(),
                    y=contacts_1[:, 1].tolist(),
                    z=contacts_1[:, 2].tolist(),
                    mode='markers',
                    marker=dict(size=6, color=color_1, symbol='circle'),
                    name=f'{label} Contacts 1',
                    legendgroup=f'electrode_{i}_recon1',
                    showlegend=False
                ))

            # Electrode 2 trajectory (dashed, blue shades)
            traj_2 = electrode_2.get('trajectory')
            if traj_2 is not None and len(traj_2) > 0:
                fig.add_trace(go.Scatter3d(
                    x=traj_2[:, 0].tolist(),
                    y=traj_2[:, 1].tolist(),
                    z=traj_2[:, 2].tolist(),
                    mode='lines',
                    line=dict(color=color_2, width=4, dash='dash'),
                    name=f'{label} Recon 2',
                    legendgroup=f'electrode_{i}_recon2',
                    showlegend=True
                ))

            # Electrode 2 contacts
            contacts_2 = electrode_2.get('contacts')
            if contacts_2 is not None and len(contacts_2) > 0:
                fig.add_trace(go.Scatter3d(
                    x=contacts_2[:, 0].tolist(),
                    y=contacts_2[:, 1].tolist(),
                    z=contacts_2[:, 2].tolist(),
                    mode='markers',
                    marker=dict(size=6, color=color_2, symbol='diamond'),
                    name=f'{label} Contacts 2',
                    legendgroup=f'electrode_{i}_recon2',
                    showlegend=False
                ))

        # Update layout to match stereotactic report style
        fig.update_layout(
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
            scene=dict(
                bgcolor='white',
                xaxis=dict(
                    title='X (mm)',
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True
                ),
                yaxis=dict(
                    title='Y (mm)',
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True
                ),
                zaxis=dict(
                    title='Z (mm)',
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True
                ),
                camera=dict(
                    projection=dict(type='orthographic'),
                    eye=dict(x=0, y=2.5, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            font=dict(color='#333'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd',
                font=dict(color='#333')
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def generate_html_section(self) -> str:
        """
        Generate HTML for the brain shift tab.

        Returns:
            HTML string for brain shift analysis
        """
        if not self.has_data():
            return """
            <div class="section full-width">
                <p style="text-align: center; color: #999;">No brain shift data available. Load two electrode reconstructions to enable this analysis.</p>
            </div>
            """

        html_parts = []

        # Add metadata card
        html_parts.extend([
            '        <div class="section full-width">',
            '            <h2>Reconstruction Metadata</h2>',
            '            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">',
            '                <div>',
            f'                    <h3 style="color: {self.colors_recon_1[0]};">Reconstruction 1 <span style="background-color: #ef5350; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; font-weight: 600; margin-left: 8px;">PRIMARY</span></h3>',
            '                    <table style="width: 100%; border-collapse: collapse;">',
        ])

        # Add metadata for reconstruction 1
        meta1 = self.metadata_1
        if meta1:
            ct_file_1 = meta1.get('ct_file', 'N/A')
            timestamp_1 = meta1.get('timestamp', 'N/A')
            pypacer_version_1 = meta1.get('pypacer_version', 'N/A')
            voxel_sizes_1 = meta1.get('voxel_sizes_mm', [])
            ct_shape_1 = meta1.get('ct_volume_shape', [])

            html_parts.extend([
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>CT File:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{ct_file_1.split("/")[-1] if ct_file_1 != "N/A" else "N/A"}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Timestamp:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{timestamp_1}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>PyPaCER Version:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{pypacer_version_1}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Voxel Size (mm):</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{" × ".join(map(str, voxel_sizes_1)) if voxel_sizes_1 else "N/A"}</td></tr>',
                '                        <tr><td style="padding: 5px;"><strong>CT Volume Shape:</strong></td>',
                f'                            <td style="padding: 5px;">{" × ".join(map(str, ct_shape_1)) if ct_shape_1 else "N/A"}</td></tr>',
            ])
        else:
            html_parts.append('                        <tr><td colspan="2" style="padding: 10px; text-align: center; color: #999;">No metadata available</td></tr>')

        html_parts.extend([
            '                    </table>',
            '                </div>',
            '                <div>',
            f'                    <h3 style="color: {self.colors_recon_2[0]};">Reconstruction 2</h3>',
            '                    <table style="width: 100%; border-collapse: collapse;">',
        ])

        # Add metadata for reconstruction 2
        meta2 = self.metadata_2
        if meta2:
            ct_file_2 = meta2.get('ct_file', 'N/A')
            timestamp_2 = meta2.get('timestamp', 'N/A')
            pypacer_version_2 = meta2.get('pypacer_version', 'N/A')
            voxel_sizes_2 = meta2.get('voxel_sizes_mm', [])
            ct_shape_2 = meta2.get('ct_volume_shape', [])

            html_parts.extend([
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>CT File:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{ct_file_2.split("/")[-1] if ct_file_2 != "N/A" else "N/A"}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Timestamp:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{timestamp_2}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>PyPaCER Version:</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{pypacer_version_2}</td></tr>',
                '                        <tr><td style="padding: 5px; border-bottom: 1px solid #eee;"><strong>Voxel Size (mm):</strong></td>',
                f'                            <td style="padding: 5px; border-bottom: 1px solid #eee;">{" × ".join(map(str, voxel_sizes_2)) if voxel_sizes_2 else "N/A"}</td></tr>',
                '                        <tr><td style="padding: 5px;"><strong>CT Volume Shape:</strong></td>',
                f'                            <td style="padding: 5px;">{" × ".join(map(str, ct_shape_2)) if ct_shape_2 else "N/A"}</td></tr>',
            ])
        else:
            html_parts.append('                        <tr><td colspan="2" style="padding: 10px; text-align: center; color: #999;">No metadata available</td></tr>')

        html_parts.extend([
            '                    </table>',
            '                </div>',
            '            </div>',
            '        </div>',
        ])

        # Calculate consistent grid spacing for all electrodes
        # Use a fixed grid spacing of 2mm for consistency
        grid_spacing = 2.0

        # Add electrode cards
        html_parts.append('        <div class="electrodes-container">')

        for pair_idx, (electrode_1, electrode_2) in enumerate(self.matched_pairs):
            label = electrode_1.get('label', f'Electrode {pair_idx + 1}')
            color_1 = self._get_color_recon_1(pair_idx)
            color_2 = self._get_color_recon_2(pair_idx)

            # Calculate virtual trajectory (returns both coordinate systems)
            result = self._calculate_virtual_trajectory(electrode_1, electrode_2)

            if result[0] is None:
                continue

            (entry_virtual, target_virtual, direction_virtual,
             anterior_perp, lateral_perp,
             anterior_axial, lateral_axial, plane_warning) = result

            # Determine medial direction based on X coordinate relative to center
            # Calculate the center of X-range from all electrode data
            all_x_coords = []
            for traj in [electrode_1.get('trajectory'), electrode_2.get('trajectory')]:
                if traj is not None and len(traj) > 0:
                    all_x_coords.extend(traj[:, 0].tolist())

            x_center = np.mean(all_x_coords) if len(all_x_coords) > 0 else 0.0

            # Determine if positive X is medial (toward center) or lateral (away from center)
            # If target is to the left of center (target_x < x_center), then +X is medial
            # If target is to the right of center (target_x > x_center), then -X is medial
            target_x = target_virtual[0]
            is_positive_x_medial = target_x < x_center

            # Generate sagittal and coronal view images of contact region with consistent grid spacing
            sagittal_image, coronal_image = self.generate_contact_view_images(
                electrode_1, electrode_2, color_1, color_2, grid_spacing=grid_spacing
            )

            # Calculate contact displacements
            displacements = self._calculate_contact_displacements(electrode_1, electrode_2)

            # Calculate trajectory lengths
            traj_1 = electrode_1.get('trajectory')
            traj_2 = electrode_2.get('trajectory')

            # Calculate curved trajectory length (sum of segment lengths)
            curved_length_1 = 0.0
            euclidean_length_1 = 0.0
            if traj_1 is not None and len(traj_1) > 1:
                # Curved length: sum of all segments
                for i in range(len(traj_1) - 1):
                    curved_length_1 += float(np.linalg.norm(traj_1[i+1] - traj_1[i]))
                # Euclidean length: direct distance from entry to tip
                euclidean_length_1 = float(np.linalg.norm(traj_1[-1] - traj_1[0]))

            curved_length_2 = 0.0
            euclidean_length_2 = 0.0
            if traj_2 is not None and len(traj_2) > 1:
                # Curved length: sum of all segments
                for i in range(len(traj_2) - 1):
                    curved_length_2 += float(np.linalg.norm(traj_2[i+1] - traj_2[i]))
                # Euclidean length: direct distance from entry to tip
                euclidean_length_2 = float(np.linalg.norm(traj_2[-1] - traj_2[0]))

            # Calculate depth range from both trajectories
            # We want depth=0 at entry (where both have data) and positive going towards tip
            depths_1 = []
            depths_2 = []

            if traj_1 is not None:
                for point in traj_1:
                    diff = point - target_virtual
                    depth = float(np.dot(diff, direction_virtual))
                    depths_1.append(depth)

            if traj_2 is not None:
                for point in traj_2:
                    diff = point - target_virtual
                    depth = float(np.dot(diff, direction_virtual))
                    depths_2.append(depth)

            # Find the calibration point: the maximum depth where both have data (closest to entry)
            # This becomes our new depth=0
            if len(depths_1) > 0 and len(depths_2) > 0:
                calibration_depth = min(max(depths_1), max(depths_2))  # Closest to entry where both exist
                # Find the deepest tip (minimum depth, furthest from entry)
                deepest_tip = min(min(depths_1), min(depths_2))
            else:
                calibration_depth = 80.0
                deepest_tip = 0.0

            # New depth range: 0 (entry/calibration) to positive (towards deepest tip)
            # Map so calibration_depth -> 0 and deepest_tip -> positive value
            depth_min = 0.0
            depth_max = calibration_depth - deepest_tip  # Total distance from calibration to deepest tip
            depth_step = 0.5

            # Calculate default depth (average of C3 contact positions if available)
            default_depth = 0.0
            if len(displacements) > 0 and len(displacements) >= 4:
                # Use C3 contact (index 3)
                contacts_1 = electrode_1.get('contacts')
                if contacts_1 is not None and len(contacts_1) > 3:
                    c3_pos = contacts_1[3]
                    diff = c3_pos - target_virtual
                    original_depth = float(np.dot(diff, direction_virtual))
                    # Recalibrate to new zero point
                    default_depth = calibration_depth - original_depth

            # Start electrode card
            html_parts.extend([
                '            <div class="section">',
                f'                <h2>{label} - Brain Shift Analysis</h2>',
                f'                <div style="margin-bottom: 10px; font-size: 0.9em; color: #666;">',
                f'                    <strong>Curved Length R1:</strong> {curved_length_1:.2f}mm | ',
                f'                    <strong>R2:</strong> {curved_length_2:.2f}mm | ',
                f'                    <strong>Diff:</strong> {abs(curved_length_1 - curved_length_2):.2f}mm<br>',
                f'                    <strong>Euclidean Length R1:</strong> {euclidean_length_1:.2f}mm | ',
                f'                    <strong>R2:</strong> {euclidean_length_2:.2f}mm | ',
                f'                    <strong>Diff:</strong> {abs(euclidean_length_1 - euclidean_length_2):.2f}mm',
                f'                </div>',
                '                <div class="target-content">',
                '                    <div class="left-column">',
            ])

            # Add sagittal and coronal view images with toggle buttons
            if sagittal_image and coronal_image:
                html_parts.extend([
                    '                        <div style="margin-bottom: 15px;">',
                    '                            <div style="display: flex; gap: 10px; margin-bottom: 10px;">',
                    f'                                <button id="toggle_sag_{pair_idx}" class="coord-toggle active" onclick="switchContactView({pair_idx}, \'sagittal\')">Sagittal</button>',
                    f'                                <button id="toggle_cor_{pair_idx}" class="coord-toggle" onclick="switchContactView({pair_idx}, \'coronal\')">Coronal</button>',
                    '                            </div>',
                    f'                            <img id="contact_view_img_{pair_idx}_sagittal" src="{sagittal_image}" style="width: 100%; max-height: 350px; object-fit: contain; border-radius: 8px; display: block;" alt="Sagittal View">',
                    f'                            <img id="contact_view_img_{pair_idx}_coronal" src="{coronal_image}" style="width: 100%; max-height: 350px; object-fit: contain; border-radius: 8px; display: none;" alt="Coronal View">',
                    '                        </div>',
                ])

            html_parts.extend([
                '                        <div class="contact-buttons">',
            ])

            # Add contact buttons styled like stereotactic report
            for disp in reversed(displacements):  # Reverse to show C3, C2, C1, C0
                contact_idx = disp['contact_index']
                contact_label = disp['contact_label']
                distance = disp['euclidean_distance']
                dx = disp['dx']
                dy = disp['dy']
                dz = disp['dz']

                # Calculate depth for this contact (recalibrated to new zero point)
                pos_1 = disp['position_1']
                diff = pos_1 - target_virtual
                original_depth = float(np.dot(diff, direction_virtual))
                depth = calibration_depth - original_depth  # Recalibrate

                # Determine direction labels
                ap_dir = 'Ant' if dx >= 0 else 'Post'
                ml_dir = 'Med' if dy >= 0 else 'Lat'

                tooltip_info = f'ΔX: {dx:+.2f}mm | ΔY: {dy:+.2f}mm | ΔZ: {dz:+.2f}mm'
                active_class = ' active' if contact_idx == 3 else ''

                html_parts.append(
                    f'                            <div class="contact-button{active_class}" onclick="setDepthForBrainShift({pair_idx}, {depth:.2f})">'
                    f'<span class="label">{contact_label}</span>'
                    f'<span class="depth">{depth:+.2f}mm</span>'
                    f'<span class="distance">{distance:.2f}mm</span>'
                    f'<span class="tooltip-info">{tooltip_info}</span>'
                    f'</div>'
                )

            html_parts.extend([
                '                        </div>',
                '                    </div>',
                '                    <div class="right-column">',
            ])

            # Calculate position data for all depths using BOTH coordinate systems
            depths_range = np.arange(depth_min, depth_max + depth_step, depth_step)
            position_data_perp = {}  # Perpendicular plane coordinates
            position_data_axial = {}  # Anatomical plane coordinates
            max_r_perp = 0.0
            max_r_axial = 0.0

            for depth in depths_range:
                # Convert recalibrated depth back to original depth relative to target_virtual
                original_depth = calibration_depth - depth

                # Calculate positions using perpendicular plane coordinates
                # Plane normal is direction_virtual (perpendicular to trajectory)
                pos_1_perp = self._calculate_position_at_depth(
                    electrode_1, entry_virtual, target_virtual, direction_virtual,
                    anterior_perp, lateral_perp, original_depth,
                    plane_normal=direction_virtual
                )
                pos_2_perp = self._calculate_position_at_depth(
                    electrode_2, entry_virtual, target_virtual, direction_virtual,
                    anterior_perp, lateral_perp, original_depth,
                    plane_normal=direction_virtual
                )

                # Calculate positions using axial coordinates
                # Plane normal is [0,0,1] (axial plane perpendicular to Z-axis)
                axial_plane_normal = np.array([0.0, 0.0, 1.0])
                pos_1_axial = self._calculate_position_at_depth(
                    electrode_1, entry_virtual, target_virtual, direction_virtual,
                    anterior_axial, lateral_axial, original_depth,
                    plane_normal=axial_plane_normal
                )
                pos_2_axial = self._calculate_position_at_depth(
                    electrode_2, entry_virtual, target_virtual, direction_virtual,
                    anterior_axial, lateral_axial, original_depth,
                    plane_normal=axial_plane_normal
                )

                # Store data for perpendicular plane if any position exists
                if pos_1_perp or pos_2_perp:
                    if pos_1_perp:
                        max_r_perp = max(max_r_perp, pos_1_perp['r'])
                    if pos_2_perp:
                        max_r_perp = max(max_r_perp, pos_2_perp['r'])

                    displacement_perp = None
                    euclidean_dist_perp = None
                    if pos_1_perp and pos_2_perp:
                        displacement_perp = np.array([
                            pos_2_perp['x'] - pos_1_perp['x'],
                            pos_2_perp['y'] - pos_1_perp['y'],
                            pos_2_perp['z'] - pos_1_perp['z']
                        ])
                        euclidean_dist_perp = float(np.linalg.norm(displacement_perp))

                    position_data_perp[f'{depth:.2f}'] = {
                        'pos_1': {
                            'theta': float(pos_1_perp['theta']),
                            'r': float(pos_1_perp['r']),
                            'ant': float(pos_1_perp['ant_coord']),
                            'lat': float(pos_1_perp['lat_coord']),
                            'x': float(pos_1_perp['x']),
                            'y': float(pos_1_perp['y']),
                            'z': float(pos_1_perp['z'])
                        } if pos_1_perp else None,
                        'pos_2': {
                            'theta': float(pos_2_perp['theta']),
                            'r': float(pos_2_perp['r']),
                            'ant': float(pos_2_perp['ant_coord']),
                            'lat': float(pos_2_perp['lat_coord']),
                            'x': float(pos_2_perp['x']),
                            'y': float(pos_2_perp['y']),
                            'z': float(pos_2_perp['z'])
                        } if pos_2_perp else None,
                        'displacement': {
                            'dx': float(displacement_perp[0]) if displacement_perp is not None else None,
                            'dy': float(displacement_perp[1]) if displacement_perp is not None else None,
                            'dz': float(displacement_perp[2]) if displacement_perp is not None else None,
                            'euclidean': euclidean_dist_perp
                        } if displacement_perp is not None else None
                    }

                # Store data for axial plane if any position exists
                if pos_1_axial or pos_2_axial:
                    if pos_1_axial:
                        max_r_axial = max(max_r_axial, pos_1_axial['r'])
                    if pos_2_axial:
                        max_r_axial = max(max_r_axial, pos_2_axial['r'])

                    displacement_axial = None
                    euclidean_dist_axial = None
                    if pos_1_axial and pos_2_axial:
                        displacement_axial = np.array([
                            pos_2_axial['x'] - pos_1_axial['x'],
                            pos_2_axial['y'] - pos_1_axial['y'],
                            pos_2_axial['z'] - pos_1_axial['z']
                        ])
                        euclidean_dist_axial = float(np.linalg.norm(displacement_axial))

                    position_data_axial[f'{depth:.2f}'] = {
                        'pos_1': {
                            'theta': float(pos_1_axial['theta']),
                            'r': float(pos_1_axial['r']),
                            'ant': float(pos_1_axial['ant_coord']),
                            'lat': float(pos_1_axial['lat_coord']),
                            'x': float(pos_1_axial['x']),
                            'y': float(pos_1_axial['y']),
                            'z': float(pos_1_axial['z'])
                        } if pos_1_axial else None,
                        'pos_2': {
                            'theta': float(pos_2_axial['theta']),
                            'r': float(pos_2_axial['r']),
                            'ant': float(pos_2_axial['ant_coord']),
                            'lat': float(pos_2_axial['lat_coord']),
                            'x': float(pos_2_axial['x']),
                            'y': float(pos_2_axial['y']),
                            'z': float(pos_2_axial['z'])
                        } if pos_1_axial else None,
                        'displacement': {
                            'dx': float(displacement_axial[0]) if displacement_axial is not None else None,
                            'dy': float(displacement_axial[1]) if displacement_axial is not None else None,
                            'dz': float(displacement_axial[2]) if displacement_axial is not None else None,
                            'euclidean': euclidean_dist_axial
                        } if displacement_axial is not None else None
                    }

            # Calculate fixed range for polar chart based on max deviation + 0.5mm padding
            # Use the maximum of both coordinate systems for consistent scaling
            max_r_combined = max(max_r_perp, max_r_axial)
            range_max = float(np.ceil((max_r_combined + 0.5) * 2) / 2)  # Round up to nearest 0.5mm
            dtick = 0.5 if range_max <= 2 else 1.0  # Use 0.5mm ticks if range is small

            # Add polar chart placeholder with coordinate system toggle
            div_id = f'polar_shift_{pair_idx}'
            slider_id = f'depth_slider_shift_{pair_idx}'

            # Add warning message if needed
            warning_html = ''
            if plane_warning:
                warning_html = f'<div style="color: #ff9800; font-size: 0.85em; margin-bottom: 5px;">⚠ {plane_warning}</div>'

            html_parts.extend([
                f'                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">',
                f'                            <div style="display: flex; gap: 10px;">',
                f'                                <button id="toggle_perp_{pair_idx}" class="coord-toggle active" onclick="switchBrainShiftCoords({pair_idx}, \'perp\')">Perpendicular Plane</button>',
                f'                                <button id="toggle_axial_{pair_idx}" class="coord-toggle" onclick="switchBrainShiftCoords({pair_idx}, \'axial\')">Axial Plane</button>',
                f'                            </div>',
                f'                        </div>',
                warning_html,
                f'                        <div class="polar-chart">',
                f'                            <div id="{div_id}"></div>',
                '                        </div>',
                '                        <div class="depth-slider-container">',
                f'                            <label for="{slider_id}">Depth: <span id="{slider_id}_value" class="depth-value">{default_depth:+.2f} mm</span></label>',
                f'                            <input type="range" id="{slider_id}" class="depth-slider" ',
                f'                                   min="{depth_min}" max="{depth_max}" step="{depth_step}" value="{default_depth:.2f}" ',
                f'                                   oninput="onBrainShiftSliderInput({pair_idx}, this.value)">',
                '                        </div>',
                f'                        <div class="position-stats" id="stats_shift_{pair_idx}">',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">Displacement:</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_distance">--</span>',
                '                            </div>',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">ΔX:</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_dx">--</span>',
                '                            </div>',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">ΔY:</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_dy">--</span>',
                '                            </div>',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">ΔZ:</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_dz">--</span>',
                '                            </div>',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">R1 (X,Y,Z):</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_xyz1">--</span>',
                '                            </div>',
                '                            <div class="stat-row">',
                '                                <span class="stat-label">R2 (X,Y,Z):</span>',
                f'                                <span class="stat-value" id="stats_shift_{pair_idx}_xyz2">--</span>',
                '                            </div>',
                '                        </div>',
                '                    </div>',
                '                </div>',
            ])

            # Add JavaScript data for this electrode pair (both coordinate systems)
            html_parts.extend([
                '                <script>',
                f'                    // Store position data for brain shift pair {pair_idx} (both coordinate systems)',
                f'                    window.brainShiftDataPerpPair{pair_idx} = {json.dumps(position_data_perp)};',
                f'                    window.brainShiftDataAxialPair{pair_idx} = {json.dumps(position_data_axial)};',
                f'                    window.brainShiftActiveCoordPair{pair_idx} = "perp";  // Default to perpendicular plane',
                f'                    window.brainShiftColorRecon1Pair{pair_idx} = "{color_1}";',
                f'                    window.brainShiftColorRecon2Pair{pair_idx} = "{color_2}";',
                f'                    window.brainShiftRangeMaxPair{pair_idx} = {range_max};',
                f'                    window.brainShiftDtickPair{pair_idx} = {dtick};',
                f'                    window.brainShiftIsPositiveXMedialPair{pair_idx} = {"true" if is_positive_x_medial else "false"};',
                f'                    // Store initialization params',
                f'                    if (!window.brainShiftChartsToInit) window.brainShiftChartsToInit = [];',
                f'                    window.brainShiftChartsToInit.push({{pairIdx: {pair_idx}, divId: "{div_id}", depth: {default_depth:.2f}}});',
                '                </script>',
            ])

            html_parts.append('            </div>')  # Close section card

        html_parts.append('        </div>')  # Close electrodes-container

        # Add 3D visualization
        fig_3d = self.generate_3d_figure()
        if fig_3d:
            html_parts.extend([
                '        <div class="section full-width">',
                '            <h2>3D Visualization</h2>',
                '            <div id="plot3d_brainshift" style="width: 100%; min-height: 600px;"></div>',
                '            <script>',
                f'                var dataBrainShift = {fig_3d.to_json()};',
                '                Plotly.newPlot("plot3d_brainshift", dataBrainShift.data, dataBrainShift.layout, {responsive: true, displayModeBar: true});',
                '            </script>',
                '        </div>',
            ])

        return '\n'.join(html_parts)
