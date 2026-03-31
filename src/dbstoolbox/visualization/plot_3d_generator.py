"""Standalone 3D plot generator for surgical and electrode data."""

from typing import List, Dict, Optional
import numpy as np
import plotly.graph_objects as go


class Plot3DGenerator:
    """Generate 3D Plotly figures from surgical and electrode data."""

    def __init__(
        self,
        surgical_targets: Optional[List[Dict]] = None,
        electrode_trajectories: Optional[List[Dict]] = None
    ):
        """
        Initialize the 3D plot generator.

        Args:
            surgical_targets: List of parsed surgical target dictionaries
            electrode_trajectories: List of parsed electrode trajectory dictionaries
        """
        self.surgical_targets = surgical_targets or []
        self.electrode_trajectories = electrode_trajectories or []

    def generate_figure(
        self,
        show_mer_tracks: bool = True,
        show_trajectories: bool = True,
        show_targets: bool = True,
        show_contacts: bool = True,
        dark_mode: bool = False
    ) -> go.Figure:
        """
        Generate a 3D Plotly figure with surgical and electrode data.

        Args:
            show_mer_tracks: Show MER tracks from surgical planning
            show_trajectories: Show electrode/surgical trajectories
            show_targets: Show target points
            show_contacts: Show electrode contacts
            dark_mode: Use dark background

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        color_offset = 0

        # Add surgical traces
        if self.surgical_targets:
            color_offset = self._add_surgical_traces(
                fig, self.surgical_targets, color_offset,
                show_mer_tracks, show_trajectories, show_targets
            )

        # Add electrode traces
        if self.electrode_trajectories:
            color_offset = self._add_electrode_traces(
                fig, self.electrode_trajectories, color_offset,
                show_trajectories, show_contacts
            )

        # Calculate isotropic axis ranges
        axis_ranges = self._calculate_isotropic_ranges(fig)

        # Apply layout styling
        if dark_mode:
            self._apply_dark_mode_layout(fig, axis_ranges)
        else:
            self._apply_light_mode_layout(fig, axis_ranges)

        return fig

    def _add_surgical_traces(
        self,
        fig: go.Figure,
        surgical_targets: List[Dict],
        color_offset: int,
        show_mer_tracks: bool,
        show_trajectory: bool,
        show_targets: bool
    ) -> int:
        """Add surgical targeting traces to the figure."""
        MER_OFFSET_MM = 2.0

        for idx, target_data in enumerate(surgical_targets):
            # Generate color for this target
            color_idx = (idx + color_offset) % 10
            color = self._get_color(color_idx)

            target = target_data['target']
            entry = target_data.get('entry')
            direction = target_data['direction']
            anterior = target_data.get('anterior')
            lateral = target_data.get('lateral')

            # Get label info
            patient_id = target_data.get('patient_id', '')
            hemisphere = target_data.get('hemisphere', '')
            anat_target = target_data.get('anatomical_target', '')
            label_base = f"{patient_id} {hemisphere} {anat_target}".strip()

            # Add target point
            if show_targets:
                fig.add_trace(go.Scatter3d(
                    x=[target[0]],
                    y=[target[1]],
                    z=[target[2]],
                    mode='markers',
                    marker=dict(size=8, color=color, symbol='diamond'),
                    name=f'{label_base} Target',
                    showlegend=True,
                    hovertemplate=f'{label_base}<br>Target<br>({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})<extra></extra>'
                ))

            # Add central trajectory
            if show_trajectory and entry is not None:
                trajectory_line = np.array([target, entry])
                fig.add_trace(go.Scatter3d(
                    x=trajectory_line[:, 0].tolist(),
                    y=trajectory_line[:, 1].tolist(),
                    z=trajectory_line[:, 2].tolist(),
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'{label_base} Central',
                    showlegend=True,
                    hovertemplate=f'{label_base}<br>Central trajectory<extra></extra>'
                ))

            # Add MER tracks
            if show_mer_tracks and anterior is not None and lateral is not None:
                # MER track positions relative to central
                mer_tracks = {
                    'Anterior': anterior * MER_OFFSET_MM,
                    'Posterior': -anterior * MER_OFFSET_MM,
                    'Lateral': lateral * MER_OFFSET_MM,
                    'Medial': -lateral * MER_OFFSET_MM,
                }

                for track_name, offset in mer_tracks.items():
                    # MER tracks extend 5mm below target
                    mer_target = target + offset - direction * 5.0
                    mer_entry = entry + offset if entry is not None else target + offset + direction * 100

                    mer_line = np.array([mer_target, mer_entry])

                    fig.add_trace(go.Scatter3d(
                        x=mer_line[:, 0].tolist(),
                        y=mer_line[:, 1].tolist(),
                        z=mer_line[:, 2].tolist(),
                        mode='lines',
                        line=dict(color=color, width=2, dash='dot'),
                        name=f'{label_base} MER {track_name}',
                        showlegend=False,
                        hovertemplate=f'{label_base}<br>MER {track_name}<extra></extra>'
                    ))

        return color_offset + len(surgical_targets)

    def _add_electrode_traces(
        self,
        fig: go.Figure,
        electrode_trajectories: List[Dict],
        color_offset: int,
        show_trajectory: bool,
        show_contacts: bool
    ) -> int:
        """Add electrode traces to the figure."""
        for idx, electrode in enumerate(electrode_trajectories):
            # Generate color for this electrode
            color_idx = (idx + color_offset) % 10
            color = self._get_color(color_idx)

            label = electrode.get('label', f'E{idx+1}')
            trajectory = electrode.get('trajectory')
            contacts = electrode.get('contacts')

            # Add trajectory line
            if show_trajectory and trajectory is not None and len(trajectory) > 0:
                fig.add_trace(go.Scatter3d(
                    x=trajectory[:, 0].tolist(),
                    y=trajectory[:, 1].tolist(),
                    z=trajectory[:, 2].tolist(),
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'{label} Trajectory',
                    showlegend=True,
                    hovertemplate=f'{label}<br>Electrode trajectory<extra></extra>'
                ))

            # Add contact points
            if show_contacts and contacts is not None and len(contacts) > 0:
                contact_labels = [f'C{i}' for i in range(len(contacts))]

                fig.add_trace(go.Scatter3d(
                    x=contacts[:, 0].tolist(),
                    y=contacts[:, 1].tolist(),
                    z=contacts[:, 2].tolist(),
                    mode='markers+text',
                    marker=dict(size=6, color=color, symbol='circle'),
                    text=contact_labels,
                    textposition='top center',
                    textfont=dict(size=10),
                    name=f'{label} Contacts',
                    showlegend=True,
                    hovertemplate='%{text}<br>(%{x:.1f}, %{y:.1f}, %{z:.1f})<extra></extra>'
                ))

            # Add orientation marker vectors (directional DBS electrodes)
            orientation = electrode.get('orientation')
            if orientation and orientation.get('has_markers') and 'markers' in orientation:
                marker_colors = {'A': '#FFA500', 'B': '#1E90FF', 'C': '#32CD32'}
                vector_length_mm = 5.0

                # Build markers to render (A, B from data + computed C)
                markers_to_render = {}
                for mk, md in orientation['markers'].items():
                    p = md.get('position_xyz')
                    d = md.get('direction_vector')
                    if p is not None and d is not None:
                        markers_to_render[mk] = (np.array(p), np.array(d))

                # Compute virtual C marker: position midpoint, direction = -(A+B) normalized
                if 'A' in markers_to_render and 'B' in markers_to_render:
                    pos_a, dir_a = markers_to_render['A']
                    pos_b, dir_b = markers_to_render['B']
                    pos_c = (pos_a + pos_b) / 2.0
                    dir_c = -(dir_a + dir_b)
                    norm_c = np.linalg.norm(dir_c)
                    if norm_c > 1e-10:
                        dir_c = dir_c / norm_c
                    markers_to_render['C'] = (pos_c, dir_c)

                for marker_key, (pos, direction) in markers_to_render.items():
                    end = pos + direction * vector_length_mm
                    m_color = marker_colors.get(marker_key, color)

                    # Marker vector line
                    fig.add_trace(go.Scatter3d(
                        x=[pos[0], end[0]],
                        y=[pos[1], end[1]],
                        z=[pos[2], end[2]],
                        mode='lines',
                        line=dict(color=m_color, width=3),
                        name=f'{label} Marker {marker_key}',
                        showlegend=False,
                        hovertemplate=f'{label}<br>Marker {marker_key}<extra></extra>'
                    ))

                    # Arrowhead cone at the end of the vector
                    fig.add_trace(go.Cone(
                        x=[end[0]], y=[end[1]], z=[end[2]],
                        u=[direction[0]], v=[direction[1]], w=[direction[2]],
                        sizemode='absolute',
                        sizeref=1.0,
                        showscale=False,
                        colorscale=[[0, m_color], [1, m_color]],
                        name=f'{label} Marker {marker_key}',
                        showlegend=False,
                        hoverinfo='skip'
                    ))

                    # Floating label at the arrow tip
                    fig.add_trace(go.Scatter3d(
                        x=[end[0]],
                        y=[end[1]],
                        z=[end[2]],
                        mode='text',
                        text=[marker_key],
                        textposition='top center',
                        textfont=dict(size=12, color=m_color),
                        name=f'{label} Marker {marker_key}',
                        showlegend=False,
                        hovertemplate=f'{label} Marker {marker_key}<br>({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})<extra></extra>'
                    ))

        return color_offset + len(electrode_trajectories)

    def _calculate_isotropic_ranges(self, fig: go.Figure) -> Dict[str, List[float]]:
        """Calculate isotropic axis ranges to make the plot cubic."""
        # Collect all points from the figure
        all_x, all_y, all_z = [], [], []

        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                all_x.extend([x for x in trace.x if x is not None])
            if hasattr(trace, 'y') and trace.y is not None:
                all_y.extend([y for y in trace.y if y is not None])
            if hasattr(trace, 'z') and trace.z is not None:
                all_z.extend([z for z in trace.z if z is not None])

        if not all_x or not all_y or not all_z:
            # Default ranges if no data
            return {'x': [-100, 100], 'y': [-100, 100], 'z': [-100, 100]}

        # Calculate ranges
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)

        # Add 10% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        z_min -= z_range * 0.1
        z_max += z_range * 0.1

        # Make cubic by using the largest range
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Center each axis
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        return {
            'x': [x_center - max_range / 2, x_center + max_range / 2],
            'y': [y_center - max_range / 2, y_center + max_range / 2],
            'z': [z_center - max_range / 2, z_center + max_range / 2]
        }

    def _apply_dark_mode_layout(self, fig: go.Figure, axis_ranges: Dict[str, List[float]]):
        """Apply dark mode styling to the figure."""
        fig.update_layout(
            title='3D Visualization',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                dragmode='orbit',
                bgcolor='#1e1e1e',
                xaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#444',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['x']
                ),
                yaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#444',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['y']
                ),
                zaxis=dict(
                    backgroundcolor='#1e1e1e',
                    gridcolor='#444',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['z']
                )
            ),
            showlegend=True,
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(30,30,30,0.8)',
                bordercolor='#444',
                font=dict(color='#fff')
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#fff')
        )

    def _apply_light_mode_layout(self, fig: go.Figure, axis_ranges: Dict[str, List[float]]):
        """Apply light mode styling to the figure (for reports)."""
        fig.update_layout(
            title='3D Visualization',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                dragmode='orbit',
                bgcolor='white',
                xaxis=dict(
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['x']
                ),
                yaxis=dict(
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['y']
                ),
                zaxis=dict(
                    backgroundcolor='white',
                    gridcolor='#ddd',
                    showbackground=True,
                    zerolinecolor='#666',
                    range=axis_ranges['z']
                )
            ),
            showlegend=True,
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#ddd',
                font=dict(color='#333')
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='#333')
        )

    @staticmethod
    def _get_color(idx: int) -> str:
        """Get a color from a predefined palette."""
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
