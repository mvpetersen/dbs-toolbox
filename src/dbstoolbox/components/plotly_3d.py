"""3D visualization component using Plotly."""

from nicegui import ui
from typing import List, Tuple, Optional, Dict, Any
import plotly.graph_objects as go
import numpy as np


class Plotly3DViewer:
    """3D viewer using Plotly for electrode and brain visualization."""
    
    def __init__(self, width: str = '100%', height: str = '600px'):
        self.width = width
        self.height = height
        self.figure = None
        self.traces = {}
        self.plot_element = None
        self.uirevision = 'constant'  # This preserves UI state including camera
        
    def create(self):
        """Create the 3D viewer UI."""
        # Initialize figure with an invisible point to force 3D view
        self.figure = go.Figure()
        
        # Add an invisible point at origin to ensure 3D axes are shown
        self.figure.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=0.1, opacity=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        self._setup_layout()
        
        # Create plot element
        self.plot_element = ui.plotly(self.figure).classes(f'w-full')
        self.plot_element.style(f'height: {self.height}')
        
        return self.plot_element
    
    def _setup_layout(self):
        """Setup 3D plot layout."""
        self.figure.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',  # Force cube aspect ratio
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis=dict(
                    showgrid=True,
                    showticklabels=True,  # Show axis labels
                    range=[-50, 50]  # Default range to ensure 3D appearance
                ),
                yaxis=dict(
                    showgrid=True,
                    showticklabels=True,  # Show axis labels
                    range=[-50, 50]  # Default range to ensure 3D appearance
                ),
                zaxis=dict(
                    showgrid=True,
                    showticklabels=True,  # Show axis labels
                    range=[-50, 50]  # Default range to ensure 3D appearance
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,  # Show legend
            hovermode='closest',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True
        )
    
    def add_electrode(
        self,
        electrode_id: str,
        trajectory: np.ndarray,
        contacts: np.ndarray,
        color: str = 'blue',
        name: str = None
    ):
        """Add electrode visualization.
        
        Args:
            electrode_id: Unique ID for the electrode
            trajectory: Nx3 array of trajectory points
            contacts: Mx3 array of contact positions
            color: Color for the electrode
            name: Display name
        """
        # Electrode trajectory line
        trajectory_trace = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines',
            line=dict(color=color, width=4),
            name=name or f'Electrode {electrode_id}',
            showlegend=True
        )
        
        # Add trajectory trace
        self.traces[f'{electrode_id}_trajectory'] = trajectory_trace
        self.figure.add_trace(trajectory_trace)
        
        # Contact points (only if contacts array is not empty and 2D)
        if contacts.size > 0 and len(contacts.shape) == 2 and contacts.shape[1] >= 3:
            contacts_trace = go.Scatter3d(
                x=contacts[:, 0],
                y=contacts[:, 1],
                z=contacts[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f'{name or electrode_id} Contacts',
                showlegend=False
            )
            self.traces[f'{electrode_id}_contacts'] = contacts_trace
            self.figure.add_trace(contacts_trace)
        
        # Update plot and adjust ranges based on data
        if self.plot_element:
            self._update_ranges_for_data()
            self.plot_element.update()
    
    def add_isosurface(
        self,
        surface_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: str = 'gray',
        opacity: float = 0.3,
        name: str = 'Surface'
    ):
        """Add isosurface visualization.
        
        Args:
            surface_id: Unique ID for the surface
            vertices: Nx3 array of vertex positions
            faces: Mx3 array of face indices
            color: Surface color
            opacity: Surface opacity
            name: Display name
        """
        surface_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            showlegend=True
        )
        
        self.traces[surface_id] = surface_trace
        self.figure.add_trace(surface_trace)
        
        if self.plot_element:
            self._update_ranges_for_data()
            self.plot_element.update()
    
    def add_points(
        self,
        points_id: str,
        points: np.ndarray,
        color: str = 'red',
        size: int = 5,
        name: str = 'Points'
    ):
        """Add point cloud visualization.
        
        Args:
            points_id: Unique ID for the points
            points: Nx3 array of points
            color: Point color
            size: Point size
            name: Display name
        """
        points_trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
            ),
            name=name
        )
        
        self.traces[points_id] = points_trace
        self.figure.add_trace(points_trace)
        
        if self.plot_element:
            self._update_ranges_for_data()
            self.plot_element.update()
    
    def clear(self):
        """Clear all traces."""
        # Store camera position before clearing
        camera_state = None
        try:
            # Try to get the current camera state from the figure layout
            if self.figure.layout.scene and hasattr(self.figure.layout.scene, 'camera'):
                camera_state = dict(self.figure.layout.scene.camera)
        except:
            pass
            
        self.traces.clear()
        # Clear all data but keep the invisible point for 3D view
        self.figure.data = []
        self.figure.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=0.1, opacity=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Restore camera position after clearing
        if camera_state:
            # Preserve the camera state when updating layout
            self.figure.update_layout(
                scene=dict(
                    camera=camera_state,
                    xaxis_title='X (mm)',
                    yaxis_title='Y (mm)',
                    zaxis_title='Z (mm)',
                    aspectmode='cube',
                    xaxis=dict(showgrid=True, showticklabels=False),
                    yaxis=dict(showgrid=True, showticklabels=False),
                    zaxis=dict(showgrid=True, showticklabels=False)
                )
            )
        else:
            # Reset to default layout
            self._setup_layout()
            
        if self.plot_element:
            self._update_ranges_for_data()
            self.plot_element.update()
    
    def remove_trace(self, trace_id: str):
        """Remove a specific trace."""
        if trace_id in self.traces:
            # Find and remove the trace
            trace_to_remove = self.traces[trace_id]
            new_data = [trace for trace in self.figure.data if trace != trace_to_remove]
            self.figure.data = new_data
            del self.traces[trace_id]
            
            if self.plot_element:
                self.plot_element.update()
    
    def update_camera(self, eye: Dict[str, float], center: Dict[str, float] = None):
        """Update camera position.
        
        Args:
            eye: Camera eye position dict with x, y, z
            center: Camera center position (optional)
        """
        camera_dict = {'eye': eye}
        if center:
            camera_dict['center'] = center
            
        self.figure.update_layout(scene_camera=camera_dict)
        
        if self.plot_element:
            self._update_ranges_for_data()
            self.plot_element.update()
    
    def _update_ranges_for_data(self):
        """Update axis ranges to fit all data (excluding invisible point)."""
        if not self.figure.data or len(self.figure.data) <= 1:
            # Only invisible point or no data
            return
            
        # Collect all x, y, z values (skip invisible marker)
        all_x, all_y, all_z = [], [], []
        for trace in self.figure.data:
            # Skip the invisible point
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'opacity') and trace.marker.opacity == 0:
                continue
                
            if hasattr(trace, 'x') and trace.x is not None:
                all_x.extend(trace.x)
            if hasattr(trace, 'y') and trace.y is not None:
                all_y.extend(trace.y)
            if hasattr(trace, 'z') and trace.z is not None:
                all_z.extend(trace.z)
        
        if all_x and all_y and all_z:
            # Calculate ranges
            x_range = [min(all_x), max(all_x)]
            y_range = [min(all_y), max(all_y)]
            z_range = [min(all_z), max(all_z)]
            
            # Find the maximum range to use for all axes
            x_span = x_range[1] - x_range[0]
            y_span = y_range[1] - y_range[0]
            z_span = z_range[1] - z_range[0]
            max_span = max(x_span, y_span, z_span)
            
            # Calculate center points
            x_center = (x_range[0] + x_range[1]) / 2
            y_center = (y_range[0] + y_range[1]) / 2
            z_center = (z_range[0] + z_range[1]) / 2
            
            # Apply padding to the maximum span
            padding = 0.1
            padded_span = max_span * (1 + padding)
            half_span = padded_span / 2
            
            # Set equal ranges for all axes centered on data
            self.figure.update_layout(
                scene=dict(
                    xaxis=dict(range=[x_center - half_span, x_center + half_span]),
                    yaxis=dict(range=[y_center - half_span, y_center + half_span]),
                    zaxis=dict(range=[z_center - half_span, z_center + half_span]),
                    aspectmode='cube'  # Keep cube aspect
                )
            )


def create_3d_viewer(width: str = '100%', height: str = '600px') -> Plotly3DViewer:
    """Create a 3D viewer component."""
    viewer = Plotly3DViewer(width, height)
    viewer.create()
    return viewer