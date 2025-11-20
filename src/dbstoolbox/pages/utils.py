"""Utils page with visualization tools."""

from typing import Tuple
from nicegui import ui, events, app
from pathlib import Path
import asyncio
import json
import csv as csv_module
import numpy as np
import plotly.graph_objects as go
import uuid

from dbstoolbox.utils.notifications import notify_info, notify_error, notify_success
from dbstoolbox.utils.validate_electrode_json import validate_electrode_reconstruction
from dbstoolbox.utils.validate_surgical_csv import validate_surgical_csv
from dbstoolbox.utils.validate_frame_fiducials import validate_frame_fiducials
from dbstoolbox.utils.validate_nifti import validate_nifti, load_nifti_for_visualization
from dbstoolbox.utils.temp_file_manager import get_session_dir, save_uploaded_file, cleanup_session
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter


class VisualizePage:
    """Visualization utilities page."""

    def __init__(self):
        self.loaded_files = []  # List of loaded file dictionaries
        self.session_id = str(uuid.uuid4())  # Unique session ID for temp files

        # Mesh rendering parameters
        self.mesh_opacity = 0.5  # Default opacity
        self.mesh_smoothness = 0.5  # Gaussian sigma for smoothing
        self.mesh_cache = {}  # Cache for extracted meshes: {(filename, smoothness): [(verts, faces), ...]}
        self.nifti_labels = {}  # Labels for NIfTI volumes: {filename: {volume_idx: label_name}}

        # UI elements
        self.upload_element = None
        self.upload_loading_container = None  # Loading indicator for file upload
        self.status_label = None
        self.plot_card = None  # Card containing the 3D plot
        self.plot_container = None
        self.maximize_btn = None  # Button to maximize/restore plot
        self.file_list_container = None
        self.clear_all_btn = None
        self.opacity_slider = None
        self.plotly_element = None  # Reference to the plotly UI element
        self.current_figure = None  # Reference to current Plotly figure
        self.plot_maximized = False  # Track maximize state

        # Virtual trajectory controls
        self.virtual_trajectory_container = None
        self.virtual_trajectory_enabled = False
        self.virtual_target_x = None
        self.virtual_target_y = None
        self.virtual_target_z = None
        self.virtual_ring_slider = None
        self.virtual_arc_slider = None
        self.virtual_trajectory_trace_idx = None  # Track which trace is the virtual trajectory

        # Register cleanup on disconnect
        app.on_disconnect(lambda: cleanup_session(self.session_id))

    def create_ui(self):
        """Create the visualization interface."""
        # Add global CSS to hide uploader file lists
        ui.add_head_html('''
            <style>
            .viz-hide-upload-list .q-uploader__list {
                display: none !important;
            }
            </style>
        ''')

        with ui.column().classes('w-full h-full p-4 gap-4'):
            # Header
            with ui.row().classes('w-full items-center mb-2'):
                ui.icon('visibility', size='lg').classes('text-primary')
                ui.label('3D Visualization').classes('text-h5 ml-2')
                ui.label('Visualize surgical targeting and electrode reconstructions').classes(
                    'text-body2 text-grey-6 ml-4'
                )

            # Upload area and plot in a row
            with ui.row().classes('w-full gap-4 flex-1'):
                # Left: Upload area
                with ui.card().classes('w-80'):
                    ui.label('Load Data').classes('text-subtitle2 mb-3')

                    # Upload area
                    self.upload_element = ui.upload(
                        auto_upload=True,
                        on_upload=self._handle_file_upload,
                        label='Drag & drop or click to upload',
                        max_file_size=100 * 1024 * 1024  # 100MB
                    ).props('accept=".csv,.json,.nii,.nii.gz,.txt" square outlined hide-upload-btn').classes(
                        'w-full border-2 border-dashed border-grey-5 rounded viz-hide-upload-list'
                    )

                    # Add hover effect
                    self.upload_element.on('dragover', lambda: self.upload_element.classes(remove='border-grey-5', add='border-primary'))
                    self.upload_element.on('dragleave', lambda: self.upload_element.classes(remove='border-primary', add='border-grey-5'))
                    self.upload_element.on('drop', lambda: self.upload_element.classes(remove='border-primary', add='border-grey-5'))

                    # Loading indicator for file upload
                    self.upload_loading_container = ui.column().classes('w-full mt-2')
                    with self.upload_loading_container:
                        with ui.row().classes('w-full items-center gap-2'):
                            ui.spinner(size='sm', color='primary')
                            ui.label('Loading file...').classes('text-caption text-grey-7')
                    self.upload_loading_container.set_visibility(False)

                    # Status
                    with ui.row().classes('w-full items-center justify-between gap-2 mt-3'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('info', size='sm').classes('text-grey-6')
                            self.status_label = ui.label('No files loaded').classes('text-caption text-grey-7')

                        # Clear all button (initially hidden)
                        self.clear_all_btn = ui.button(
                            icon='clear_all',
                            on_click=self._clear_all_files
                        ).props('flat dense round size=sm color=negative')
                        self.clear_all_btn.set_visibility(False)

                    ui.separator().classes('my-3')

                    # Info about supported formats (collapsible)
                    with ui.expansion('Supported Formats', icon='help_outline').classes('w-full').props('dense'):
                        with ui.column().classes('gap-1 p-2'):
                            with ui.row().classes('items-start gap-2'):
                                ui.icon('description', size='xs').classes('text-grey-6')
                                ui.label('CSV: Surgical targeting data').classes('text-caption text-grey-7')
                            with ui.row().classes('items-start gap-2'):
                                ui.icon('data_object', size='xs').classes('text-grey-6')
                                ui.label('JSON: Electrode, surgical, or fiducials').classes('text-caption text-grey-7')
                            with ui.row().classes('items-start gap-2'):
                                ui.icon('view_in_ar', size='xs').classes('text-grey-6')
                                ui.label('NIfTI: 3D/4D probability maps').classes('text-caption text-grey-7')
                            with ui.row().classes('items-start gap-2'):
                                ui.icon('label', size='xs').classes('text-grey-6')
                                ui.label('TXT: Volume labels for NIfTI').classes('text-caption text-grey-7')

                    ui.separator().classes('my-3')

                    # Loaded files list
                    ui.label('Loaded Files').classes('text-caption font-medium mb-2')
                    self.file_list_container = ui.column().classes('w-full gap-2')
                    self._update_file_list()

                # Right: 3D Plot (fills remaining space)
                self.plot_card = ui.card().classes('flex-1')
                with self.plot_card:
                    # Header with title and maximize button
                    with ui.row().classes('w-full items-center justify-between mb-3'):
                        ui.label('3D View').classes('text-subtitle2')
                        self.maximize_btn = ui.button(
                            icon='fullscreen',
                            on_click=self._toggle_plot_maximize
                        ).props('flat dense round size=sm')
                        self.maximize_btn.tooltip('Maximize plot')

                    # Plot container with relative positioning for overlay
                    with ui.column().classes('w-full h-full relative'):
                        self.plot_container = ui.column().classes('w-full h-full')
                        self._update_plot()

                        # Virtual trajectory controls as overlay (initially hidden)
                        self.virtual_trajectory_container = ui.row().classes('absolute bottom-0 left-0 right-0 items-center gap-3 p-2').style('background: transparent; pointer-events: auto;')
                        with self.virtual_trajectory_container:
                            # Checkbox (no text)
                            ui.checkbox(value=False, on_change=self._on_virtual_trajectory_toggle).props('dense').tooltip('Show virtual trajectory')

                            # Target coordinates
                            with ui.row().classes('items-center gap-2'):
                                ui.label('X').classes('text-caption')
                                self.virtual_target_x = ui.number(
                                    value=100.0,
                                    min=0,
                                    max=200,
                                    step=0.5,
                                    on_change=self._on_virtual_trajectory_change
                                ).props('dense outlined').classes('w-16').style('font-size: 0.75rem;')

                                ui.label('Y').classes('text-caption')
                                self.virtual_target_y = ui.number(
                                    value=100.0,
                                    min=0,
                                    max=200,
                                    step=0.5,
                                    on_change=self._on_virtual_trajectory_change
                                ).props('dense outlined').classes('w-16').style('font-size: 0.75rem;')

                                ui.label('Z').classes('text-caption')
                                self.virtual_target_z = ui.number(
                                    value=100.0,
                                    min=0,
                                    max=200,
                                    step=0.5,
                                    on_change=self._on_virtual_trajectory_change
                                ).props('dense outlined').classes('w-16').style('font-size: 0.75rem;')

                            # Ring angle slider (compact)
                            with ui.row().classes('items-center gap-1'):
                                ui.label('Ring').classes('text-caption')
                                self.virtual_ring_slider = ui.slider(
                                    min=0,
                                    max=180,
                                    step=1,
                                    value=90
                                ).props('label-always color=primary dense').classes('w-32').style('font-size: 0.7rem;')
                                self.virtual_ring_slider.on('update:model-value', self._on_virtual_trajectory_change)

                            # Arc angle slider (compact)
                            with ui.row().classes('items-center gap-1'):
                                ui.label('Arc').classes('text-caption')
                                self.virtual_arc_slider = ui.slider(
                                    min=0,
                                    max=180,
                                    step=1,
                                    value=90
                                ).props('label-always color=primary dense').classes('w-32').style('font-size: 0.7rem;')
                                self.virtual_arc_slider.on('update:model-value', self._on_virtual_trajectory_change)

                        # Hide virtual trajectory controls initially
                        self.virtual_trajectory_container.set_visibility(False)

                        # Mesh opacity control as overlay (initially hidden)
                        self.mesh_opacity_container = ui.row().classes('absolute bottom-0 left-0 right-0 items-center gap-3 p-2').style('background: rgba(30, 30, 30, 0.7); pointer-events: auto;')
                        with self.mesh_opacity_container:
                            ui.label('Mesh Opacity').classes('text-caption text-white')
                            self.opacity_slider = ui.slider(
                                min=0.1,
                                max=1.0,
                                step=0.05,
                                value=0.5
                            ).props('label-always color=primary dense').classes('w-64').style('font-size: 0.7rem;')
                            # Only update on mouse release
                            self.opacity_slider.on('change', self._on_opacity_change)

                        # Hide opacity controls initially
                        self.mesh_opacity_container.set_visibility(False)

    async def _handle_file_upload(self, e: events.UploadEventArguments):
        """Handle file upload."""
        # Show loading indicator
        self.upload_loading_container.set_visibility(True)

        try:
            uploaded_file = e.file
            file_name = uploaded_file.name
            file_content = await uploaded_file.read()

            # Check if file is already loaded
            if any(f['filename'] == file_name for f in self.loaded_files):
                notify_info(f'{file_name} is already loaded')
                self.upload_loading_container.set_visibility(False)
                return

            # Save to session directory
            temp_path = save_uploaded_file(file_content, file_name, self.session_id)

            file_size_kb = len(file_content) / 1024

            # Determine file type and validate
            if file_name.endswith('.csv'):
                await self._load_surgical_csv(temp_path, file_name, file_size_kb)
            elif file_name.endswith('.json'):
                await self._load_json_file(temp_path, file_name, file_size_kb)
            elif file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                await self._load_nifti_file(temp_path, file_name, file_size_kb)
            elif file_name.endswith('.txt'):
                await self._load_label_file(temp_path, file_name, file_size_kb)
            else:
                notify_error('Unsupported file type. Please upload CSV, JSON, NIfTI, or TXT.')
                return

            # Update displays
            import time
            t1 = time.time()
            self._update_file_list()
            t2 = time.time()
            print(f"[Upload] _update_file_list took {t2-t1:.3f}s")

            self._update_status()
            t3 = time.time()
            print(f"[Upload] _update_status took {t3-t2:.3f}s")

            self._update_mesh_controls_visibility()
            t4 = time.time()
            print(f"[Upload] _update_mesh_controls_visibility took {t4-t3:.3f}s")

            # Update plot to show all loaded files
            # NIfTI meshes will only render if already in cache
            self._update_plot()
            t5 = time.time()
            print(f"[Upload] _update_plot took {t5-t4:.3f}s")
            print(f"[Upload] Total UI updates took {t5-t1:.3f}s")

            # Update virtual trajectory visibility
            self._update_virtual_trajectory_visibility()

        except Exception as ex:
            notify_error(f'Failed to load file: {str(ex)}')
            import traceback
            traceback.print_exc()
        finally:
            # Always hide loading indicator
            self.upload_loading_container.set_visibility(False)

    async def _load_surgical_csv(self, file_path: Path, file_name: str, file_size_kb: float):
        """Load and validate surgical CSV data."""
        # Validate
        is_valid, metadata, error_msg = validate_surgical_csv(file_path)

        if not is_valid:
            notify_error(f'Invalid surgical CSV: {error_msg}')
            return

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            data = list(reader)

        # Detect if this is raw (untransformed) data
        # Raw data won't have 'entry_x' or 'x_original' fields
        is_raw = True
        if data:
            first_row = data[0]
            if 'entry_x' in first_row or 'x_original' in first_row:
                is_raw = False

        # Add to loaded files list with surgical-specific toggles
        self.loaded_files.append({
            'filename': file_name,
            'type': 'surgical',
            'data': data,
            'metadata': metadata,
            'size_kb': file_size_kb,
            'file_path': file_path,
            'visible': True,
            'show_mer_tracks': True,
            'show_trajectory': True,
            'show_targets': True,
            'show_frame': False,  # Default off
            'is_raw': is_raw
        })

        notify_success(f'Loaded {metadata["num_records"]} surgical records from {file_name}')

    async def _load_json_file(self, file_path: Path, file_name: str, file_size_kb: float):
        """Load and validate JSON file (electrode, surgical data, or frame fiducials)."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if it's electrode reconstruction
        is_electrode, metadata, error_msg = validate_electrode_reconstruction(file_path)

        if is_electrode:
            # Add to loaded files list with electrode-specific toggles
            self.loaded_files.append({
                'filename': file_name,
                'type': 'electrode',
                'data': data,
                'metadata': metadata,
                'size_kb': file_size_kb,
                'file_path': file_path,
                'visible': True,
                'show_trajectory': True
            })
            notify_success(f'Loaded {metadata["num_electrodes"]} electrodes from {file_name}')

        else:
            # Check if it's frame fiducials
            is_fiducials, fiducial_metadata, fiducial_error = validate_frame_fiducials(file_path)

            if is_fiducials:
                # Add to loaded files list
                self.loaded_files.append({
                    'filename': file_name,
                    'type': 'fiducials',
                    'data': data,
                    'metadata': fiducial_metadata,
                    'size_kb': file_size_kb,
                    'file_path': file_path,
                    'visible': True
                })
                notify_success(f'Loaded {fiducial_metadata["num_fiducials"]} frame fiducials from {file_name}')

            # Check if it's surgical data in JSON format
            elif 'records' in data or 'surgical_data' in data:
                # Extract records
                records = data.get('records', data.get('surgical_data', []))

                # Detect if this is raw (untransformed) data
                is_raw = True
                if records:
                    first_row = records[0]
                    if 'entry_x' in first_row or 'x_original' in first_row:
                        is_raw = False

                # Add to loaded files list with surgical-specific toggles
                self.loaded_files.append({
                    'filename': file_name,
                    'type': 'surgical',
                    'data': records,
                    'metadata': {'num_records': len(records), 'format': 'json'},
                    'size_kb': file_size_kb,
                    'file_path': file_path,
                    'visible': True,
                    'show_mer_tracks': True,
                    'show_trajectory': True,
                    'show_targets': True,
                    'show_frame': False,  # Default off
                    'is_raw': is_raw
                })
                notify_success(f'Loaded {len(records)} surgical records from {file_name}')

            else:
                notify_error('Unrecognized JSON format. Expected electrode reconstruction, frame fiducials, or surgical data.')

    async def _load_nifti_file(self, file_path: Path, file_name: str, file_size_kb: float):
        """Load and validate NIfTI file for surface visualization."""
        # Validate (run in executor to avoid blocking)
        import asyncio
        import concurrent.futures
        import time

        loop = asyncio.get_event_loop()

        print(f"[NIfTI Load] Starting load for {file_name}")
        t_start = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1 = time.time()
            is_valid, metadata, error_msg = await loop.run_in_executor(
                executor,
                validate_nifti,
                file_path
            )
            t2 = time.time()
            print(f"[NIfTI Load] Validation took {t2-t1:.3f}s")

        if not is_valid:
            notify_error(f'Invalid NIfTI file: {error_msg}')
            return

        # Load the data (run in executor to avoid blocking)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t1 = time.time()
            data, affine, load_error = await loop.run_in_executor(
                executor,
                load_nifti_for_visualization,
                file_path
            )
            t2 = time.time()
            print(f"[NIfTI Load] Data loading took {t2-t1:.3f}s")

        if load_error:
            notify_error(f'Failed to load NIfTI: {load_error}')
            return

        # Calculate data range (metadata has it as None initially)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        metadata['data_range'] = (data_min, data_max)

        # Calculate default threshold (midpoint of data range)
        default_threshold = data_min + (data_max - data_min) * 0.5

        # Add to loaded files list
        self.loaded_files.append({
            'filename': file_name,
            'type': 'nifti',
            'data': data,
            'affine': affine,
            'metadata': metadata,
            'size_kb': file_size_kb,
            'file_path': file_path,
            'visible': True,
            'smoothness': 0.5,
            'threshold': default_threshold
        })

        t_end = time.time()
        print(f"[NIfTI Load] Total NIfTI load took {t_end-t_start:.3f}s")

        # Build success message
        shape_str = ' × '.join(str(s) for s in metadata['shape'])
        if metadata['dimensions'] == 4:
            notify_success(f'Loaded 4D NIfTI with {metadata["num_volumes"]} volumes ({shape_str}) from {file_name}')
        else:
            notify_success(f'Loaded 3D NIfTI ({shape_str}) from {file_name}')

    async def _load_label_file(self, file_path: Path, file_name: str, file_size_kb: float):
        """Load and validate label file for NIfTI volumes."""
        try:
            # Read the label file
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Parse labels (format: "index<whitespace>label")
            labels = {}
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue

                parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                if len(parts) != 2:
                    notify_error(f'Invalid label file format at line: {line}')
                    return

                try:
                    idx = int(parts[0])
                    label = parts[1].strip()
                    labels[idx] = label
                except ValueError:
                    notify_error(f'Invalid index in label file: {parts[0]}')
                    return

            if not labels:
                notify_error('Label file is empty or has no valid labels')
                return

            # Find matching NIfTI files
            nifti_files = [f for f in self.loaded_files if f['type'] == 'nifti']

            if not nifti_files:
                notify_error('No NIfTI files loaded. Load a NIfTI file first before uploading labels.')
                return

            # Check if number of labels matches any NIfTI file
            matched_nifti = None
            for nifti_file in nifti_files:
                num_volumes = nifti_file['metadata'].get('num_volumes', 1)
                if len(labels) == num_volumes:
                    matched_nifti = nifti_file
                    break

            if not matched_nifti:
                expected_volumes = [f['metadata'].get('num_volumes', 1) for f in nifti_files]
                notify_error(f'Label count ({len(labels)}) does not match any loaded NIfTI volumes ({expected_volumes})')
                return

            # Update labels for the matched NIfTI file
            nifti_filename = matched_nifti['filename']
            self.nifti_labels[nifti_filename] = labels

            # Add label file to loaded_files (for tracking, won't show in UI)
            self.loaded_files.append({
                'filename': file_name,
                'type': 'labels',
                'nifti_target': nifti_filename,  # Track which NIfTI this applies to
                'labels': labels,
                'size_kb': file_size_kb,
                'file_path': file_path
            })

            notify_success(f'Loaded {len(labels)} labels for {nifti_filename}')

            # Refresh the plot to show new labels
            self._update_plot()

        except Exception as e:
            notify_error(f'Failed to load label file: {str(e)}')
            import traceback
            traceback.print_exc()

    def _truncate_filename(self, filename: str, max_length: int = 20) -> str:
        """Truncate a filename to a maximum length, preserving the extension."""
        if len(filename) <= max_length:
            return filename

        # Split into name and extension
        if '.' in filename:
            parts = filename.rsplit('.', 1)
            name = parts[0]
            ext = '.' + parts[1]

            # Handle .nii.gz double extension
            if name.endswith('.nii'):
                name = name[:-4]
                ext = '.nii.gz'

            # Calculate available space for name
            available = max_length - len(ext) - 3  # 3 for "..."

            if available > 0:
                return name[:available] + '...' + ext
            else:
                # If extension is too long, just truncate everything
                return filename[:max_length-3] + '...'
        else:
            # No extension
            return filename[:max_length-3] + '...'

    def _update_file_list(self):
        """Update the list of loaded files."""
        self.file_list_container.clear()

        # Filter out label files (they're tracked internally but not shown in UI)
        visible_files = [f for f in self.loaded_files if f['type'] != 'labels']

        with self.file_list_container:
            if not visible_files:
                with ui.column().classes('w-full items-center justify-center p-4 border border-dashed border-grey-4 rounded'):
                    ui.icon('folder_off', size='sm').classes('text-grey-5 mb-1')
                    ui.label('No files loaded').classes('text-caption text-grey-6 italic')
            else:
                for file_info in visible_files:
                    with ui.card().classes('w-full p-2'):
                        with ui.row().classes('w-full items-center justify-between gap-2'):
                            # File info
                            with ui.column().classes('gap-0 flex-grow'):
                                with ui.row().classes('items-center gap-2'):
                                    # Icon based on type
                                    if file_info['type'] == 'electrode':
                                        icon_name = 'sensors'
                                    elif file_info['type'] == 'fiducials':
                                        icon_name = 'grain'
                                    elif file_info['type'] == 'nifti':
                                        icon_name = 'view_in_ar'
                                    else:
                                        icon_name = 'my_location'
                                    ui.icon(icon_name, size='xs').classes('text-primary')

                                    # Truncate filename if too long, show full name on hover
                                    full_filename = file_info['filename']
                                    display_filename = self._truncate_filename(full_filename)
                                    ui.label(display_filename).classes('text-caption font-medium').tooltip(full_filename)

                                # Type badge(s)
                                with ui.row().classes('items-center gap-1 mt-1'):
                                    if file_info['type'] == 'electrode':
                                        type_label = 'Electrode'
                                    elif file_info['type'] == 'fiducials':
                                        type_label = 'Fiducials'
                                    elif file_info['type'] == 'nifti':
                                        type_label = 'NIfTI 3D' if file_info['metadata']['dimensions'] == 3 else f'NIfTI 4D ({file_info["metadata"]["num_volumes"]})'
                                    else:
                                        type_label = 'Surgical'
                                    ui.chip(type_label).props('dense').classes('text-caption')

                                    # Labels badge for NIfTI files with remove button
                                    if file_info['type'] == 'nifti' and file_info['filename'] in self.nifti_labels:
                                        num_labels = len(self.nifti_labels[file_info['filename']])
                                        with ui.chip().props('dense').classes('text-caption bg-grey-7 text-white'):
                                            ui.label(f'Labels ({num_labels})')
                                            ui.button(icon='close', on_click=lambda f=file_info: self._remove_labels(f)).props(
                                                'flat dense round size=xs color=red'
                                            ).classes('ml-1').style('margin: -4px; padding: 2px;')

                                # Electrode-specific toggle buttons
                                if file_info['type'] == 'electrode':
                                    with ui.row().classes('items-center gap-1 mt-1'):
                                        # Trajectory toggle
                                        traj_icon = 'timeline' if file_info.get('show_trajectory', True) else 'timeline'
                                        traj_color = 'primary' if file_info.get('show_trajectory', True) else 'grey'
                                        ui.button(
                                            icon=traj_icon,
                                            on_click=lambda f=file_info: self._toggle_electrode_trajectory(f)
                                        ).props(f'flat dense round size=xs color={traj_color}').tooltip('Toggle trajectory')

                                # Surgical-specific toggle buttons
                                if file_info['type'] == 'surgical':
                                    with ui.row().classes('items-center gap-1 mt-1'):
                                        # MER tracks toggle
                                        mer_icon = 'grid_on' if file_info.get('show_mer_tracks', True) else 'grid_off'
                                        mer_color = 'primary' if file_info.get('show_mer_tracks', True) else 'grey'
                                        ui.button(
                                            icon=mer_icon,
                                            on_click=lambda f=file_info: self._toggle_surgical_mer_tracks(f)
                                        ).props(f'flat dense round size=xs color={mer_color}').tooltip('Toggle MER tracks')

                                        # Trajectory toggle
                                        traj_icon = 'timeline' if file_info.get('show_trajectory', True) else 'timeline'
                                        traj_color = 'primary' if file_info.get('show_trajectory', True) else 'grey'
                                        ui.button(
                                            icon=traj_icon,
                                            on_click=lambda f=file_info: self._toggle_surgical_trajectory(f)
                                        ).props(f'flat dense round size=xs color={traj_color}').tooltip('Toggle trajectory')

                                        # Targets toggle
                                        targets_icon = 'control_point' if file_info.get('show_targets', True) else 'control_point'
                                        targets_color = 'primary' if file_info.get('show_targets', True) else 'grey'
                                        ui.button(
                                            icon=targets_icon,
                                            on_click=lambda f=file_info: self._toggle_surgical_targets(f)
                                        ).props(f'flat dense round size=xs color={targets_color}').tooltip('Toggle clinical/research targets')

                                        # Frame toggle (only for raw surgical data)
                                        if file_info.get('is_raw', False):
                                            frame_icon = 'crop_square' if file_info.get('show_frame', False) else 'crop_square'
                                            frame_color = 'primary' if file_info.get('show_frame', False) else 'grey'
                                            ui.button(
                                                icon=frame_icon,
                                                on_click=lambda f=file_info: self._toggle_surgical_frame(f)
                                            ).props(f'flat dense round size=xs color={frame_color}').tooltip('Toggle Leksell frame')

                            # Visibility toggle and remove buttons
                            with ui.row().classes('gap-1'):
                                # Visibility toggle button
                                visibility_icon = 'visibility' if file_info['visible'] else 'visibility_off'
                                ui.button(
                                    icon=visibility_icon,
                                    on_click=lambda f=file_info: self._toggle_file_visibility(f)
                                ).props('flat dense round size=sm')

                                # Remove button
                                ui.button(
                                    icon='close',
                                    on_click=lambda f=file_info: self._remove_file(f)
                                ).props('flat dense round size=sm')

                        # NIfTI-specific controls (below the main row)
                        if file_info['type'] == 'nifti':
                            with ui.column().classes('w-full gap-2 mt-2 p-2 rounded').style('background: transparent;'):
                                # Sliders in horizontal layout
                                with ui.row().classes('w-full gap-3'):
                                    # Threshold slider (left)
                                    with ui.column().classes('flex-1 gap-0'):
                                        data_min, data_max = file_info['metadata']['data_range']
                                        threshold_value = file_info.get('threshold', data_min + (data_max - data_min) * 0.5)
                                        threshold_slider = ui.slider(
                                            min=data_min,
                                            max=data_max,
                                            step=(data_max - data_min) / 100,
                                            value=threshold_value
                                        ).props('color=primary dense').classes('w-full')
                                        threshold_slider.on('change', lambda e, f=file_info: self._on_nifti_threshold_change(f, e.args))
                                        with ui.row().classes('items-center gap-2'):
                                            ui.label('Threshold').classes('text-caption text-grey-4')
                                            ui.label().classes('text-caption text-grey-3 font-mono').bind_text_from(
                                                threshold_slider, 'value', backward=lambda v: f'{v:.2f}'
                                            )

                                    # Smoothness slider (right)
                                    with ui.column().classes('flex-1 gap-0'):
                                        smoothness_value = file_info.get('smoothness', 0.5)
                                        smoothness_slider = ui.slider(
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=smoothness_value
                                        ).props('color=primary dense').classes('w-full')
                                        smoothness_slider.on('change', lambda e, f=file_info: self._on_nifti_smoothness_change(f, e.args))
                                        with ui.row().classes('items-center gap-2'):
                                            ui.label('Smoothness').classes('text-caption text-grey-4')
                                            ui.label().classes('text-caption text-grey-3 font-mono').bind_text_from(
                                                smoothness_slider, 'value', backward=lambda v: f'{v:.2f}'
                                            )

                                # Generate mesh button
                                ui.button(
                                    'Generate Mesh',
                                    on_click=lambda f=file_info: self._on_generate_mesh_for_file(f),
                                    icon='auto_fix_high'
                                ).props('dense color=primary').classes('w-full').style('font-size: 0.7rem;')

    def _update_status(self):
        """Update the status label and clear button visibility."""
        # Filter out label files (internal tracking only)
        visible_files = [f for f in self.loaded_files if f['type'] != 'labels']

        if not visible_files:
            self.status_label.set_text('No files loaded')
            self.clear_all_btn.set_visibility(False)
        else:
            num_surgical = sum(1 for f in visible_files if f['type'] == 'surgical')
            num_electrode = sum(1 for f in visible_files if f['type'] == 'electrode')
            num_fiducials = sum(1 for f in visible_files if f['type'] == 'fiducials')
            num_nifti = sum(1 for f in visible_files if f['type'] == 'nifti')

            parts = []
            if num_surgical > 0:
                parts.append(f'{num_surgical} surgical')
            if num_electrode > 0:
                parts.append(f'{num_electrode} electrode')
            if num_fiducials > 0:
                parts.append(f'{num_fiducials} fiducials')
            if num_nifti > 0:
                parts.append(f'{num_nifti} nifti')

            self.status_label.set_text(f'{len(visible_files)} files: ' + ', '.join(parts))
            self.clear_all_btn.set_visibility(True)

    def _update_mesh_controls_visibility(self):
        """Update mesh opacity overlay visibility based on whether NIfTI files are loaded."""
        has_nifti = any(f['type'] == 'nifti' for f in self.loaded_files)
        self.mesh_opacity_container.set_visibility(has_nifti)

    def _toggle_surgical_mer_tracks(self, file_info: dict):
        """Toggle MER tracks visibility for surgical data."""
        file_info['show_mer_tracks'] = not file_info.get('show_mer_tracks', True)

        # Toggle visibility using JavaScript to preserve camera (BEFORE updating file list)
        visible = file_info['show_mer_tracks']
        ui.run_javascript(f'''
            const plot = document.querySelector(".js-plotly-plot");
            if (plot && plot.data) {{
                const updates = {{}};
                const indices = [];

                // Find all MER track traces (contain " MER " in name)
                for (let i = 0; i < plot.data.length; i++) {{
                    if (plot.data[i].name && plot.data[i].name.includes(' MER ')) {{
                        indices.push(i);
                    }}
                }}

                // Update visibility
                if (indices.length > 0) {{
                    updates.visible = {str(visible).lower()};
                    Plotly.restyle(plot, updates, indices);
                }}
            }}
        ''')

        # Update file list after JavaScript runs
        self._update_file_list()

    def _toggle_surgical_trajectory(self, file_info: dict):
        """Toggle trajectory visibility for surgical data."""
        file_info['show_trajectory'] = not file_info.get('show_trajectory', True)

        # Toggle visibility using JavaScript to preserve camera (BEFORE updating file list)
        visible = file_info['show_trajectory']
        ui.run_javascript(f'''
            const plot = document.querySelector(".js-plotly-plot");
            if (plot && plot.data) {{
                const updates = {{}};
                const indices = [];

                // Find trajectory traces (name ends with Target or Entry, or is the main trajectory without MER/Clinical/Research)
                for (let i = 0; i < plot.data.length; i++) {{
                    const name = plot.data[i].name;
                    if (name && !name.includes(' MER ') && !name.includes('Clinical') && !name.includes('Research') && !name.includes('Leksell Frame') && !name.includes('Virtual Trajectory')) {{
                        indices.push(i);
                    }}
                }}

                // Update visibility
                if (indices.length > 0) {{
                    updates.visible = {str(visible).lower()};
                    Plotly.restyle(plot, updates, indices);
                }}
            }}
        ''')

        # Update file list after JavaScript runs
        self._update_file_list()

    def _toggle_electrode_trajectory(self, file_info: dict):
        """Toggle trajectory visibility for electrode data."""
        file_info['show_trajectory'] = not file_info.get('show_trajectory', True)

        # Refresh the plot
        self._update_plot()
        self._update_file_list()

    def _toggle_surgical_targets(self, file_info: dict):
        """Toggle clinical/research targets visibility for surgical data."""
        file_info['show_targets'] = not file_info.get('show_targets', True)

        # Toggle visibility using JavaScript to preserve camera (BEFORE updating file list)
        visible = file_info['show_targets']
        ui.run_javascript(f'''
            const plot = document.querySelector(".js-plotly-plot");
            if (plot && plot.data) {{
                const updates = {{}};
                const indices = [];

                // Find clinical/research target traces
                for (let i = 0; i < plot.data.length; i++) {{
                    const name = plot.data[i].name;
                    if (name && (name.includes('Clinical') || name.includes('Research'))) {{
                        indices.push(i);
                    }}
                }}

                // Update visibility
                if (indices.length > 0) {{
                    updates.visible = {str(visible).lower()};
                    Plotly.restyle(plot, updates, indices);
                }}
            }}
        ''')

        # Update file list after JavaScript runs
        self._update_file_list()

    def _toggle_surgical_frame(self, file_info: dict):
        """Toggle Leksell frame visibility for raw surgical data."""
        file_info['show_frame'] = not file_info.get('show_frame', False)

        # Add or remove frame traces using JavaScript to preserve camera
        visible = file_info['show_frame']

        if visible:
            # Add frame traces
            # Leksell frame configuration with 6 fiducials
            frame_fiducials = [
                {'name': 'left_posterior', 'start': [195.0, 40.0, 40.0], 'end': [195.0, 40.0, 160.0]},
                {'name': 'left_oblique', 'start': [195.0, 40.0, 40.0], 'end': [195.0, 160.0, 160.0]},
                {'name': 'left_anterior', 'start': [195.0, 160.0, 40.0], 'end': [195.0, 160.0, 160.0]},
                {'name': 'right_posterior', 'start': [5.0, 40.0, 40.0], 'end': [5.0, 40.0, 160.0]},
                {'name': 'right_oblique', 'start': [5.0, 40.0, 40.0], 'end': [5.0, 160.0, 160.0]},
                {'name': 'right_anterior', 'start': [5.0, 160.0, 40.0], 'end': [5.0, 160.0, 160.0]}
            ]

            # Build JavaScript to add all fiducial traces
            traces_js = []
            for idx, fid in enumerate(frame_fiducials):
                show_legend = 'true' if idx == 0 else 'false'
                fid_display_name = fid['name'].replace("_", " ").title()
                trace_js = f'''{{
                    x: [{fid['start'][0]}, {fid['end'][0]}],
                    y: [{fid['start'][1]}, {fid['end'][1]}],
                    z: [{fid['start'][2]}, {fid['end'][2]}],
                    mode: 'lines',
                    type: 'scatter3d',
                    name: 'Leksell Frame',
                    line: {{width: 4, color: 'rgba(255, 255, 255, 0.3)'}},
                    legendgroup: 'leksell_frame',
                    showlegend: {show_legend},
                    hoverinfo: 'text',
                    text: ['Frame: {fid_display_name}', 'Frame: {fid_display_name}']
                }}'''
                traces_js.append(trace_js)

            traces_array = ',\n'.join(traces_js)

            js_code = f'''
                const plot = document.querySelector(".js-plotly-plot");
                console.log("Toggle frame ON - plot found:", plot ? "yes" : "no");
                if (plot && plot.data) {{
                    console.log("Current traces:", plot.data.length);

                    // Check if frame already exists
                    let hasFrame = false;
                    for (let i = 0; i < plot.data.length; i++) {{
                        if (plot.data[i].name === 'Leksell Frame') {{
                            hasFrame = true;
                            break;
                        }}
                    }}

                    console.log("Frame already exists:", hasFrame);

                    if (!hasFrame) {{
                        // Add frame traces
                        const frameTraces = [
                            {traces_array}
                        ];
                        console.log("Adding frame traces:", frameTraces.length);
                        Plotly.addTraces(plot, frameTraces).then(() => {{
                            console.log("Frame traces added successfully");

                            // Recalculate axis ranges to include frame
                            let allX = [], allY = [], allZ = [];
                            for (let trace of plot.data) {{
                                if (trace.x) allX.push(...trace.x.filter(v => v !== null && v !== undefined));
                                if (trace.y) allY.push(...trace.y.filter(v => v !== null && v !== undefined));
                                if (trace.z) allZ.push(...trace.z.filter(v => v !== null && v !== undefined));
                            }}

                            const xMin = Math.min(...allX), xMax = Math.max(...allX);
                            const yMin = Math.min(...allY), yMax = Math.max(...allY);
                            const zMin = Math.min(...allZ), zMax = Math.max(...allZ);

                            const xCenter = (xMin + xMax) / 2;
                            const yCenter = (yMin + yMax) / 2;
                            const zCenter = (zMin + zMax) / 2;

                            const xRange = xMax - xMin;
                            const yRange = yMax - yMin;
                            const zRange = zMax - zMin;

                            const maxRange = Math.max(xRange, yRange, zRange) * 1.1;
                            const halfRange = maxRange / 2;

                            // Update axis ranges while preserving camera
                            Plotly.relayout(plot, {{
                                'scene.xaxis.range': [xCenter - halfRange, xCenter + halfRange],
                                'scene.yaxis.range': [yCenter - halfRange, yCenter + halfRange],
                                'scene.zaxis.range': [zCenter - halfRange, zCenter + halfRange]
                            }}).then(() => {{
                                console.log("Axis ranges updated");
                            }});
                        }}).catch((err) => {{
                            console.error("Error adding frame traces:", err);
                        }});
                    }}
                }}
            '''

            ui.run_javascript(js_code)
        else:
            # Remove frame traces
            ui.run_javascript('''
                const plot = document.querySelector(".js-plotly-plot");
                if (plot && plot.data) {
                    // Find all Leksell frame traces
                    const indicesToRemove = [];
                    for (let i = plot.data.length - 1; i >= 0; i--) {
                        if (plot.data[i].name === 'Leksell Frame') {
                            indicesToRemove.push(i);
                        }
                    }

                    // Remove in reverse order to maintain indices
                    if (indicesToRemove.length > 0) {
                        Plotly.deleteTraces(plot, indicesToRemove);
                    }
                }
            ''')

        # Update file list after JavaScript runs
        self._update_file_list()
        self._update_virtual_trajectory_visibility()

    def _on_virtual_trajectory_toggle(self, e):
        """Handle virtual trajectory toggle."""
        self.virtual_trajectory_enabled = e.value
        self._update_virtual_trajectory()

    def _on_virtual_trajectory_change(self, e=None):
        """Handle changes to virtual trajectory parameters."""
        if self.virtual_trajectory_enabled:
            self._update_virtual_trajectory()

    def _update_virtual_trajectory_visibility(self):
        """Show/hide virtual trajectory controls based on whether frame is visible."""
        # Check if any visible surgical file has frame enabled
        has_frame_visible = False
        for file_info in self.loaded_files:
            if (file_info['type'] == 'surgical' and
                file_info.get('visible', True) and
                file_info.get('is_raw', False) and
                file_info.get('show_frame', False)):
                has_frame_visible = True
                break

        # Show/hide controls
        if self.virtual_trajectory_container:
            self.virtual_trajectory_container.set_visibility(has_frame_visible)

    def _update_virtual_trajectory(self):
        """Update the virtual trajectory in the plot without resetting camera."""
        if not self.current_figure or not self.plotly_element:
            return

        # Get parameters
        target_x = self.virtual_target_x.value if self.virtual_target_x else 100.0
        target_y = self.virtual_target_y.value if self.virtual_target_y else 100.0
        target_z = self.virtual_target_z.value if self.virtual_target_z else 100.0
        ring_angle = self.virtual_ring_slider.value if self.virtual_ring_slider else 90.0
        arc_angle = self.virtual_arc_slider.value if self.virtual_arc_slider else 90.0

        target = np.array([target_x, target_y, target_z])

        # Calculate trajectory direction
        direction, arc_axis = self._calculate_direction_from_angles(ring_angle, arc_angle)
        entry = target + direction * 100  # 100mm entry offset

        # Convert numpy arrays to lists for JSON serialization
        entry_list = entry.tolist()
        target_list = target.tolist()

        if self.virtual_trajectory_enabled:
            # Use JavaScript to add/update trace without resetting camera
            ui.run_javascript(f'''
                const plot = document.querySelector(".js-plotly-plot");
                if (plot && plot.data) {{
                    // Find existing virtual trajectory trace
                    let virtualIdx = -1;
                    for (let i = 0; i < plot.data.length; i++) {{
                        if (plot.data[i].name === 'Virtual Trajectory') {{
                            virtualIdx = i;
                            break;
                        }}
                    }}

                    if (virtualIdx >= 0) {{
                        // Update existing trace using restyle (preserves camera)
                        Plotly.restyle(plot, {{
                            'x': [[{entry_list[0]}, {target_list[0]}]],
                            'y': [[{entry_list[1]}, {target_list[1]}]],
                            'z': [[{entry_list[2]}, {target_list[2]}]]
                        }}, virtualIdx);
                    }} else {{
                        // Add new trace using addTraces (preserves camera)
                        const newTrace = {{
                            x: [{entry_list[0]}, {target_list[0]}],
                            y: [{entry_list[1]}, {target_list[1]}],
                            z: [{entry_list[2]}, {target_list[2]}],
                            mode: 'lines+markers',
                            type: 'scatter3d',
                            name: 'Virtual Trajectory',
                            line: {{width: 8, color: 'yellow', dash: 'dash'}},
                            marker: {{size: [6, 10], color: 'yellow', symbol: ['diamond', 'circle']}},
                            showlegend: true
                        }};
                        Plotly.addTraces(plot, newTrace);
                    }}
                }}
            ''')
        else:
            # Remove virtual trajectory trace if it exists
            ui.run_javascript('''
                const plot = document.querySelector(".js-plotly-plot");
                if (plot && plot.data) {
                    // Find existing virtual trajectory trace
                    let virtualIdx = -1;
                    for (let i = 0; i < plot.data.length; i++) {
                        if (plot.data[i].name === 'Virtual Trajectory') {
                            virtualIdx = i;
                            break;
                        }
                    }

                    if (virtualIdx >= 0) {
                        // Remove trace using deleteTraces (preserves camera)
                        Plotly.deleteTraces(plot, virtualIdx);
                    }
                }
            ''')

    def _toggle_file_visibility(self, file_info: dict):
        """Toggle visibility of a file in the plot."""
        if file_info in self.loaded_files:
            # Toggle visibility
            file_info['visible'] = not file_info['visible']
            status = 'visible' if file_info['visible'] else 'hidden'
            notify_info(f'{file_info["filename"]} is now {status}')

            # Update displays
            self._update_file_list()
            self._update_plot()
            self._update_virtual_trajectory_visibility()

    def _remove_file(self, file_info: dict):
        """Remove a file from the loaded files list."""
        if file_info in self.loaded_files:
            self.loaded_files.remove(file_info)
            notify_info(f'Removed {file_info["filename"]}')

            # Clear cache entries for this file
            filename = file_info['filename']
            keys_to_remove = [key for key in self.mesh_cache.keys() if key[0] == filename]
            for key in keys_to_remove:
                del self.mesh_cache[key]

            # Clear labels if this is a NIfTI file
            if filename in self.nifti_labels:
                del self.nifti_labels[filename]

                # Also remove the label file from loaded_files
                label_files = [f for f in self.loaded_files if f['type'] == 'labels' and f.get('nifti_target') == filename]
                for label_file in label_files:
                    self.loaded_files.remove(label_file)

            # Reset the uploader to allow re-uploading the same file
            self.upload_element.reset()

            # Update displays
            self._update_file_list()
            self._update_status()
            self._update_mesh_controls_visibility()
            self._update_plot()
            self._update_virtual_trajectory_visibility()

    def _remove_labels(self, file_info: dict):
        """Remove labels associated with a NIfTI file."""
        filename = file_info['filename']
        if filename in self.nifti_labels:
            del self.nifti_labels[filename]

            # Also remove the label file from loaded_files
            label_files = [f for f in self.loaded_files if f['type'] == 'labels' and f.get('nifti_target') == filename]
            for label_file in label_files:
                self.loaded_files.remove(label_file)

            # Reset the uploader to allow re-uploading the same label file
            self.upload_element.reset()

            notify_info(f'Removed labels for {filename}')

            # Update displays
            self._update_file_list()
            self._update_plot()

    def _clear_all_files(self):
        """Clear all loaded files."""
        self.loaded_files.clear()
        self.mesh_cache.clear()  # Clear all cached meshes
        self.nifti_labels.clear()  # Clear all labels

        # Reset the uploader to allow re-uploading files
        self.upload_element.reset()

        notify_info('Cleared all files')

        # Update displays
        self._update_file_list()
        self._update_status()
        self._update_mesh_controls_visibility()
        self._update_plot()
        self._update_virtual_trajectory_visibility()

    def _on_opacity_change(self, e):
        """Handle mesh opacity slider change (triggered on mouse release)."""
        self.mesh_opacity = e.value if hasattr(e, 'value') else e.args

        # Update opacity using JavaScript to preserve camera position
        ui.run_javascript(f'''
            const plot = document.querySelector(".js-plotly-plot");
            if (plot && plot.data) {{
                const updates = {{}};
                const indices = [];

                // Find all Mesh3d traces (NIfTI meshes)
                for (let i = 0; i < plot.data.length; i++) {{
                    if (plot.data[i].type === 'mesh3d') {{
                        indices.push(i);
                    }}
                }}

                // Update opacity without resetting camera
                if (indices.length > 0) {{
                    updates.opacity = {self.mesh_opacity};
                    Plotly.restyle(plot, updates, indices);
                }}
            }}
        ''')

    def _on_nifti_smoothness_change(self, file_info: dict, value: float):
        """Handle per-file smoothness slider change."""
        file_info['smoothness'] = value
        # Note: Meshes are NOT automatically regenerated
        # User must click "Generate Mesh" button to apply new smoothness

    def _on_nifti_threshold_change(self, file_info: dict, value: float):
        """Handle per-file threshold slider change."""
        file_info['threshold'] = value
        # Note: Meshes are NOT automatically regenerated
        # User must click "Generate Mesh" button to apply new threshold

    def _on_generate_mesh_for_file(self, file_info: dict):
        """Generate mesh for a specific NIfTI file."""
        asyncio.create_task(self._update_plot_async())

    async def _toggle_plot_maximize(self):
        """Toggle maximize state of the 3D plot."""
        self.plot_maximized = not self.plot_maximized

        if self.plot_maximized:
            # Maximize: Set plot card to fixed position covering viewport (with padding for header)
            self.plot_card.classes(remove='flex-1', add='fixed left-0 right-0 bottom-0 z-50')
            self.plot_card.style('top: 104px')
            self.maximize_btn.props('icon=fullscreen_exit')

            # Remove explicit height to allow plot to expand, then trigger resize
            if self.plotly_element:
                self.plotly_element.style(remove='height: 600px')
                await asyncio.sleep(0.1)
                ui.run_javascript('''
                    setTimeout(() => {
                        window.dispatchEvent(new Event('resize'));
                    }, 100);
                ''')
        else:
            # Restore: Return to normal layout
            self.plot_card.classes(remove='fixed left-0 right-0 bottom-0 z-50', add='flex-1')
            self.plot_card.style(remove='top: 104px')
            self.maximize_btn.props('icon=fullscreen')

            # Set explicit height on plotly element when restoring
            if self.plotly_element:
                self.plotly_element.style('height: 600px')
                await asyncio.sleep(0.1)
                ui.run_javascript('''
                    console.log("Minimize: Starting resize");
                    setTimeout(() => {
                        const plot = document.querySelector(".js-plotly-plot");
                        console.log("Minimize: Plot element:", plot);
                        if (plot && plot._fullData) {
                            console.log("Minimize: Plot has data, relayouting");
                            // Get actual container dimensions
                            const width = plot.offsetWidth;
                            console.log("Minimize: Container width:", width);
                            Plotly.relayout(plot, {
                                width: width,
                                height: 600
                            }).then(() => {
                                console.log("Minimize: Relayout complete, dispatching resize");
                                // Dispatch resize events to ensure proper recalculation
                                window.dispatchEvent(new Event('resize'));
                                setTimeout(() => {
                                    console.log("Minimize: Second resize dispatch");
                                    window.dispatchEvent(new Event('resize'));
                                }, 150);
                            });
                        } else {
                            console.log("Minimize: Plot not found or no data");
                        }
                    }, 100);
                ''')

    def _on_generate_meshes(self):
        """Handle generate meshes button click."""
        asyncio.create_task(self._update_plot_async())

    async def _update_plot_async(self):
        """Update the 3D plot based on loaded data (async version)."""
        self.plot_container.clear()

        with self.plot_container:
            if not self.loaded_files:
                # Show placeholder
                with ui.column().classes('w-full h-full items-center justify-center'):
                    ui.icon('3d_rotation', size='xl').classes('text-grey-4')
                    ui.label('No data to display').classes('text-body2 text-grey-6 mt-2')
                    ui.label('Upload files to visualize').classes('text-caption text-grey-5')
                self.plotly_element = None
                self.current_figure = None
            else:
                # Check if we have any NIfTI files that need processing
                has_nifti = any(f['type'] == 'nifti' for f in self.loaded_files)

                if has_nifti:
                    # Show loading indicator
                    with ui.column().classes('w-full h-full items-center justify-center'):
                        ui.spinner(size='lg', color='primary')
                        ui.label('Processing mesh surfaces...').classes('text-body2 mt-2')

                # Create combined plot (with async mesh extraction)
                fig = await self._create_combined_plot_async()

                # Clear and show the plot
                self.plot_container.clear()
                with self.plot_container:
                    self.current_figure = fig
                    # Display plot and store reference
                    self.plotly_element = ui.plotly(fig).classes('w-full h-full')

                    # Re-add virtual trajectory if it was enabled
                    if self.virtual_trajectory_enabled:
                        self._update_virtual_trajectory()

    def _update_plot(self):
        """Update the 3D plot based on loaded data (uses cached meshes only)."""
        self.plot_container.clear()

        with self.plot_container:
            if not self.loaded_files:
                # Show placeholder
                with ui.column().classes('w-full h-full items-center justify-center'):
                    ui.icon('3d_rotation', size='xl').classes('text-grey-4')
                    ui.label('No data to display').classes('text-body2 text-grey-6 mt-2')
                    ui.label('Upload files to visualize').classes('text-caption text-grey-5')
                self.plotly_element = None
                self.current_figure = None
            else:
                # Create combined plot (sync - uses cached meshes only, no extraction)
                fig = self._create_combined_plot()
                self.current_figure = fig

                # Display plot and store reference
                self.plotly_element = ui.plotly(fig).classes('w-full h-full')

                # Re-add virtual trajectory if it was enabled
                if self.virtual_trajectory_enabled:
                    self._update_virtual_trajectory()

    def _create_combined_plot(self):
        """Create a combined 3D plot with all loaded files."""
        fig = go.Figure()

        trace_offset = 0  # Color offset for different files

        # Process each loaded file (only visible files)
        for file_info in self.loaded_files:
            # Skip if file is hidden
            if not file_info.get('visible', True):
                continue

            if file_info['type'] == 'surgical':
                trace_offset = self._add_surgical_traces(fig, file_info, trace_offset)
            elif file_info['type'] == 'electrode':
                trace_offset = self._add_electrode_traces(fig, file_info, trace_offset)
            elif file_info['type'] == 'fiducials':
                trace_offset = self._add_fiducial_traces(fig, file_info['data'], trace_offset)
            elif file_info['type'] == 'nifti':
                trace_offset = self._add_nifti_mesh_traces(fig, file_info, trace_offset)

        # Calculate isotropic axis ranges
        axis_ranges = self._calculate_isotropic_ranges(fig)

        # Update layout with dark mode and orbital controls
        fig.update_layout(
            title='3D Visualization',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                dragmode='orbit',  # Use orbital controls instead of turntable
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
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#fff')
        )

        return fig

    def _calculate_isotropic_ranges(self, fig: go.Figure) -> dict:
        """
        Calculate isotropic axis ranges based on the largest axis extent.
        Ensures all axes have the same scale.

        Args:
            fig: Plotly figure with traces

        Returns:
            Dictionary with 'x', 'y', 'z' keys containing [min, max] ranges
        """
        # Collect all coordinates from all traces
        all_x, all_y, all_z = [], [], []

        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                all_x.extend([v for v in trace.x if v is not None])
            if hasattr(trace, 'y') and trace.y is not None:
                all_y.extend([v for v in trace.y if v is not None])
            if hasattr(trace, 'z') and trace.z is not None:
                all_z.extend([v for v in trace.z if v is not None])

        # If no data, return default ranges
        if not all_x or not all_y or not all_z:
            return {'x': [-10, 10], 'y': [-10, 10], 'z': [-10, 10]}

        # Calculate min/max for each axis
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)

        # Calculate centers and ranges
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Find the maximum range across all axes
        max_range = max(x_range, y_range, z_range)

        # Add 10% padding
        max_range *= 1.1
        half_range = max_range / 2

        # Set all axes to the same range, centered on their respective data
        return {
            'x': [x_center - half_range, x_center + half_range],
            'y': [y_center - half_range, y_center + half_range],
            'z': [z_center - half_range, z_center + half_range]
        }

    async def _create_combined_plot_async(self):
        """Create a combined 3D plot with all loaded files (async version with background mesh processing)."""
        fig = go.Figure()

        trace_offset = 0  # Color offset for different files

        # Process each loaded file (only visible files)
        for file_info in self.loaded_files:
            # Skip if file is hidden
            if not file_info.get('visible', True):
                continue

            if file_info['type'] == 'surgical':
                trace_offset = self._add_surgical_traces(fig, file_info, trace_offset)
            elif file_info['type'] == 'electrode':
                trace_offset = self._add_electrode_traces(fig, file_info, trace_offset)
            elif file_info['type'] == 'fiducials':
                trace_offset = self._add_fiducial_traces(fig, file_info['data'], trace_offset)
            elif file_info['type'] == 'nifti':
                # Run mesh extraction in background thread
                trace_offset = await self._add_nifti_mesh_traces_async(fig, file_info, trace_offset)

        # Calculate isotropic axis ranges
        axis_ranges = self._calculate_isotropic_ranges(fig)

        # Update layout with dark mode and orbital controls
        fig.update_layout(
            title='3D Visualization',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                dragmode='orbit',  # Use orbital controls instead of turntable
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
            height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='#fff')
        )

        return fig

    def _add_surgical_traces(self, fig: go.Figure, file_info: dict, color_offset: int = 0) -> int:
        """Add surgical targeting traces to the figure."""
        data = file_info['data']

        # Get toggle states
        show_mer_tracks = file_info.get('show_mer_tracks', True)
        show_trajectory = file_info.get('show_trajectory', True)
        show_targets = file_info.get('show_targets', True)

        # Process each surgical target
        for idx, row in enumerate(data):
            try:
                # Extract target coordinates
                x = float(row.get('x', 0))
                y = float(row.get('y', 0))
                z = float(row.get('z', 0))

                target = np.array([x, y, z])

                # Get ring and arc angles (prefer original values for display if available)
                ring = float(row.get('ring_original', row.get('ring', 0)))
                arc = float(row.get('arc_original', row.get('arc', 0)))

                # Check if virtual entry point is provided (from transformed data)
                if 'entry_x' in row and 'entry_y' in row and 'entry_z' in row:
                    # Use pre-calculated transformed entry point
                    entry_x = float(row.get('entry_x'))
                    entry_y = float(row.get('entry_y'))
                    entry_z = float(row.get('entry_z'))
                    entry = np.array([entry_x, entry_y, entry_z])
                    using_virtual_entry = True
                else:
                    # Calculate entry from ring/arc angles (for untransformed data)
                    direction, arc_axis = self._calculate_direction_from_angles(ring, arc)
                    entry = target + direction * 100  # 100mm entry offset
                    using_virtual_entry = False

                # Create trajectory line
                trajectory_x = [entry[0], target[0]]
                trajectory_y = [entry[1], target[1]]
                trajectory_z = [entry[2], target[2]]

                # Add trajectory line
                patient_id = row.get('patient_id', f'Target {idx+1}')
                hemisphere = row.get('hemisphere', '')
                target_name = row.get('anatomical_target', '')

                label = f"{patient_id} {hemisphere} {target_name}".strip()

                color_idx = (idx + color_offset) % 360
                color = f'hsl({color_idx * 60}, 70%, 50%)'

                # Add trajectory line (if enabled)
                if show_trajectory:
                    fig.add_trace(go.Scatter3d(
                        x=trajectory_x,
                        y=trajectory_y,
                        z=trajectory_z,
                        mode='lines',
                        name=label,
                        line=dict(width=6, color=color),
                        showlegend=True
                    ))

                    # Build hover text for target
                    target_hover = f'Target: [{x:.1f}, {y:.1f}, {z:.1f}]'
                    if using_virtual_entry:
                        # Show it's transformed data
                        target_hover += '<br>(Transformed)'
                        if 'x_original' in row:
                            x_orig = float(row['x_original'])
                            y_orig = float(row['y_original'])
                            z_orig = float(row['z_original'])
                            target_hover += f'<br>Original: [{x_orig:.1f}, {y_orig:.1f}, {z_orig:.1f}]'
                    # Always show original surgical plan angles
                    target_hover += f'<br>Ring: {ring:.1f}°<br>Arc: {arc:.1f}°'

                    # Add target point (circle for better visibility)
                    fig.add_trace(go.Scatter3d(
                        x=[target[0]],
                        y=[target[1]],
                        z=[target[2]],
                        mode='markers',
                        name=f'{label} Target',
                        marker=dict(
                            size=10,
                            color=color,
                            symbol='circle',
                            line=dict(color='black', width=2)
                        ),
                        text=[target_hover],
                        hoverinfo='text',
                        showlegend=False
                    ))

                    # Build hover text for entry
                    entry_hover = f'Entry: [{entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f}]'
                    if using_virtual_entry:
                        entry_hover += '<br>(Transformed)'

                    # Add entry point (diamond, smaller)
                    fig.add_trace(go.Scatter3d(
                        x=[entry[0]],
                        y=[entry[1]],
                        z=[entry[2]],
                        mode='markers',
                        name=f'{label} Entry',
                        marker=dict(
                            size=8,
                            color=color,
                            symbol='diamond',
                            line=dict(color='black', width=1)
                        ),
                        text=[entry_hover],
                        hoverinfo='text',
                        showlegend=False
                    ))

                # Add clinical MER target if available (and toggle is on)
                if show_targets:
                    clinical_x = None
                    clinical_y = None
                    clinical_z = None
                    clinical_track = row.get('clinical_track', '')
                    clinical_depth = row.get('clinical_depth', '')

                    # Check if transformed data exists
                    if 'clinical_target_x' in row and 'clinical_target_y' in row and 'clinical_target_z' in row:
                        clinical_x = float(row['clinical_target_x'])
                        clinical_y = float(row['clinical_target_y'])
                        clinical_z = float(row['clinical_target_z'])
                    # Otherwise calculate from raw data
                    elif clinical_track and clinical_depth:
                        try:
                            from dbstoolbox.utils.transform_coordinates import calculate_mer_track_position
                            depth = float(clinical_depth)
                            clinical_mer_target, _ = calculate_mer_track_position(
                                target, ring, arc, clinical_track, depth
                            )
                            clinical_x, clinical_y, clinical_z = clinical_mer_target[0], clinical_mer_target[1], clinical_mer_target[2]
                        except:
                            pass

                    if clinical_x is not None:
                        clinical_hover = f'Clinical MER<br>Track: {clinical_track}<br>Depth: {clinical_depth}mm<br>[{clinical_x:.1f}, {clinical_y:.1f}, {clinical_z:.1f}]'

                        fig.add_trace(go.Scatter3d(
                            x=[clinical_x],
                            y=[clinical_y],
                            z=[clinical_z],
                            mode='markers',
                            name=f'{label} Clinical',
                            marker=dict(
                                size=10,
                                color='cyan',
                                symbol='square',
                                line=dict(color='black', width=2)
                            ),
                            text=[clinical_hover],
                            hoverinfo='text',
                            showlegend=False
                        ))

                # Add research MER target if available (and toggle is on)
                if show_targets:
                    research_x = None
                    research_y = None
                    research_z = None
                    research_track = row.get('research_track', '')
                    research_depth = row.get('research_depth', '')

                    # Check if transformed data exists
                    if 'research_target_x' in row and 'research_target_y' in row and 'research_target_z' in row:
                        research_x = float(row['research_target_x'])
                        research_y = float(row['research_target_y'])
                        research_z = float(row['research_target_z'])
                    # Otherwise calculate from raw data
                    elif research_track and research_depth:
                        try:
                            from dbstoolbox.utils.transform_coordinates import calculate_mer_track_position
                            depth = float(research_depth)
                            research_mer_target, _ = calculate_mer_track_position(
                                target, ring, arc, research_track, depth
                            )
                            research_x, research_y, research_z = research_mer_target[0], research_mer_target[1], research_mer_target[2]
                        except:
                            pass

                    if research_x is not None:
                        research_hover = f'Research MER<br>Track: {research_track}<br>Depth: {research_depth}mm<br>[{research_x:.1f}, {research_y:.1f}, {research_z:.1f}]'

                        fig.add_trace(go.Scatter3d(
                            x=[research_x],
                            y=[research_y],
                            z=[research_z],
                            mode='markers',
                            name=f'{label} Research',
                            marker=dict(
                                size=10,
                                color='magenta',
                                symbol='square',
                                line=dict(color='black', width=2)
                            ),
                            text=[research_hover],
                            hoverinfo='text',
                            showlegend=False
                        ))

                # Add MER track trajectories (if toggle is on)
                if show_mer_tracks:
                    # Find all mer_*_target_x fields to identify track types
                    mer_track_types = set()
                    for key in row.keys():
                        if key.startswith('mer_') and key.endswith('_target_x'):
                            # Extract track type from key (e.g., 'mer_lateral_target_x' -> 'lateral')
                            track_type = key.replace('mer_', '').replace('_target_x', '')
                            mer_track_types.add(track_type)

                    # If no MER tracks in data (raw data), calculate them from ring/arc
                    if not mer_track_types:
                        # Use all standard track types for raw data
                        mer_track_types = {'anterior', 'posterior', 'medial', 'lateral', 'central'}

                    # Define colors for different MER track types
                    mer_track_colors = {
                        'anterior': 'rgba(255, 100, 100, 0.6)',    # Light red
                        'posterior': 'rgba(100, 100, 255, 0.6)',   # Light blue
                        'medial': 'rgba(100, 255, 100, 0.6)',      # Light green
                        'lateral': 'rgba(255, 255, 100, 0.6)',     # Light yellow
                        'central': 'rgba(200, 200, 200, 0.6)'      # Light gray
                    }

                    # Visualize each MER track
                    for track_type in mer_track_types:
                        try:
                            # Check if transformed MER track data exists
                            if f'mer_{track_type}_target_x' in row:
                                # Use pre-calculated transformed MER track data
                                mer_target_x = float(row[f'mer_{track_type}_target_x'])
                                mer_target_y = float(row[f'mer_{track_type}_target_y'])
                                mer_target_z = float(row[f'mer_{track_type}_target_z'])
                                mer_entry_x = float(row[f'mer_{track_type}_entry_x'])
                                mer_entry_y = float(row[f'mer_{track_type}_entry_y'])
                                mer_entry_z = float(row[f'mer_{track_type}_entry_z'])
                            else:
                                # Calculate MER track from ring/arc angles (raw data)
                                from dbstoolbox.utils.transform_coordinates import (
                                    calculate_direction_from_angles,
                                    calculate_parallel_track_offset
                                )

                                direction, arc_axis_mer = calculate_direction_from_angles(ring, arc)
                                perpendicular_offset = calculate_parallel_track_offset(
                                    track_type, target[0], direction, arc_axis_mer
                                )

                                # Calculate MER track target (5mm below central target)
                                mer_target = target + perpendicular_offset - direction * 5.0
                                mer_entry = mer_target + direction * 100.0

                                mer_target_x, mer_target_y, mer_target_z = mer_target[0], mer_target[1], mer_target[2]
                                mer_entry_x, mer_entry_y, mer_entry_z = mer_entry[0], mer_entry[1], mer_entry[2]

                            # Calculate position at central target depth (depth=0)
                            # The track extends 5mm below target, so we move 5mm back along the trajectory
                            direction_vec = np.array([mer_entry_x - mer_target_x,
                                                     mer_entry_y - mer_target_y,
                                                     mer_entry_z - mer_target_z])
                            direction_norm = direction_vec / np.linalg.norm(direction_vec)

                            # Position at central target depth (5mm back from end point)
                            mer_target_depth0 = np.array([mer_target_x, mer_target_y, mer_target_z]) + direction_norm * 5.0

                            # Create trajectory line
                            mer_trajectory_x = [mer_entry_x, mer_target_x]
                            mer_trajectory_y = [mer_entry_y, mer_target_y]
                            mer_trajectory_z = [mer_entry_z, mer_target_z]

                            # Get color for this track type
                            mer_color = mer_track_colors.get(track_type, 'rgba(150, 150, 150, 0.6)')

                            # Add MER trajectory line
                            fig.add_trace(go.Scatter3d(
                                x=mer_trajectory_x,
                                y=mer_trajectory_y,
                                z=mer_trajectory_z,
                                mode='lines',
                                name=f'{label} MER {track_type.capitalize()}',
                                line=dict(width=3, color=mer_color, dash='dot'),
                                showlegend=False,
                                hoverinfo='text',
                                text=[f'MER Track: {track_type}']
                            ))

                            # Add MER marker at central target depth (depth=0)
                            fig.add_trace(go.Scatter3d(
                                x=[mer_target_depth0[0]],
                                y=[mer_target_depth0[1]],
                                z=[mer_target_depth0[2]],
                                mode='markers',
                                name=f'{label} MER {track_type} Target',
                                marker=dict(
                                    size=6,
                                    color=mer_color,
                                    symbol='circle',
                                    line=dict(color='black', width=0.5)
                                ),
                                text=[f'MER {track_type} Track<br>At central target depth<br>[{mer_target_depth0[0]:.1f}, {mer_target_depth0[1]:.1f}, {mer_target_depth0[2]:.1f}]'],
                                hoverinfo='text',
                                showlegend=False
                            ))

                            # Add MER entry marker
                            fig.add_trace(go.Scatter3d(
                                x=[mer_entry_x],
                                y=[mer_entry_y],
                                z=[mer_entry_z],
                                mode='markers',
                                name=f'{label} MER {track_type} Entry',
                                marker=dict(
                                    size=4,
                                    color=mer_color,
                                    symbol='circle-open',
                                    line=dict(color='black', width=0.5)
                                ),
                                text=[f'MER {track_type} Entry<br>[{mer_entry_x:.1f}, {mer_entry_y:.1f}, {mer_entry_z:.1f}]'],
                                hoverinfo='text',
                                showlegend=False
                            ))

                        except (ValueError, KeyError):
                            # Skip if coordinates are missing or invalid for this track
                            continue

            except (ValueError, KeyError) as e:
                print(f"Skipping row {idx}: {e}")
                continue

        # Add Leksell frame visualization (if enabled and data is raw)
        show_frame = file_info.get('show_frame', False)
        is_raw = file_info.get('is_raw', False)

        if show_frame and is_raw:
            # Leksell frame configuration with 6 fiducials
            # Coordinate system: X increases LEFT (center=100), Y increases FRONT, Z increases DOWN
            # Left side (X=195): 3 fiducials
            # Right side (X=5): 3 fiducials
            frame_fiducials = {
                'left_posterior': {'start': [195.0, 40.0, 40.0], 'end': [195.0, 40.0, 160.0]},
                'left_oblique': {'start': [195.0, 40.0, 40.0], 'end': [195.0, 160.0, 160.0]},
                'left_anterior': {'start': [195.0, 160.0, 40.0], 'end': [195.0, 160.0, 160.0]},
                'right_posterior': {'start': [5.0, 40.0, 40.0], 'end': [5.0, 40.0, 160.0]},
                'right_oblique': {'start': [5.0, 40.0, 40.0], 'end': [5.0, 160.0, 160.0]},
                'right_anterior': {'start': [5.0, 160.0, 40.0], 'end': [5.0, 160.0, 160.0]}
            }

            # Add each fiducial line
            for fiducial_name, coords in frame_fiducials.items():
                start = np.array(coords['start'])
                end = np.array(coords['end'])

                # Add fiducial line (semi-transparent white)
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    name='Leksell Frame' if fiducial_name == 'left_posterior' else None,
                    line=dict(width=4, color='rgba(255, 255, 255, 0.3)'),
                    legendgroup='leksell_frame',
                    showlegend=(fiducial_name == 'left_posterior'),
                    hoverinfo='text',
                    text=[f'Frame: {fiducial_name.replace("_", " ").title()}']
                ))

        # Return updated color offset
        return color_offset + len(data)

    def _add_electrode_traces(self, fig: go.Figure, file_info: dict, color_offset: int = 0) -> int:
        """Add electrode reconstruction traces to the figure."""
        data = file_info['data']
        electrodes = data.get('electrodes', [])

        # Get toggle state
        show_trajectory = file_info.get('show_trajectory', True)

        # Process each electrode
        for idx, electrode in enumerate(electrodes):
            color_idx = (idx + color_offset) % 360
            color = f'hsl({color_idx * 60}, 70%, 50%)'

            # Get trajectory coordinates
            if show_trajectory and 'trajectory_coordinates' in electrode:
                trajectory = np.array(electrode['trajectory_coordinates'])

                # Add trajectory line
                fig.add_trace(go.Scatter3d(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    z=trajectory[:, 2],
                    mode='lines',
                    name=f'Electrode {idx+1}',
                    line=dict(width=6, color=color),
                    showlegend=True
                ))

            # Get contact positions
            if 'contact_positions_3d' in electrode:
                contacts = np.array(electrode['contact_positions_3d'])

                # Add contact points
                hover_text = [
                    f'E{idx+1} C{i+1}<br>[{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}]'
                    for i, c in enumerate(contacts)
                ]

                fig.add_trace(go.Scatter3d(
                    x=contacts[:, 0],
                    y=contacts[:, 1],
                    z=contacts[:, 2],
                    mode='markers',
                    name=f'E{idx+1} Contacts',
                    marker=dict(
                        size=8,
                        color=color,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    showlegend=False
                ))

        # Return updated color offset
        return color_offset + len(electrodes)

    def _add_fiducial_traces(self, fig: go.Figure, data: dict, color_offset: int = 0) -> int:
        """Add stereotactic frame fiducial traces to the figure."""
        fiducial_rods = data.get('fiducial_rods', [])

        # Process each fiducial rod
        for idx, rod in enumerate(fiducial_rods):
            # Skip if doesn't have required fields
            if 'bottom_point' not in rod or 'top_point' not in rod:
                continue

            bottom = np.array(rod['bottom_point'])
            top = np.array(rod['top_point'])

            # Validate they are 3D points
            if len(bottom) != 3 or len(top) != 3:
                continue

            # Create color for this fiducial
            color_idx = (idx + color_offset) % 360
            color = f'hsl({color_idx * 60}, 70%, 50%)'

            # Add fiducial rod line
            # Only show legend for the first fiducial, but group all together
            fig.add_trace(go.Scatter3d(
                x=[bottom[0], top[0]],
                y=[bottom[1], top[1]],
                z=[bottom[2], top[2]],
                mode='lines',
                name='Frame Fiducials',
                line=dict(width=8, color=color),
                legendgroup='fiducials',
                showlegend=(idx == 0)
            ))

            # Add bottom point marker
            fig.add_trace(go.Scatter3d(
                x=[bottom[0]],
                y=[bottom[1]],
                z=[bottom[2]],
                mode='markers',
                name='Frame Fiducials',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=[f'Fiducial {idx+1} Bottom: [{bottom[0]:.1f}, {bottom[1]:.1f}, {bottom[2]:.1f}]'],
                hoverinfo='text',
                legendgroup='fiducials',
                showlegend=False
            ))

            # Add top point marker
            fig.add_trace(go.Scatter3d(
                x=[top[0]],
                y=[top[1]],
                z=[top[2]],
                mode='markers',
                name='Frame Fiducials',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                text=[f'Fiducial {idx+1} Top: [{top[0]:.1f}, {top[1]:.1f}, {top[2]:.1f}]'],
                hoverinfo='text',
                legendgroup='fiducials',
                showlegend=False
            ))

        # Return updated color offset
        return color_offset + len(fiducial_rods)

    def _add_nifti_mesh_traces(self, fig: go.Figure, file_info: dict, color_offset: int = 0) -> int:
        """
        Add NIfTI surface mesh traces to the figure using marching cubes.

        Args:
            fig: Plotly figure
            file_info: File info dictionary containing data and metadata
            color_offset: Color offset for this file

        Returns:
            Updated color offset
        """
        data = file_info['data']
        affine = file_info['affine']
        metadata = file_info['metadata']
        filename = file_info['filename']

        # Add bounding box to show data extent (always visible)
        self._add_nifti_bounding_box(fig, data, affine, filename)

        # Check if 3D or 4D
        if metadata['dimensions'] == 4:
            # 4D: Extract surface for each volume
            num_volumes = metadata['num_volumes']

            for vol_idx in range(num_volumes):
                # Extract volume
                volume = data[:, :, :, vol_idx]

                # Only use cached meshes (don't extract automatically)
                cache_key = (filename, self.mesh_smoothness, vol_idx)
                if cache_key in self.mesh_cache:
                    verts, faces = self.mesh_cache[cache_key]
                else:
                    # Skip - user must click "Generate Meshes" to extract
                    verts, faces = None, None

                if verts is not None and len(verts) > 0:
                    # Color based on volume index
                    color_idx = (vol_idx + color_offset) % 12
                    colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                        '#F8B739', '#52B788', '#E76F51', '#A8DADC'
                    ]
                    color = colors[color_idx]

                    # Add mesh trace
                    # Check if we have a label for this volume
                    if hasattr(self, 'nifti_labels') and self.nifti_labels and filename in self.nifti_labels:
                        label = self.nifti_labels[filename].get(vol_idx, f'nifti volume {vol_idx + 1}')
                    else:
                        label = f'nifti volume {vol_idx + 1}'

                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=self.mesh_opacity,
                        name=label,
                        showlegend=True,
                        hoverinfo='name'
                    ))

            return color_offset + num_volumes

        else:
            # 3D: Single volume
            volume = data

            # Only use cached meshes (don't extract automatically)
            cache_key = (filename, self.mesh_smoothness)
            if cache_key in self.mesh_cache:
                verts, faces = self.mesh_cache[cache_key]
            else:
                # Skip - user must click "Generate Meshes" to extract
                verts, faces = None, None

            if verts is not None and len(verts) > 0:
                # Single color
                color_idx = color_offset % 12
                colors = [
                    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
                    '#F8B739', '#52B788', '#E76F51', '#A8DADC'
                ]
                color = colors[color_idx]

                # Add mesh trace
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=self.mesh_opacity,
                    name=filename,
                    showlegend=True,
                    hoverinfo='name'
                ))

            return color_offset + 1

    async def _add_nifti_mesh_traces_async(self, fig: go.Figure, file_info: dict, color_offset: int = 0) -> int:
        """
        Add NIfTI surface mesh traces to the figure using marching cubes (async version).
        Runs mesh extraction in a background thread to avoid blocking the UI.

        Args:
            fig: Plotly figure
            file_info: File info dictionary containing data and metadata
            color_offset: Color offset for this file

        Returns:
            Updated color offset
        """
        data = file_info['data']
        affine = file_info['affine']
        metadata = file_info['metadata']
        filename = file_info['filename']

        # Get per-file settings
        smoothness = file_info.get('smoothness', 0.5)
        threshold = file_info.get('threshold', 0.5)

        # Add bounding box to show data extent (always visible)
        self._add_nifti_bounding_box(fig, data, affine, filename)

        # Color palette
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
            '#F8B739', '#52B788', '#E76F51', '#A8DADC'
        ]

        # Check if 3D or 4D
        if metadata['dimensions'] == 4:
            # 4D: Extract surface for each volume
            num_volumes = metadata['num_volumes']
            print(f"[Mesh Generation] Processing 4D NIfTI with {num_volumes} volumes")

            for vol_idx in range(num_volumes):
                print(f"[Mesh Generation] Processing volume {vol_idx + 1}/{num_volumes}")

                # Extract volume
                volume = data[:, :, :, vol_idx]

                # Check cache first (cache key includes smoothness and threshold)
                cache_key = (filename, smoothness, threshold, vol_idx)
                if cache_key in self.mesh_cache:
                    print(f"[Mesh Generation] Using cached mesh for volume {vol_idx + 1}")
                    verts, faces = self.mesh_cache[cache_key]
                else:
                    print(f"[Mesh Generation] Extracting mesh for volume {vol_idx + 1}")
                    # Run marching cubes in background thread
                    verts, faces = await asyncio.to_thread(
                        self._extract_surface_marching_cubes,
                        volume, affine, threshold=threshold, smoothness=smoothness
                    )
                    # Cache the result
                    if verts is not None:
                        self.mesh_cache[cache_key] = (verts, faces)
                        print(f"[Mesh Generation] Cached mesh for volume {vol_idx + 1}")

                if verts is not None and len(verts) > 0:
                    print(f"[Mesh Generation] Adding trace for volume {vol_idx + 1} with {len(verts)} vertices")
                    # Color based on volume index
                    color_idx = (vol_idx + color_offset) % 12
                    color = colors[color_idx]

                    # Add mesh trace
                    # Check if we have a label for this volume
                    if hasattr(self, 'nifti_labels') and self.nifti_labels and filename in self.nifti_labels:
                        label = self.nifti_labels[filename].get(vol_idx, f'nifti volume {vol_idx + 1}')
                    else:
                        label = f'nifti volume {vol_idx + 1}'

                    fig.add_trace(go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=self.mesh_opacity,
                        name=label,
                        showlegend=True,
                        hoverinfo='name'
                    ))
                else:
                    if verts is None:
                        print(f"[Mesh Generation] WARNING: No mesh extracted for volume {vol_idx + 1} (verts is None)")
                    else:
                        print(f"[Mesh Generation] WARNING: No vertices for volume {vol_idx + 1} (empty mesh)")

            return color_offset + num_volumes

        else:
            # 3D: Single volume
            volume = data

            # Check cache first (cache key includes smoothness and threshold)
            cache_key = (filename, smoothness, threshold)
            if cache_key in self.mesh_cache:
                verts, faces = self.mesh_cache[cache_key]
            else:
                # Run marching cubes in background thread
                verts, faces = await asyncio.to_thread(
                    self._extract_surface_marching_cubes,
                    volume, affine, threshold=threshold, smoothness=smoothness
                )
                # Cache the result
                if verts is not None:
                    self.mesh_cache[cache_key] = (verts, faces)

            if verts is not None and len(verts) > 0:
                # Single color
                color_idx = color_offset % 12
                color = colors[color_idx]

                # Add mesh trace
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=self.mesh_opacity,
                    name=filename,
                    showlegend=True,
                    hoverinfo='name'
                ))

            return color_offset + 1

    def _add_nifti_bounding_box(self, fig: go.Figure, data: np.ndarray, affine: np.ndarray, filename: str):
        """
        Add a bounding box for NIfTI data to show the volume extent.

        Args:
            fig: Plotly figure
            data: 3D or 4D numpy array
            affine: 4x4 affine transformation matrix
            filename: Name of the NIfTI file
        """
        # Get volume shape (ignore 4th dimension if 4D)
        if data.ndim == 4:
            shape = data.shape[:3]
        else:
            shape = data.shape

        # Define 8 corners of the bounding box in voxel space
        # Corners are at (0,0,0) and (nx-1, ny-1, nz-1)
        corners_voxel = np.array([
            [0, 0, 0],
            [shape[0]-1, 0, 0],
            [shape[0]-1, shape[1]-1, 0],
            [0, shape[1]-1, 0],
            [0, 0, shape[2]-1],
            [shape[0]-1, 0, shape[2]-1],
            [shape[0]-1, shape[1]-1, shape[2]-1],
            [0, shape[1]-1, shape[2]-1]
        ])

        # Transform corners to world space
        corners_homogeneous = np.hstack([corners_voxel, np.ones((8, 1))])
        corners_world = (affine @ corners_homogeneous.T).T[:, :3]

        # Define edges connecting the corners (12 edges of a cube)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        # Create line segments for each edge
        x_lines, y_lines, z_lines = [], [], []
        for edge in edges:
            i, j = edge
            # Add the two points
            x_lines.extend([corners_world[i, 0], corners_world[j, 0], None])
            y_lines.extend([corners_world[i, 1], corners_world[j, 1], None])
            z_lines.extend([corners_world[i, 2], corners_world[j, 2], None])

        # Add bounding box as line trace
        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='cyan', width=3, dash='dash'),
            name=f'{filename} (bbox)',
            showlegend=True,
            hoverinfo='name'
        ))

    def _extract_surface_marching_cubes(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        threshold: float = 0.5,
        smoothness: float = 0.5
    ) -> tuple:
        """
        Extract surface mesh from 3D volume using marching cubes algorithm.

        Args:
            volume: 3D numpy array (probability map or segmentation)
            affine: 4x4 affine transformation matrix
            threshold: Threshold value for surface extraction (default 0.5)
            smoothness: Gaussian smoothing sigma (default 0.5)

        Returns:
            Tuple of (vertices, faces) or (None, None) if extraction fails
        """
        try:
            print(f"[Marching Cubes] Volume shape: {volume.shape}, data range: [{volume.min():.3f}, {volume.max():.3f}], threshold: {threshold}")

            # Apply Gaussian smoothing if smoothness > 0
            if smoothness > 0:
                volume = gaussian_filter(volume, sigma=smoothness)
                print(f"[Marching Cubes] After smoothing (sigma={smoothness}), data range: [{volume.min():.3f}, {volume.max():.3f}]")

            # Apply marching cubes algorithm in voxel space (no spacing)
            # We'll transform to world space using the full affine matrix afterwards
            verts, faces, normals, values = marching_cubes(
                volume,
                level=threshold,
                allow_degenerate=False
            )

            print(f"[Marching Cubes] Extracted {len(verts)} vertices, {len(faces)} faces")

            # Transform vertices from voxel coordinates to world coordinates
            # IMPORTANT: Marching cubes uses corner convention (integer = corner)
            # but NIfTI (nibabel) affine uses center convention (integer = center)
            # So we add 0.5 to shift from corner to center before applying affine
            verts_centered = verts + 0.5

            # Add homogeneous coordinate
            verts_homogeneous = np.hstack([verts_centered, np.ones((verts_centered.shape[0], 1))])

            # Apply affine transformation to convert voxel space -> world space
            verts_world = (affine @ verts_homogeneous.T).T[:, :3]

            print(f"[Marching Cubes] World coords range: X[{verts_world[:,0].min():.1f}, {verts_world[:,0].max():.1f}], Y[{verts_world[:,1].min():.1f}, {verts_world[:,1].max():.1f}], Z[{verts_world[:,2].min():.1f}, {verts_world[:,2].max():.1f}]")

            return verts_world, faces

        except Exception as e:
            print(f"[Marching Cubes] Failed to extract surface: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _calculate_direction_from_angles(self, ring_angle: float, arc_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate 3D direction vector from stereotactic angles using center-of-arc principle.

        Coordinate system (origin at UPPER, POSTERIOR, RIGHT corner):
        - X-axis increases LEFT, Y-axis increases FRONT, Z-axis increases DOWN

        Process:
        1. Base direction when arc=0°: (-1, 0, 0) pointing RIGHT
        2. Arc rotation axis rotates with ring angle around X-axis
        3. Rotate base direction around the rotated arc axis by arc angle

        Examples:
        - ring=0°, arc=0° → (-1, 0, 0) RIGHT
        - ring=0°, arc=90° → (0, 1, 0) FORWARD
        - ring=90°, arc=0° → (-1, 0, 0) RIGHT
        - ring=90°, arc=90° → (0, 0, -1) UP

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

        # Rotate base_direction around arc_axis using Rodriguez formula
        # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
        k = arc_axis
        v = base_direction

        direction = (v * np.cos(arc_rad) +
                     np.cross(k, v) * np.sin(arc_rad) +
                     k * np.dot(k, v) * (1 - np.cos(arc_rad)))

        direction = direction / np.linalg.norm(direction)

        return direction, arc_axis


