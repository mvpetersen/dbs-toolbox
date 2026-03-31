"""Simplified transform page for coordinate transformations."""

from nicegui import ui, events, app
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
import uuid

from dbstoolbox.utils.notifications import notify_error, notify_success, notify_info
from dbstoolbox.utils.validate_electrode_json import validate_electrode_reconstruction
from dbstoolbox.utils.validate_surgical_csv import validate_surgical_csv
from dbstoolbox.utils.validate_frame_registration import validate_frame_registration
from dbstoolbox.utils.validate_ants_transform import validate_ants_transform
from dbstoolbox.utils.transform_coordinates import (
    transform_pypacer_reconstruction,
    transform_surgical_csv,
    convert_csv_to_json
)
from dbstoolbox.utils.temp_file_manager import save_uploaded_file, get_session_file_path, cleanup_session
import csv as csv_module
import json


class TransformItem:
    """Represents a loaded transformation."""

    def __init__(self, name: str, file_path: Path, file_type: str, invertible: bool = True, metadata: Optional[Dict] = None):
        self.name = name
        self.file_path = file_path
        self.file_type = file_type  # 'mat', 'json', 'nifti_warp'
        self.inverted = False
        self.invertible = invertible  # Whether this transform can be inverted
        self.metadata = metadata or {}  # Store validation metadata

        # Determine transform category
        if self.metadata.get('is_frame_registration'):
            self.transform_category = 'frame_registration'
        elif self.metadata.get('is_ants_transform'):
            self.transform_category = 'ants'
        else:
            self.transform_category = 'unknown'


class SimpleTransformPage:
    """Simplified transform page for coordinate transformations."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())  # Unique session ID for temp files
        self.data_file: Optional[Path] = None
        self.data_filename: Optional[str] = None
        self.data_metadata: Optional[Dict] = None  # Store metadata for PyPaCER files
        self.transforms: List[TransformItem] = []
        self.preview_count = 0  # Track number of open previews for stacking offset

        # Register cleanup on disconnect
        app.on_disconnect(lambda: cleanup_session(self.session_id))
        self.transformed_files: List[Path] = []  # List of output files
        self.has_current_output = False  # Track if output is current (blocks transform button)

        # References to visualize page and tabs (set during create_ui)
        self.visualize_page = None
        self.tabs = None

        # UI elements to update
        self.data_status_label = None
        self.data_metadata_container = None
        self.transforms_list_container = None
        self.output_container = None
        self.transform_button = None

    def create_ui(self, visualize_page=None, tabs=None):
        """Create the simplified transform interface.

        Args:
            visualize_page: Reference to VisualizePage for loading data
            tabs: Reference to tabs object for switching tabs
        """
        self.visualize_page = visualize_page
        self.tabs = tabs

        # Add global CSS to hide uploader file lists
        ui.add_head_html('''
            <style>
            .hide-upload-list .q-uploader__list {
                display: none !important;
            }
            </style>
        ''')

        with ui.column().classes('w-full h-full p-6 gap-4'):
            # Header
            with ui.row().classes('w-full items-center mb-2'):
                ui.icon('transform', size='lg').classes('text-primary')
                ui.label('Coordinate Transform').classes('text-h5 ml-2')
                ui.label('Transform 3D coordinates using ANTs transformation matrices').classes(
                    'text-body2 text-grey-6 ml-4'
                )

            # Main content - horizontal layout with center alignment
            with ui.row().classes('w-full gap-4 justify-center items-start'):
                # Input data upload
                self._create_data_upload_section()

                # Arrow indicating data flow
                ui.icon('arrow_forward', size='lg').classes('text-primary mt-20')

                # Transform matrices upload and list (combined)
                self._create_transform_section()

                # Arrow indicating data flow
                ui.icon('arrow_forward', size='lg').classes('text-primary mt-20')

                # Output and transform button (combined)
                self._create_output_section()

    def _create_data_upload_section(self):
        """Create the data file upload section."""
        with ui.card().classes('w-80'):
            ui.label('1. Input Data').classes('text-subtitle2 mb-3')

            # Custom styled uploader
            self.data_upload = ui.upload(
                auto_upload=True,
                on_upload=self._handle_data_upload,
                label='Drag & drop or click to upload',
                max_file_size=500 * 1024 * 1024
            ).props('accept=".csv,.json" square outlined hide-upload-btn').classes('w-full')

            # Add custom styling to make it look like our design
            self.data_upload.classes('border-2 border-dashed border-grey-5 rounded hide-upload-list')

            # Add hover effect using Quasar's native events
            self.data_upload.on('dragover', lambda: self.data_upload.classes(remove='border-grey-5', add='border-primary'))
            self.data_upload.on('dragleave', lambda: self.data_upload.classes(remove='border-primary', add='border-grey-5'))
            self.data_upload.on('drop', lambda: self.data_upload.classes(remove='border-primary', add='border-grey-5'))

            # Status and buttons row
            with ui.row().classes('w-full items-center justify-between mt-3 gap-2'):
                with ui.row().classes('items-center gap-2 flex-grow'):
                    ui.icon('info', size='sm').classes('text-grey-6')
                    self.data_status_label = ui.label('No file loaded').classes('text-caption text-grey-7')

                # Action buttons (hidden initially)
                with ui.row().classes('gap-1'):
                    self.data_preview_btn = ui.button(
                        icon='visibility',
                        on_click=self._preview_input_data
                    ).props('flat dense round size=sm color=primary')
                    self.data_preview_btn.set_visibility(False)

                    self.data_remove_btn = ui.button(
                        icon='delete',
                        on_click=self._remove_data_file
                    ).props('flat dense round size=sm color=negative')
                    self.data_remove_btn.set_visibility(False)

            # Metadata container (shown for valid PyPaCER files)
            self.data_metadata_container = ui.column().classes('w-full mt-2')
            self._update_data_metadata()

    def _create_transform_section(self):
        """Create the transform matrices upload and list section (combined)."""
        with ui.card().classes('flex-1 max-w-[600px]'):
            ui.label('2. Transformation Matrices').classes('text-subtitle2 mb-3')

            # Custom styled uploader
            self.transform_upload = ui.upload(
                auto_upload=True,
                multiple=True,
                on_upload=self._handle_transform_upload,
                label='Drag & drop or click to upload',
                max_file_size=500 * 1024 * 1024
            ).props('accept=".mat,.json,.nii,.nii.gz" square outlined hide-upload-btn').classes('w-full mb-4')

            # Add custom styling to make it look like our design
            self.transform_upload.classes('border-2 border-dashed border-grey-5 rounded hide-upload-list')

            # Add hover effect using Quasar's native events
            self.transform_upload.on('dragover', lambda: self.transform_upload.classes(remove='border-grey-5', add='border-primary'))
            self.transform_upload.on('dragleave', lambda: self.transform_upload.classes(remove='border-primary', add='border-grey-5'))
            self.transform_upload.on('drop', lambda: self.transform_upload.classes(remove='border-primary', add='border-grey-5'))

            # Help section with transform ordering instructions
            with ui.expansion('How to chain transforms', icon='help').classes('w-full mb-3'):
                with ui.column().classes('gap-3 p-3 bg-grey-9 rounded'):
                    ui.markdown('''
**Transform Order:**

Chain ANTs transforms in the **reverse order** of your desired conversion.

To convert from A → B → C, upload transforms in this order:
1. First: C → B transform
2. Second: B → A transform

Use the invert toggle if you need to reverse a transform's direction.
                                
Note: Stereotactic frame transforms are handled separately and will always be applied first. When loaded they will default to invert=ON which transforms points from frame to ct space.
''').classes('text-body2 text-grey-1')

            # Info note
            with ui.row().classes('w-full items-start mb-3 gap-2'):
                ui.icon('lightbulb', size='sm').classes('text-amber-7')
                ui.label('.mat (affine/rigid), .json (4x4 matrix), .nii/.nii.gz (warp)').classes('text-caption text-grey-7')

            # Transforms list container
            ui.separator()
            ui.label('Loaded Transforms').classes('text-subtitle2 mt-3 mb-2')
            self.transforms_list_container = ui.column().classes('w-full gap-2')
            self._update_transform_list()

    def _create_output_section(self):
        """Create the output and transform button section (combined)."""
        with ui.card().classes('w-[400px]'):
            ui.label('3. Transform & Output').classes('text-subtitle2 mb-3')

            # Transform button with loading state
            self.transform_button = ui.button(
                'Transform Data',
                icon='play_arrow',
                on_click=self._apply_transformation
            ).props('unelevated color=primary').classes('w-full mb-4')
            self.transform_button.disable()

            # Progress indicator (hidden by default)
            self.transform_progress = ui.linear_progress(value=0, show_value=False).props('indeterminate color=primary')
            self.transform_progress.set_visibility(False)

            # Info
            with ui.row().classes('w-full items-center mb-4 gap-2'):
                ui.icon('info', size='sm').classes('text-grey-6')
                ui.label('Load data + transforms').classes('text-caption text-grey-7')

            # Output container
            ui.separator()
            ui.label('Output').classes('text-subtitle2 mt-3 mb-2')
            self.output_container = ui.column().classes('w-full')
            self._update_output_display()

    # Event handlers
    async def _handle_data_upload(self, e: events.UploadEventArguments):
        """Handle data file upload."""
        try:
            uploaded_file = e.file
            file_name = uploaded_file.name
            file_content = await uploaded_file.read()

            # Save to session directory
            temp_path = save_uploaded_file(file_content, file_name, self.session_id)

            self.data_file = temp_path
            self.data_filename = file_name
            self.data_metadata = None

            # Clear output data when new input is loaded
            self.transformed_files = []
            self.has_current_output = False

            file_size_kb = len(file_content) / 1024

            # Validate based on file type
            if file_name.endswith('.json'):
                is_valid, metadata, error_msg = validate_electrode_reconstruction(temp_path)

                if is_valid and metadata:
                    self.data_metadata = metadata
                    self.data_status_label.set_text(
                        f'Valid PyPaCER reconstruction ({file_size_kb:.1f} KB)'
                    )
                    notify_success(f'Loaded PyPaCER reconstruction: {file_name}')
                else:
                    # Still accept it but show warning
                    self.data_status_label.set_text(
                        f'Loaded: {file_name} ({file_size_kb:.1f} KB) - Not a PyPaCER file'
                    )
                    notify_info(f'Loaded JSON file (not PyPaCER format): {file_name}')

            elif file_name.endswith('.csv'):
                is_valid, metadata, error_msg = validate_surgical_csv(temp_path)

                if is_valid and metadata:
                    self.data_metadata = metadata
                    self.data_status_label.set_text(
                        f'Valid surgical data CSV ({file_size_kb:.1f} KB)'
                    )
                    notify_success(f'Loaded surgical data: {file_name}')
                else:
                    # Show error but don't load
                    self.data_file = None
                    self.data_filename = None
                    self.data_status_label.set_text('Invalid CSV format')
                    notify_error(f'Invalid CSV file: {error_msg}')
                    return
            else:
                # Unknown file type
                self.data_status_label.set_text(
                    f'Loaded: {file_name} ({file_size_kb:.1f} KB)'
                )
                notify_info(f'Loaded file: {file_name}')

            # Show action buttons
            self.data_preview_btn.set_visibility(True)
            self.data_remove_btn.set_visibility(True)

            self._update_data_metadata()
            self._update_output_display()  # Clear output display
            self._update_transform_button_state()

        except Exception as ex:
            notify_error(f'Failed to load data file: {str(ex)}')

    async def _handle_transform_upload(self, e: events.UploadEventArguments):
        """Handle transform file upload."""
        try:
            uploaded_file = e.file
            file_name = uploaded_file.name
            file_content = await uploaded_file.read()

            # Save to session directory
            temp_path = save_uploaded_file(file_content, file_name, self.session_id)

            # Validate based on file type
            is_valid = False
            metadata = None
            error_msg = None
            invertible = True
            file_type = 'unknown'

            if file_name.endswith('.mat'):
                # ANTs .mat file
                is_valid, metadata, error_msg = validate_ants_transform(temp_path)
                file_type = 'mat'
                if metadata:
                    invertible = metadata.get('invertible', True)

            elif file_name.endswith('.json'):
                # Frame registration JSON
                is_valid, metadata, error_msg = validate_frame_registration(temp_path)
                file_type = 'json'
                invertible = True  # JSON 4x4 matrices are invertible

            elif file_name.endswith('.nii.gz') or file_name.endswith('.nii'):
                # NIfTI warp field
                is_valid, metadata, error_msg = validate_ants_transform(temp_path)
                file_type = 'nifti_warp'
                if metadata:
                    invertible = metadata.get('invertible', False)

            # Check validation result
            if not is_valid:
                notify_error(f'Invalid transform file: {error_msg}')
                return

            # Create transform item
            transform = TransformItem(file_name, temp_path, file_type, invertible, metadata)

            # Auto-invert frame registration transforms (most common use case)
            # Frame registration goes from CT -> Frame, but we usually want Frame -> CT
            if metadata and metadata.get('is_frame_registration'):
                transform.inverted = True

            self.transforms.append(transform)

            # Show success with metadata
            if metadata and 'transform_type' in metadata:
                invert_note = " (auto-inverted)" if transform.inverted else ""
                notify_success(f'Added {metadata["transform_type"]} transform: {file_name}{invert_note}')
            else:
                notify_success(f'Added transform: {file_name}')

            # Clear output data when transform chain is modified
            self.transformed_files = []
            self.has_current_output = False
            self._update_output_display()

            self._update_transform_list()
            self._update_transform_button_state()

        except Exception as ex:
            notify_error(f'Failed to load transform: {str(ex)}')

    def _update_transform_list(self):
        """Update the transform list display."""
        self.transforms_list_container.clear()

        with self.transforms_list_container:
            if not self.transforms:
                with ui.column().classes('w-full items-center justify-center p-4 border border-dashed border-grey-4 rounded'):
                    ui.icon('folder_off', size='sm').classes('text-grey-5 mb-1')
                    ui.label('No transforms').classes('text-caption text-grey-6 italic')
            else:
                for idx, transform in enumerate(self.transforms):
                    with ui.card().classes('w-full p-2'):
                        with ui.row().classes('w-full items-center justify-between gap-2'):
                            # Transform info
                            with ui.column().classes('gap-0 flex-grow'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.label(f'{idx + 1}.').classes('text-caption text-grey-7')
                                    ui.label(transform.name).classes('text-body2 font-medium')

                                    # Show file type badge
                                    badge_text = transform.file_type.upper()
                                    if transform.metadata and 'transform_type' in transform.metadata:
                                        badge_text = transform.metadata['transform_type']
                                    ui.chip(badge_text).props('dense').classes('text-caption')

                                # Invert toggle (only for invertible transforms)
                                if transform.invertible:
                                    ui.switch(
                                        'Invert',
                                        value=transform.inverted,
                                        on_change=lambda e, t=transform: self._toggle_invert(t, e.value)
                                    ).props('dense')
                                else:
                                    # Show non-invertible indicator
                                    with ui.row().classes('items-center gap-1'):
                                        ui.icon('info', size='xs').classes('text-grey-6')
                                        ui.label('Non-invertible').classes('text-caption text-grey-6 italic')

                            # Remove button
                            ui.button(
                                icon='delete',
                                on_click=lambda t=transform: self._remove_transform(t)
                            ).props('flat dense round size=sm color=negative')

    def _toggle_invert(self, transform: TransformItem, inverted: bool):
        """Toggle transform inversion."""
        transform.inverted = inverted
        status = 'inverted' if inverted else 'normal'
        notify_info(f'Transform "{transform.name}" set to {status}')

        # Clear output data when transform chain is modified
        self.transformed_files = []
        self.has_current_output = False
        self._update_output_display()
        self._update_transform_button_state()

    def _remove_transform(self, transform: TransformItem):
        """Remove a transform from the list."""
        self.transforms.remove(transform)

        # Reset the uploader to allow re-uploading the same file
        self.transform_upload.reset()

        notify_info(f'Removed transform: {transform.name}')

        # Clear output data when transform chain is modified
        self.transformed_files = []
        self.has_current_output = False
        self._update_output_display()

        self._update_transform_list()
        self._update_transform_button_state()

    def _remove_data_file(self):
        """Remove the loaded data file."""
        self.data_file = None
        self.data_filename = None
        self.data_metadata = None

        # Reset uploader
        self.data_upload.reset()

        # Update UI
        self.data_status_label.set_text('No file loaded')
        self.data_preview_btn.set_visibility(False)
        self.data_remove_btn.set_visibility(False)
        self._update_data_metadata()
        self._update_transform_button_state()

        notify_info('Input data removed')

    def _update_data_metadata(self):
        """Update the data metadata display."""
        self.data_metadata_container.clear()

        if self.data_metadata:
            with self.data_metadata_container:
                with ui.card().classes('w-full p-3 bg-positive-1'):
                    with ui.column().classes('gap-1'):
                        # Check if PyPaCER reconstruction or surgical CSV
                        if self.data_metadata.get('is_pypacer_reconstruction'):
                            # PyPaCER reconstruction
                            with ui.row().classes('items-center gap-2 mb-2'):
                                ui.icon('check_circle', color='positive', size='sm')
                                ui.label('PyPaCER Reconstruction').classes('text-subtitle2 font-medium')

                            with ui.column().classes('gap-1'):
                                # Date
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('event', size='xs').classes('text-grey-7')
                                    ui.label('Date:').classes('text-caption text-grey-8 font-medium')
                                    ui.label(self.data_metadata.get('timestamp', 'Unknown')).classes('text-caption text-grey-8')

                                # PyPaCER version
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('settings', size='xs').classes('text-grey-7')
                                    ui.label('PyPaCER:').classes('text-caption text-grey-8 font-medium')
                                    ui.label(f"v{self.data_metadata.get('pypacer_version', 'unknown')}").classes('text-caption text-grey-8')

                                # Number of electrodes
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('sensors', size='xs').classes('text-grey-7')
                                    ui.label('Electrodes:').classes('text-caption text-grey-8 font-medium')
                                    ui.label(str(self.data_metadata.get('num_electrodes', 0))).classes('text-caption text-grey-8')

                        elif self.data_metadata.get('is_surgical_data'):
                            # Surgical data CSV
                            with ui.row().classes('items-center gap-2 mb-2'):
                                ui.icon('check_circle', color='positive', size='sm')
                                ui.label('Surgical Data CSV').classes('text-subtitle2 font-medium')

                            with ui.column().classes('gap-1'):
                                # Number of records
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('table_rows', size='xs').classes('text-grey-7')
                                    ui.label('Records:').classes('text-caption text-grey-8 font-medium')
                                    ui.label(str(self.data_metadata.get('num_records', 0))).classes('text-caption text-grey-8')

                                # Number of patients
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('people', size='xs').classes('text-grey-7')
                                    ui.label('Patients:').classes('text-caption text-grey-8 font-medium')
                                    ui.label(str(self.data_metadata.get('num_patients', 0))).classes('text-caption text-grey-8')

                                # Number of coordinates
                                if 'num_coordinates' in self.data_metadata:
                                    with ui.row().classes('items-center gap-2'):
                                        ui.icon('place', size='xs').classes('text-grey-7')
                                        ui.label('Coordinates:').classes('text-caption text-grey-8 font-medium')
                                        ui.label(str(self.data_metadata.get('num_coordinates', 0))).classes('text-caption text-grey-8')

    def _update_transform_button_state(self):
        """Enable/disable transform button based on loaded data."""
        if self.data_file and len(self.transforms) > 0 and not self.has_current_output:
            self.transform_button.enable()
        else:
            self.transform_button.disable()

    async def _apply_transformation(self):
        """Apply the transformations to the data."""
        if not self.data_file or not self.transforms:
            notify_error('Missing data file or transforms')
            return

        try:
            # Show progress and disable button
            self.transform_progress.set_visibility(True)
            self.transform_button.disable()
            notify_info(f'Applying {len(self.transforms)} transformation(s)...')

            # Prepare transform data
            transform_files = []
            invert_flags = []
            transform_types = []

            print("\n" + "="*60)
            print("TRANSFORM ORDER DEBUG")
            print("="*60)
            print("Upload order (as user sees in UI):")
            for idx, transform in enumerate(self.transforms):
                transform_files.append(transform.file_path)
                invert_flags.append(transform.inverted)
                transform_types.append(transform.transform_category)
                print(f"{idx+1}. {transform.name}")
                print(f"   - File: {transform.file_path.name}")
                print(f"   - Inverted: {transform.inverted}")
                print(f"   - Type: {transform.transform_category}")

            # REVERSE the lists so ANTs applies them in the order the user uploaded
            # User thinks: "I want A→C, so upload C→B then B→A"
            # We reverse: [B→A, C→B]
            # ANTs reverses: applies C→B then B→A
            # Result: A→B→C ✓
            transform_files.reverse()
            invert_flags.reverse()
            transform_types.reverse()

            print(f"\nAfter reversing (passed to ANTs):")
            print(f"  transformlist = {[f.name for f in transform_files]}")
            print(f"  whichtoinvert = {invert_flags}")
            print(f"\nANTs will apply in REVERSE, giving the original upload order")
            print("="*60 + "\n")

            # Apply transformations based on data type
            if self.data_filename.endswith('.json'):
                await self._transform_json_data(transform_files, invert_flags, transform_types)
            elif self.data_filename.endswith('.csv'):
                await self._transform_csv_data(transform_files, invert_flags, transform_types)
            else:
                notify_error('Unsupported data file format')
                return

            notify_success('Transformation complete!')
            self.has_current_output = True  # Mark output as current (blocks transform button)
            self._update_output_display()

        except Exception as ex:
            notify_error(f'Transformation failed: {str(ex)}')
            import traceback
            traceback.print_exc()

        finally:
            # Hide progress and update button state
            self.transform_progress.set_visibility(False)
            self._update_transform_button_state()

    async def _transform_json_data(self, transform_files: List[Path], invert_flags: List[bool], transform_types: List[str]):
        """Transform PyPaCER reconstruction JSON data."""
        # Load JSON data
        with open(self.data_file, 'r') as f:
            data = json.load(f)

        # Check if it's a PyPaCER reconstruction
        if self.data_metadata and self.data_metadata.get('is_pypacer_reconstruction'):
            # Transform using PyPaCER reconstruction method (run in executor to avoid blocking UI)
            import concurrent.futures
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                transformed_data, num_contacts = await loop.run_in_executor(
                    executor,
                    transform_pypacer_reconstruction,
                    data, transform_files, invert_flags, transform_types
                )

            # Save transformed data to session directory
            output_filename = f"{self.data_filename.rsplit('.', 1)[0]}_transformed.json"
            output_path = get_session_file_path(output_filename, self.session_id)

            with open(output_path, 'w') as f:
                json.dump(transformed_data, f, indent=2)

            # Store output file
            self.transformed_files = [output_path]

            notify_info(f'Transformed {num_contacts} electrode contacts')

        else:
            # Generic JSON - try to transform any x,y,z coordinates
            notify_error('Only PyPaCER reconstruction JSON files are currently supported')
            return

    async def _transform_csv_data(self, transform_files: List[Path], invert_flags: List[bool], transform_types: List[str]):
        """Transform surgical data CSV."""
        # Load CSV data
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            csv_data = list(reader)
            fieldnames = reader.fieldnames

        # Transform coordinates (run in executor to avoid blocking UI)
        import concurrent.futures
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            transformed_data = await loop.run_in_executor(
                executor,
                transform_surgical_csv,
                csv_data, transform_files, invert_flags, transform_types
            )

        # Add new columns to fieldnames if not present
        new_fieldnames = list(fieldnames)

        # Add original coordinate and angle columns
        for col in ['x_original', 'y_original', 'z_original', 'ring_original', 'arc_original']:
            if col not in new_fieldnames:
                # Insert after the corresponding coordinate/angle column
                base_col = col.split('_')[0]
                if base_col in new_fieldnames:
                    idx = new_fieldnames.index(base_col) + 1
                    new_fieldnames.insert(idx, col)

        # Add virtual entry point columns (from transformed surgical data)
        for col in ['entry_x', 'entry_y', 'entry_z']:
            if col not in new_fieldnames:
                new_fieldnames.append(col)

        # Convert to JSON and save to session directory
        base_filename = self.data_filename.rsplit('.', 1)[0]
        json_data = convert_csv_to_json(transformed_data)
        json_output_filename = f"{base_filename}_transformed.json"
        json_output_path = get_session_file_path(json_output_filename, self.session_id)

        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # Store only JSON file
        self.transformed_files = [json_output_path]

        num_coords = sum(1 for row in transformed_data if 'x_original' in row)
        notify_info(f'Transformed {num_coords} coordinate points')

    def _update_output_display(self):
        """Update the output display."""
        self.output_container.clear()

        with self.output_container:
            if not self.transformed_files or not all(f.exists() for f in self.transformed_files):
                with ui.column().classes('w-full items-center justify-center p-4 border border-dashed border-grey-4 rounded'):
                    ui.icon('pending', size='sm').classes('text-grey-5 mb-1')
                    ui.label('No output yet').classes('text-caption text-grey-6 italic')
            else:
                # Success state with single download button at bottom
                with ui.column().classes('w-full gap-3'):
                    with ui.row().classes('w-full items-center gap-2 p-3 bg-positive-1 rounded'):
                        ui.icon('check_circle', color='positive', size='sm')
                        ui.label('Transformation Complete').classes('text-subtitle2 font-medium')

                    # List output file with preview button
                    ui.label('Output File').classes('text-caption text-grey-7 mb-1')
                    for output_file in self.transformed_files:
                        with ui.row().classes('w-full items-center justify-between p-2 rounded border border-grey-7'):
                            with ui.row().classes('items-center gap-2 flex-grow'):
                                # Icon based on file type
                                icon_name = 'description' if output_file.suffix == '.json' else 'table_chart'
                                ui.icon(icon_name, size='sm').classes('text-grey-7')
                                ui.label(output_file.name).classes('text-body2')

                            # Preview button only
                            ui.button(
                                icon='visibility',
                                on_click=lambda f=output_file: self._preview_output_data(f)
                            ).props('flat dense round size=sm color=primary')

                    # Buttons at bottom
                    with ui.row().classes('w-full gap-2 mt-2'):
                        # Download button
                        ui.button(
                            'Download',
                            icon='download',
                            on_click=lambda: self._download_file(self.transformed_files[0])
                        ).props('unelevated color=positive').classes('flex-1')

                        # Visualize button (if visualize page is available)
                        if self.visualize_page is not None and self.tabs is not None:
                            ui.button(
                                'Visualize',
                                icon='visibility',
                                on_click=lambda: self._load_to_visualize()
                            ).props('unelevated color=primary').classes('flex-1')

    async def _load_to_visualize(self):
        """Load transformed data to visualize page and switch to visualize tab."""
        if not self.visualize_page or not self.tabs:
            notify_error('Visualize page not available')
            return

        if not self.transformed_files or not self.transformed_files[0].exists():
            notify_error('No transformed data to visualize')
            return

        try:
            # Get the transformed JSON file
            json_file = self.transformed_files[0]
            file_name = json_file.name
            file_size_kb = json_file.stat().st_size / 1024

            # Load the file into the visualize page
            await self.visualize_page._load_json_file(json_file, file_name, file_size_kb)

            # Update visualize page displays
            self.visualize_page._update_file_list()
            self.visualize_page._update_status()
            self.visualize_page._update_mesh_controls_visibility()
            self.visualize_page._update_plot()

            # Switch to visualize tab
            self.tabs.set_value('visualize')

            notify_success(f'Loaded {file_name} to visualize tab')

        except Exception as ex:
            notify_error(f'Failed to load to visualize: {str(ex)}')
            import traceback
            traceback.print_exc()

    def _preview_input_data(self):
        """Preview the input data file."""
        if not self.data_file or not self.data_file.exists():
            notify_error('No input data to preview')
            return

        try:
            if self.data_filename.endswith('.csv'):
                self._show_csv_preview(self.data_file, f'Preview: {self.data_filename}')
            elif self.data_filename.endswith('.json'):
                self._show_json_preview(self.data_file, f'Preview: {self.data_filename}')
        except Exception as ex:
            notify_error(f'Failed to preview data: {str(ex)}')

    def _preview_output_data(self, file_path: Path):
        """Preview a transformed output file."""
        if not file_path or not file_path.exists():
            notify_error('File not available for preview')
            return

        try:
            if file_path.suffix == '.csv':
                self._show_csv_preview(file_path, f'Preview: {file_path.name}')
            elif file_path.suffix == '.json':
                self._show_json_preview(file_path, f'Preview: {file_path.name}')
        except Exception as ex:
            notify_error(f'Failed to preview file: {str(ex)}')

    def _show_csv_preview(self, file_path: Path, title: str):
        """Show a preview dialog for CSV data."""
        # Load CSV data
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []

        # Calculate offset for stacking
        offset = self.preview_count * 30  # 30px offset per preview
        self.preview_count += 1

        # Create dialog with responsive sizing
        with ui.dialog() as dialog, ui.card().classes('w-full').style('min-width: 80vw; max-width: 90vw; max-height: 80vh'):
            # Apply offset positioning after dialog is created
            ui.run_javascript(f'''
                setTimeout(() => {{
                    const dialogs = document.querySelectorAll('.q-dialog');
                    const lastDialog = dialogs[dialogs.length - 1];
                    if (lastDialog) {{
                        const card = lastDialog.querySelector('.q-card');
                        if (card) {{
                            card.style.transform = 'translate({offset}px, {offset}px)';
                        }}
                    }}
                }}, 100);
            ''')

            # Decrement count when dialog closes
            dialog.on('close', lambda: setattr(self, 'preview_count', max(0, self.preview_count - 1)))
            # Header
            with ui.row().classes('w-full items-center justify-between mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('table_chart', size='md').classes('text-primary')
                    ui.label(title).classes('text-h6')
                    ui.label(f'{len(rows)} rows × {len(columns)} columns').classes('text-body2 text-grey-6 ml-2')
                ui.button(icon='close', on_click=dialog.close).props('flat round')

            # Table with scroll
            with ui.scroll_area().classes('w-full').style('max-height: 60vh'):
                # Limit preview to first 100 rows for performance
                preview_rows = rows[:100]

                # Create table data
                table_columns = [
                    {'name': col, 'label': col, 'field': col, 'align': 'left', 'sortable': True}
                    for col in columns
                ]

                ui.table(
                    columns=table_columns,
                    rows=preview_rows,
                    row_key='patient_id' if 'patient_id' in columns else None
                ).props('dense flat bordered').classes('w-full')

                if len(rows) > 100:
                    ui.label(f'Showing first 100 of {len(rows)} rows').classes('text-caption text-grey-6 mt-2')

        dialog.open()

    def _show_json_preview(self, file_path: Path, title: str):
        """Show a preview dialog for JSON electrode data."""
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Calculate offset for stacking
        offset = self.preview_count * 30  # 30px offset per preview
        self.preview_count += 1

        # Create dialog with responsive sizing
        with ui.dialog() as dialog, ui.card().classes('w-full').style('min-width: 80vw; max-width: 90vw; max-height: 80vh'):
            # Apply offset positioning after dialog is created
            ui.run_javascript(f'''
                setTimeout(() => {{
                    const dialogs = document.querySelectorAll('.q-dialog');
                    const lastDialog = dialogs[dialogs.length - 1];
                    if (lastDialog) {{
                        const card = lastDialog.querySelector('.q-card');
                        if (card) {{
                            card.style.transform = 'translate({offset}px, {offset}px)';
                        }}
                    }}
                }}, 100);
            ''')

            # Decrement count when dialog closes
            dialog.on('close', lambda: setattr(self, 'preview_count', max(0, self.preview_count - 1)))
            # Header
            with ui.row().classes('w-full items-center justify-between mb-4'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('description', size='md').classes('text-primary')
                    ui.label(title).classes('text-h6')
                ui.button(icon='close', on_click=dialog.close).props('flat round')

            # Content with scroll
            with ui.scroll_area().classes('w-full').style('max-height: 60vh'):
                # Check if it's a PyPaCER reconstruction
                if 'electrodes' in data and isinstance(data['electrodes'], list):
                    # Show electrode list
                    ui.label('Electrodes').classes('text-subtitle1 mb-3')

                    for idx, electrode in enumerate(data['electrodes']):
                        with ui.expansion(f"Electrode {idx + 1}", icon='sensors').classes('w-full mb-2'):
                            with ui.column().classes('gap-2 p-2'):
                                # Show tip position
                                if 'tip_position' in electrode:
                                    tip = electrode['tip_position']
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        ui.icon('location_on', size='sm').classes('text-green')
                                        ui.label('Tip:').classes('text-body2 font-medium')
                                        ui.label(f'[{tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f}]').classes('text-body2 text-grey-7')

                                # Show entry position
                                if 'entry_position' in electrode:
                                    entry = electrode['entry_position']
                                    with ui.row().classes('items-center gap-2 mb-2'):
                                        ui.icon('flag', size='sm').classes('text-blue')
                                        ui.label('Entry:').classes('text-body2 font-medium')
                                        ui.label(f'[{entry[0]:.2f}, {entry[1]:.2f}, {entry[2]:.2f}]').classes('text-body2 text-grey-7')

                                # Show contact positions
                                if 'contact_positions_3d' in electrode:
                                    contacts = electrode['contact_positions_3d']
                                    ui.label(f'Contacts: {len(contacts)}').classes('text-body2 font-medium mb-2 mt-2')

                                    # Create table for contacts
                                    contact_columns = [
                                        {'name': 'contact', 'label': 'Contact', 'field': 'contact', 'align': 'left'},
                                        {'name': 'x', 'label': 'X', 'field': 'x', 'align': 'right'},
                                        {'name': 'y', 'label': 'Y', 'field': 'y', 'align': 'right'},
                                        {'name': 'z', 'label': 'Z', 'field': 'z', 'align': 'right'},
                                    ]

                                    contact_rows = []
                                    for contact_idx, coords in enumerate(contacts):
                                        if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                                            contact_rows.append({
                                                'contact': f"C{contact_idx}",
                                                'x': f"{coords[0]:.2f}",
                                                'y': f"{coords[1]:.2f}",
                                                'z': f"{coords[2]:.2f}"
                                            })

                                    if contact_rows:
                                        ui.table(
                                            columns=contact_columns,
                                            rows=contact_rows,
                                            row_key='contact'
                                        ).props('dense flat bordered').classes('w-full')

                                # Show trajectory info if available
                                if 'trajectory_coordinates' in electrode:
                                    trajectory = electrode['trajectory_coordinates']
                                    ui.label(f'Trajectory points: {len(trajectory)}').classes('text-caption text-grey-6 mt-2')

                elif 'surgical_data' in data and isinstance(data['surgical_data'], list):
                    # Transformed CSV converted to JSON
                    ui.label('Surgical Data').classes('text-subtitle1 mb-3')

                    surgical_data = data['surgical_data']
                    ui.label(f'{len(surgical_data)} records').classes('text-body2 text-grey-6 mb-2')

                    # Show as table
                    if surgical_data:
                        # Get columns from first record
                        columns = list(surgical_data[0].keys())
                        table_columns = [
                            {'name': col, 'label': col, 'field': col, 'align': 'left'}
                            for col in columns
                        ]

                        # Limit to first 100 rows
                        preview_data = surgical_data[:100]

                        ui.table(
                            columns=table_columns,
                            rows=preview_data,
                            row_key='patient_id' if 'patient_id' in columns else None
                        ).props('dense flat bordered').classes('w-full')

                        if len(surgical_data) > 100:
                            ui.label(f'Showing first 100 of {len(surgical_data)} records').classes('text-caption text-grey-6 mt-2')

                else:
                    # Generic JSON - show formatted text
                    ui.label('JSON Content').classes('text-subtitle1 mb-3')
                    ui.code(json.dumps(data, indent=2)).classes('w-full')

        dialog.open()

    def _download_file(self, file_path: Path):
        """Download a specific output file."""
        if not file_path or not file_path.exists():
            notify_error('File not available')
            return

        try:
            # Use NiceGUI's download functionality
            ui.download(file_path, filename=file_path.name)
            notify_success(f'Downloading {file_path.name}')
        except Exception as ex:
            notify_error(f'Failed to download file: {str(ex)}')


def simple_transform_page():
    """Create the simple transform page."""
    page = SimpleTransformPage()
    page.create_ui()
