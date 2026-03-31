"""Report generation dialog component with automatic file type detection."""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from nicegui import ui
from dbstoolbox.utils.temp_file_manager import save_uploaded_file
from dbstoolbox.reports import StereotacticReportGenerator
import json
import sys
import queue


class _StdoutCapture:
    """Captures stdout writes and forwards them to a queue for UI display."""

    def __init__(self, original_stdout, status_queue: queue.Queue):
        self.original = original_stdout
        self.queue = status_queue

    def write(self, text: str):
        self.original.write(text)
        stripped = text.strip()
        if stripped:
            self.queue.put(stripped)

    def flush(self):
        self.original.flush()


class ReportGenerationDialog:
    """Dialog for generating reports from uploaded data files with automatic type detection."""

    def __init__(self):
        """Initialize the dialog."""
        self.uploaded_files: Dict[str, str] = {}  # {file_type: file_path}
        self.electrode_files: List[str] = []  # Support multiple electrode files for brain shift
        self.primary_electrode_idx: int = 0  # Index of primary electrode file
        self.file_list_container = None
        self.report_options: Dict[str, ui.element] = {}
        self.dialog = None
        self.ct_file_available: bool = False
        self.is_full_recon_json: bool = False
        self.is_transformed: bool = False
        self.pypacer_ct_warning = None  # UI element for CT warning in pypacer card
        self.pypacer_mini_warning = None  # UI element for mini JSON warning
        self.pypacer_transformed_warning = None  # UI element for transformed JSON warning
        self.pypacer_generate_btn = None
        self.upload_widget = None
        self.stimulation_threshold: float = 0.5
        self.stimulation_threshold_slider = None
        self.stimulation_threshold_container = None
        self._overlay = None
        self._status_label = None
        self._status_timer = None
        self._status_queue: queue.Queue = queue.Queue()
        self.patient_id_input = None

    def _detect_file_type(self, file_path: str, filename: str) -> Optional[str]:
        """
        Detect the type of file based on content and filename.

        Returns: 'surgical', 'electrode', 'nifti', 'nifti_3d', 'nifti_4d', 'label', 'acpc', or None
        """
        # Check CSV files - detect ACPC landmark files
        if filename.endswith('.csv'):
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip().lower()
                    content = f.read().lower()
                if 'label' in header and ('ac' in content or 'pc' in content):
                    return 'acpc'
            except Exception:
                pass
            return None

        # Check NIfTI files first (by extension) - distinguish 3D vs 4D
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            try:
                import nibabel as nib
                img = nib.load(file_path)
                if img.ndim == 4:
                    return 'nifti_4d'
                return 'nifti_3d'
            except Exception:
                return 'nifti_3d'

        # Check label TXT files
        if filename.endswith('.txt'):
            return 'label'

        # For JSON files, inspect the content
        if filename.endswith('.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Check for electrode reconstruction
                if isinstance(data, dict) and 'electrodes' in data:
                    return 'electrode'

                # Check for surgical planning data
                if isinstance(data, list):
                    # Check if list items have surgical target fields
                    if data and isinstance(data[0], dict):
                        if 'x' in data[0] and 'y' in data[0] and 'z' in data[0]:
                            # Could be surgical or electrode, check for surgical-specific fields
                            if 'ring' in data[0] or 'arc' in data[0] or 'anatomical_target' in data[0]:
                                return 'surgical'

                if isinstance(data, dict):
                    # Check for records/targets/data keys (surgical planning)
                    if 'records' in data or 'targets' in data:
                        return 'surgical'
                    if 'data' in data and isinstance(data['data'], list):
                        if data['data'] and 'x' in data['data'][0]:
                            return 'surgical'

            except Exception as e:
                print(f"Error detecting file type: {e}")
                return None

        return None

    def _detect_available_reports(self) -> List[str]:
        """Detect which reports can be generated based on uploaded files."""
        available = []

        # Check for stereotactic report (surgical + electrode)
        if 'surgical' in self.uploaded_files and 'electrode' in self.uploaded_files:
            available.append('stereotactic')

        # Check for PyPaCER report (electrode only)
        if 'electrode' in self.uploaded_files:
            available.append('pypacer')

        # Check for stimulation report (electrode + at least one NIfTI)
        has_nifti = 'nifti_3d' in self.uploaded_files or 'nifti_4d' in self.uploaded_files
        if 'electrode' in self.uploaded_files and has_nifti:
            available.append('stimulation')

        return available

    # File requirements for each report type: (check_key, label, is_optional)
    # check_key maps to self.uploaded_files keys, except 'electrode_primary' and
    # 'electrode_secondary' which check self.electrode_files length.
    REPORT_FILE_REQUIREMENTS = {
        'stereotactic': [
            ('surgical', 'Surgical Planning JSON', False),
            ('electrode_primary', 'Primary Electrode Recon JSON', False),
            ('electrode_secondary', 'Secondary Electrode Recon JSON', True),
            ('nifti_3d', 'NIfTI Volume (3D)', True),
        ],
        'pypacer': [
            ('electrode_primary', 'Primary Electrode Recon JSON (full)', False),
        ],
        'stimulation': [
            ('electrode', 'Electrode Recon JSON', False),
            ('nifti_3d', 'NIfTI Volume (3D)', True),
            ('nifti_4d', 'NIfTI Probability Map (4D)', True),
            ('label', 'Label File (TXT)', True),
            ('acpc', 'AC-PC Landmarks (CSV)', True),
        ],
    }

    MAX_ELECTRODE_FILES = 2

    def _is_file_loaded(self, check_key: str) -> bool:
        """Check whether a file requirement is satisfied."""
        if check_key == 'electrode_primary':
            return len(self.electrode_files) >= 1
        if check_key == 'electrode_secondary':
            return len(self.electrode_files) >= 2
        return check_key in self.uploaded_files

    def _update_report_options(self):
        """Update enabled/disabled state of report options and file badges."""
        available = self._detect_available_reports()

        for report_type, elements in self.report_options.items():
            card = elements['card']
            btn = elements['btn']
            is_available = report_type in available

            # Update card styling
            if is_available:
                card.classes(remove='opacity-60')
                if btn and report_type != 'pypacer':
                    btn.enable()
            else:
                card.classes(add='opacity-60')
                if btn and report_type != 'pypacer':
                    btn.disable()

            # Update file requirement badges
            has_any_nifti = 'nifti_3d' in self.uploaded_files or 'nifti_4d' in self.uploaded_files
            for check_key, badge, is_optional in elements.get('badges', []):
                if self._is_file_loaded(check_key):
                    badge.props('color=green')
                else:
                    # For stimulation NIfTI badges: once one is loaded the other is optional
                    effectively_optional = is_optional
                    if report_type == 'stimulation' and check_key in ('nifti_3d', 'nifti_4d') and has_any_nifti:
                        effectively_optional = True
                    if effectively_optional:
                        badge.props('color=orange')
                    else:
                        badge.props('color=blue')

        # Show/hide stimulation threshold slider based on 4D NIfTI
        if self.stimulation_threshold_container is not None:
            self.stimulation_threshold_container.set_visibility('nifti_4d' in self.uploaded_files)

    def _create_file_badges(self, report_type: str) -> List[Tuple[str, ui.element, bool]]:
        """Create badge elements for a report's file requirements. Returns list of (check_key, badge, is_optional)."""
        badges = []
        for check_key, label, is_optional in self.REPORT_FILE_REQUIREMENTS.get(report_type, []):
            prefix = '(optional) ' if is_optional else ''
            color = 'orange' if is_optional else 'blue'
            badge = ui.badge(f'{prefix}{label}', color=color)
            badges.append((check_key, badge, is_optional))
        return badges

    def _update_file_list(self):
        """Update the file list display."""
        if not self.file_list_container:
            return

        self.file_list_container.clear()

        with self.file_list_container:
            if not self.uploaded_files and not self.electrode_files:
                ui.label('No files loaded yet').classes('text-caption text-grey-6 italic')
                return

            # Show electrode files separately to support multiple
            for idx, file_path in enumerate(self.electrode_files):
                filename = Path(file_path).name
                is_primary = (idx == self.primary_electrode_idx)
                label = 'Primary Electrode Reconstruction' if is_primary else 'Secondary Electrode Reconstruction'

                # Different styling for primary electrode
                card_classes = 'w-full items-center gap-2 p-2 rounded bg-grey-50'
                if is_primary:
                    card_classes += ' border-2 border-red-500'
                else:
                    card_classes += ' border-2 border-transparent'

                with ui.row().classes(card_classes):
                    ui.icon('sensors', color='red' if is_primary else 'grey')
                    with ui.column().classes('flex-1 gap-0'):
                        with ui.row().classes('items-center gap-2'):
                            ui.label(label).classes('text-body2 font-medium')
                            if is_primary:
                                ui.badge('Primary', color='red').classes('text-xs')
                        ui.label(filename).classes('text-caption text-grey-7')

                    # Only show "Set Primary" button if not already primary and there are multiple files
                    if not is_primary and len(self.electrode_files) > 1:
                        ui.button(
                            'Set Primary',
                            on_click=lambda i=idx: self._set_primary_electrode(i)
                        ).props('flat dense size=sm color=primary').tooltip('Use this file for stereotactic report')

                    ui.button(
                        icon='close',
                        on_click=lambda i=idx: self._remove_electrode_file(i)
                    ).props('flat round dense size=sm').tooltip('Remove file')

            # Show other loaded files (excluding electrode since we handle it above)
            for file_type, file_path in self.uploaded_files.items():
                if file_type == 'electrode':
                    continue  # Already handled above

                filename = Path(file_path).name

                # Icon based on file type
                icon_map = {
                    'surgical': 'medical_services',
                    'nifti_3d': 'view_in_ar',
                    'nifti_4d': 'layers',
                    'label': 'label',
                    'acpc': 'straighten',
                }
                icon = icon_map.get(file_type, 'insert_drive_file')

                # Label based on file type
                label_map = {
                    'surgical': 'Surgical Planning',
                    'nifti_3d': 'NIfTI Volume (3D)',
                    'nifti_4d': 'NIfTI Probability Map (4D)',
                    'label': 'Volume Labels',
                    'acpc': 'AC-PC Landmarks',
                }
                label = label_map.get(file_type, 'Unknown')

                with ui.row().classes('w-full items-center gap-2 p-2 bg-grey-50 rounded'):
                    ui.icon(icon, color='primary')
                    with ui.column().classes('flex-1 gap-0'):
                        ui.label(label).classes('text-body2 font-medium')
                        ui.label(filename).classes('text-caption text-grey-7')
                    ui.button(
                        icon='close',
                        on_click=lambda ft=file_type: self._remove_file(ft)
                    ).props('flat round dense size=sm').tooltip('Remove file')

    def _remove_file(self, file_type: str):
        """Remove a loaded file."""
        if file_type in self.uploaded_files:
            del self.uploaded_files[file_type]
            if file_type == 'surgical' and self.patient_id_input is not None:
                self.patient_id_input.value = ''
            self._update_file_list()
            self._update_report_options()
            if self.upload_widget:
                self.upload_widget.reset()

    def _set_primary_electrode(self, index: int):
        """Set the primary electrode file by index."""
        if 0 <= index < len(self.electrode_files):
            self.primary_electrode_idx = index
            # Update the main dict to point to the primary electrode
            self.uploaded_files['electrode'] = self.electrode_files[index]
            self._update_file_list()
            self._check_pypacer_compatibility()

    def _remove_electrode_file(self, index: int):
        """Remove an electrode file by index."""
        if 0 <= index < len(self.electrode_files):
            del self.electrode_files[index]

            # Adjust primary index if needed
            if self.primary_electrode_idx >= len(self.electrode_files):
                self.primary_electrode_idx = max(0, len(self.electrode_files) - 1)
            elif self.primary_electrode_idx > index:
                self.primary_electrode_idx -= 1

            # Update the main dict to point to primary electrode if available
            if self.electrode_files:
                self.uploaded_files['electrode'] = self.electrode_files[self.primary_electrode_idx]
            elif 'electrode' in self.uploaded_files:
                del self.uploaded_files['electrode']
            self._update_file_list()
            self._update_report_options()
            self._check_pypacer_compatibility()
            if self.upload_widget:
                self.upload_widget.reset()

    def _check_pypacer_compatibility(self):
        """Check if the primary electrode JSON is compatible with the PyPaCER report generator."""
        self.ct_file_available = False
        self.is_full_recon_json = False
        self.is_transformed = False
        electrode_file = self.uploaded_files.get('electrode')
        if electrode_file:
            try:
                with open(electrode_file, 'r') as f:
                    recon_data = json.load(f)
                metadata = recon_data.get('metadata', {})
                # Check CT file availability
                ct_file = metadata.get('ct_file')
                if ct_file and Path(ct_file).exists():
                    self.ct_file_available = True
                # Check if this is a full reconstruction JSON (not mini)
                electrodes = recon_data.get('electrodes', [])
                if electrodes and 'intensity_profile' in electrodes[0]:
                    self.is_full_recon_json = True
                # Check if the data has been transformed to a different coordinate space
                if metadata.get('transformed', False):
                    self.is_transformed = True
            except Exception:
                pass

        has_electrode = 'electrode' in self.uploaded_files
        can_generate = has_electrode and self.is_full_recon_json and not self.is_transformed

        # Update UI elements
        if self.pypacer_ct_warning is not None:
            self.pypacer_ct_warning.set_visibility(
                can_generate and not self.ct_file_available
            )
        if self.pypacer_mini_warning is not None:
            self.pypacer_mini_warning.set_visibility(
                has_electrode and not self.is_full_recon_json
            )
        if self.pypacer_transformed_warning is not None:
            self.pypacer_transformed_warning.set_visibility(
                has_electrode and self.is_full_recon_json and self.is_transformed
            )
        if self.pypacer_generate_btn is not None:
            if can_generate:
                self.pypacer_generate_btn.enable()
            else:
                self.pypacer_generate_btn.disable()

    async def _handle_file_upload(self, e):
        """Handle file upload with automatic type detection."""
        try:
            uploaded_file = e.file
            filename = uploaded_file.name

            # Read file content
            content = await uploaded_file.read()

            # Save using temp file manager
            file_path = save_uploaded_file(content, filename)

            # Detect file type
            file_type = self._detect_file_type(str(file_path), filename)

            if file_type is None:
                ui.notify(f'Could not determine file type for {filename}', type='warning')
                return

            # Handle electrode files specially to support multiple reconstructions
            if file_type == 'electrode':
                if len(self.electrode_files) >= self.MAX_ELECTRODE_FILES:
                    ui.notify(
                        f'Maximum of {self.MAX_ELECTRODE_FILES} electrode files allowed. '
                        'Remove one before adding another.',
                        type='warning'
                    )
                    return
                self.electrode_files.append(str(file_path))
                # If this is the first electrode, set it as primary
                if len(self.electrode_files) == 1:
                    self.primary_electrode_idx = 0
                    self.uploaded_files['electrode'] = str(file_path)
                ui.notify(f'Electrode file {len(self.electrode_files)} loaded', type='positive')
            else:
                # Check if this file type already exists
                if file_type in self.uploaded_files:
                    ui.notify(f'{file_type.capitalize()} file replaced', type='info')
                else:
                    ui.notify(f'{file_type.capitalize()} file loaded', type='positive')

                # Store file path
                self.uploaded_files[file_type] = str(file_path)

            # Auto-fill patient ID from surgical JSON if the input is empty
            if file_type == 'surgical' and self.patient_id_input is not None:
                if not self.patient_id_input.value:
                    try:
                        with open(str(file_path), 'r') as f:
                            data = json.load(f)
                        patient_id = None
                        if isinstance(data, list) and data:
                            patient_id = data[0].get('patient_id')
                        elif isinstance(data, dict):
                            records = data.get('records') or data.get('targets') or data.get('data', [])
                            if records and isinstance(records, list):
                                patient_id = records[0].get('patient_id')
                        if patient_id:
                            self.patient_id_input.value = patient_id
                    except Exception:
                        pass

            # Update displays
            self._update_file_list()
            self._update_report_options()
            if file_type == 'electrode':
                self._check_pypacer_compatibility()

        except Exception as ex:
            ui.notify(f'Error uploading file: {str(ex)}', type='negative')
            import traceback
            traceback.print_exc()

    def _show_overlay(self):
        """Show the generation overlay with spinner."""
        self._status_label.text = 'Preparing...'
        self._overlay.style('display: flex')
        self._status_timer = ui.timer(0.1, self._poll_status)

    def _hide_overlay(self):
        """Hide the generation overlay."""
        self._overlay.style('display: none')
        if self._status_timer:
            self._status_timer.deactivate()
            self._status_timer = None
        # Drain remaining messages
        while not self._status_queue.empty():
            try:
                self._status_queue.get_nowait()
            except queue.Empty:
                break

    def _poll_status(self):
        """Poll the status queue and update the overlay label."""
        latest = None
        while not self._status_queue.empty():
            try:
                latest = self._status_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._status_label.text = latest

    def _run_with_capture(self, fn):
        """Run fn in the current thread while capturing stdout to the status queue."""
        old_stdout = sys.stdout
        sys.stdout = _StdoutCapture(old_stdout, self._status_queue)
        try:
            return fn()
        finally:
            sys.stdout = old_stdout

    async def _generate_stereotactic_report(self):
        """Generate stereotactic targeting report."""
        try:
            surgical_file = self.uploaded_files.get('surgical')
            electrode_file = self.uploaded_files.get('electrode')

            if not surgical_file or not electrode_file:
                ui.notify('Please upload both surgical and electrode files', type='warning')
                return

            import asyncio

            self._show_overlay()
            patient_id = self.patient_id_input.value.strip() if self.patient_id_input else ''

            # Load NIfTI file if available
            nifti_files = []
            nifti_file_path = self.uploaded_files.get('nifti_3d')
            if nifti_file_path:
                try:
                    import nibabel as nib
                    self._status_label.text = 'Loading NIfTI file...'
                    await asyncio.sleep(0.01)  # Allow UI update

                    nifti_img = nib.load(nifti_file_path)
                    nifti_files.append({
                        'data': nifti_img.get_fdata(),
                        'affine': nifti_img.affine,
                        'filename': Path(nifti_file_path).name
                    })
                    print(f"Loaded NIfTI file: {Path(nifti_file_path).name}")
                except Exception as e:
                    print(f"Warning: Failed to load NIfTI file: {e}")

            # Generate report in executor to avoid blocking
            await asyncio.sleep(0.01)  # Allow UI update

            def generate_report():
                # Use primary electrode file for stereotactic report
                primary_electrode_file = self.electrode_files[self.primary_electrode_idx]

                # Get secondary electrode file if available for brain shift analysis
                # Use the first non-primary electrode file
                electrode_json_2 = None
                if len(self.electrode_files) > 1:
                    # Find first electrode file that is not the primary
                    for idx, file_path in enumerate(self.electrode_files):
                        if idx != self.primary_electrode_idx:
                            electrode_json_2 = file_path
                            break

                report_generator = StereotacticReportGenerator.from_json(
                    surgical_json=surgical_file,
                    electrode_json=primary_electrode_file,
                    electrode_json_2=electrode_json_2
                )
                report_generator.nifti_files = nifti_files
                if patient_id:
                    report_generator.patient_id = patient_id
                return report_generator.save_and_download()

            # Run in background thread with stdout capture
            loop = asyncio.get_event_loop()
            temp_path, filename = await loop.run_in_executor(
                None, lambda: self._run_with_capture(generate_report)
            )

            self._hide_overlay()

            # Trigger download
            ui.download(temp_path, filename=filename)
            ui.notify('Stereotactic report generated successfully!', type='positive')

        except Exception as ex:
            self._hide_overlay()
            ui.notify(f'Failed to generate report: {str(ex)}', type='negative')
            import traceback
            traceback.print_exc()

    async def _generate_pypacer_report(self):
        """Generate PyPaCER electrode report."""
        try:
            electrode_file = self.uploaded_files.get('electrode')

            if not electrode_file:
                ui.notify('Please upload an electrode reconstruction file', type='warning')
                return

            import asyncio
            from pypacer.visualization import generate_html_report

            self._show_overlay()

            def generate_report():
                import tempfile
                output_path = Path(tempfile.mktemp(suffix='.html'))
                result_path = generate_html_report(
                    reconstruction_json_path=electrode_file,
                    output_path=str(output_path),
                )
                return result_path

            loop = asyncio.get_event_loop()
            await asyncio.sleep(0.01)

            result_path = await loop.run_in_executor(
                None, lambda: self._run_with_capture(generate_report)
            )

            self._hide_overlay()

            filename = f'{Path(electrode_file).stem}_pypacer_report.html'
            ui.download(result_path, filename=filename)
            ui.notify('PyPaCER report generated successfully!', type='positive')

        except Exception as ex:
            self._hide_overlay()
            ui.notify(f'Failed to generate report: {str(ex)}', type='negative')
            import traceback
            traceback.print_exc()

    async def _generate_stimulation_report(self):
        """Generate stimulation targeting report."""
        try:
            electrode_file = self.uploaded_files.get('electrode')
            nifti_3d_file = self.uploaded_files.get('nifti_3d')
            nifti_4d_file = self.uploaded_files.get('nifti_4d')
            label_file = self.uploaded_files.get('label')
            acpc_file = self.uploaded_files.get('acpc')

            if not electrode_file or (not nifti_3d_file and not nifti_4d_file):
                ui.notify('Please upload electrode and at least one NIfTI file', type='warning')
                return

            import asyncio
            from dbstoolbox.reports import StimulationReportGenerator

            self._show_overlay()

            threshold = self.stimulation_threshold
            patient_id = self.patient_id_input.value.strip() if self.patient_id_input else ''

            def generate_report():
                generator = StimulationReportGenerator.from_json(
                    electrode_json=electrode_file,
                    nifti_3d_path=nifti_3d_file,
                    nifti_4d_path=nifti_4d_file,
                    label_path=label_file,
                    acpc_path=acpc_file,
                    threshold=threshold,
                )
                if patient_id:
                    generator.patient_id = patient_id
                return generator.save_and_download()

            await asyncio.sleep(0.01)

            loop = asyncio.get_event_loop()
            temp_path, filename = await loop.run_in_executor(
                None, lambda: self._run_with_capture(generate_report)
            )

            self._hide_overlay()

            ui.download(temp_path, filename=filename)
            ui.notify('Stimulation report generated successfully!', type='positive')

        except Exception as ex:
            self._hide_overlay()
            ui.notify(f'Failed to generate report: {str(ex)}', type='negative')
            import traceback
            traceback.print_exc()

    def show(self):
        """Show the report generation dialog."""
        # Reset state
        self.uploaded_files.clear()
        self.electrode_files.clear()
        self.primary_electrode_idx = 0
        self.report_options.clear()

        with ui.dialog().props('persistent') as dialog, \
                ui.card().classes('w-full max-w-3xl').style('position: relative'):
            self.dialog = dialog

            # Generation overlay (hidden by default)
            with ui.element('div').style(
                'display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%;'
                ' background: rgba(0,0,0,0.85); z-index: 10; flex-direction: column;'
                ' align-items: center; justify-content: center; border-radius: inherit;'
            ) as self._overlay:
                ui.spinner('dots', size='xl', color='primary')
                self._status_label = ui.label('Preparing...').classes(
                    'text-body2 text-grey-8 mt-4 text-center'
                ).style('max-width: 80%')

            # Header
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label('Generate Reports').classes('text-h6')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense').tooltip('Close')

            ui.separator().classes('mb-4')

            # File upload section
            with ui.column().classes('w-full gap-3 mb-6'):
                self.upload_widget = ui.upload(
                    label='Drag & drop files or click to select',
                    on_upload=self._handle_file_upload,
                    auto_upload=True,
                    multiple=True
                ).classes('w-full').props('accept=".json,.nii,.nii.gz,.txt,.csv" hide-upload-btn')
                # Hide the built-in file list since we have our own custom one
                self.upload_widget._props['no-thumbnails'] = True
                ui.add_css('.q-uploader__list { display: none !important; }')

                # File list display
                ui.label('Loaded Files:').classes('text-body2 font-medium mt-2')
                self.file_list_container = ui.column().classes('w-full gap-2')
                self._update_file_list()

            # Patient ID input
            self.patient_id_input = ui.input(
                label='Patient ID',
                placeholder='Enter patient identifier',
            ).classes('w-full mb-2').props('outlined dense')

            ui.separator().classes('mb-4')

            # Report generation options
            with ui.column().classes('w-full gap-3'):
                ui.label('Available Reports').classes('text-subtitle2 mb-2')

                # Stimulation/Clinical report
                stimulation_card = ui.card().classes('w-full p-4 opacity-60')
                with stimulation_card:
                    with ui.row().classes('w-full items-start gap-4'):
                        ui.icon('view_in_ar', size='lg', color='primary')
                        with ui.column().classes('flex-1 gap-1'):
                            ui.label('Clinical Report').classes('text-h6')
                            ui.label(
                                'NIfTI slice visualization along electrode trajectories showing anatomical context for each contact'
                            ).classes('text-caption text-grey-7')
                            with ui.row().classes('gap-1 mt-2 flex-wrap'):
                                stimulation_badges = self._create_file_badges('stimulation')

                        stimulation_btn = ui.button(
                            'Generate',
                            icon='file_download',
                            on_click=self._generate_stimulation_report
                        ).props('color=primary')
                        stimulation_btn.disable()

                    # Threshold slider (shown when 4D NIfTI is loaded)
                    self.stimulation_threshold_container = ui.row().classes(
                        'w-full items-center gap-2 mt-2'
                    )
                    self.stimulation_threshold_container.set_visibility(False)
                    with self.stimulation_threshold_container:
                        ui.label('Isosurface threshold:').classes('text-caption text-grey-7')
                        self.stimulation_threshold_slider = ui.slider(
                            min=0.05, max=0.95, step=0.05, value=0.5
                        ).props('label-always').classes('flex-1')
                        self.stimulation_threshold_slider.on(
                            'update:model-value',
                            lambda e: setattr(self, 'stimulation_threshold', e.args)
                        )

                self.report_options['stimulation'] = {
                    'card': stimulation_card,
                    'btn': stimulation_btn,
                    'badges': stimulation_badges,
                }

                # Set initial missing-file labels
                self._update_report_options()

                # Stereotactic/Surgical report
                stereotactic_card = ui.card().classes('w-full p-4 opacity-60')
                with stereotactic_card:
                    with ui.row().classes('w-full items-start gap-4'):
                        ui.icon('analytics', size='lg', color='primary')
                        with ui.column().classes('flex-1 gap-1'):
                            ui.label('Surgical Report').classes('text-h6')
                            ui.label(
                                'Comprehensive analysis of electrode positions relative to planned MER tracks at each contact depth'
                            ).classes('text-caption text-grey-7')
                            with ui.row().classes('gap-1 mt-2 flex-wrap'):
                                stereotactic_badges = self._create_file_badges('stereotactic')

                        stereotactic_btn = ui.button(
                            'Generate',
                            icon='file_download',
                            on_click=self._generate_stereotactic_report
                        ).props('color=primary')
                        stereotactic_btn.disable()

                self.report_options['stereotactic'] = {
                    'card': stereotactic_card,
                    'btn': stereotactic_btn,
                    'badges': stereotactic_badges,
                }


                # PyPaCER report
                pypacer_card = ui.card().classes('w-full p-4 opacity-60')
                with pypacer_card:
                    with ui.row().classes('w-full items-start gap-4'):
                        ui.icon('sensors', size='lg', color='primary')
                        with ui.column().classes('flex-1 gap-1'):
                            ui.label('PyPaCER Report').classes('text-h6')
                            ui.label(
                                'Detailed electrode reconstruction summary with trajectory visualization'
                            ).classes('text-caption text-grey-7')
                            with ui.row().classes('gap-1 mt-2 flex-wrap'):
                                pypacer_badges = self._create_file_badges('pypacer')

                        self.pypacer_generate_btn = ui.button(
                            'Generate',
                            icon='file_download',
                            on_click=self._generate_pypacer_report
                        ).props('color=primary')
                        self.pypacer_generate_btn.disable()

                    # Mini JSON warning (shown when electrode JSON is missing full reconstruction data)
                    self.pypacer_mini_warning = ui.row().classes(
                        'w-full items-center gap-2 p-2 bg-red-50 rounded mt-2'
                    )
                    self.pypacer_mini_warning.set_visibility(False)
                    with self.pypacer_mini_warning:
                        ui.icon('error', color='red')
                        ui.label(
                            'This is a mini reconstruction JSON which does not contain the full data '
                            'required for report generation. Please use the full reconstruction JSON.'
                        ).classes('text-caption text-red-900')

                    # Transformed JSON warning (shown when electrode data has been transformed)
                    self.pypacer_transformed_warning = ui.row().classes(
                        'w-full items-center gap-2 p-2 bg-red-50 rounded mt-2'
                    )
                    self.pypacer_transformed_warning.set_visibility(False)
                    with self.pypacer_transformed_warning:
                        ui.icon('error', color='red')
                        ui.label(
                            'This reconstruction has been transformed to a different coordinate space '
                            'and cannot be used for PyPaCER report generation. '
                            'Please use the original (untransformed) reconstruction JSON.'
                        ).classes('text-caption text-red-900')

                    # CT file warning (shown when electrode loaded but CT missing)
                    self.pypacer_ct_warning = ui.row().classes(
                        'w-full items-center gap-2 p-2 bg-orange-50 rounded mt-2'
                    )
                    self.pypacer_ct_warning.set_visibility(False)
                    with self.pypacer_ct_warning:
                        ui.icon('warning', color='orange')
                        ui.label(
                            'CT file referenced in the reconstruction data was not found. '
                            'Some visualizations (e.g. volume rendering) will be skipped.'
                        ).classes('text-caption text-orange-900')

                self.report_options['pypacer'] = {
                    'card': pypacer_card,
                    'btn': self.pypacer_generate_btn,
                    'badges': pypacer_badges,
                }

        dialog.open()


def show_report_generation_dialog():
    """Show the report generation dialog."""
    dialog = ReportGenerationDialog()
    dialog.show()
