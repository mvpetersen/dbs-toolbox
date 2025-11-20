"""Home page for DBS Toolbox."""

import subprocess
import sys
import threading
import queue
from pathlib import Path
from nicegui import ui
from dbstoolbox.utils.temp_file_manager import save_uploaded_file

# Global process tracker
_running_processes = {
    'PyPaCER': {'active': False, 'output': [], 'process': None, 'output_dir': ''},
    'Leksell': {'active': False, 'output': [], 'process': None, 'output_dir': ''}
}


def home_page(tabs=None):
    """Create the home page content.

    Args:
        tabs: The tabs element for navigation
    """

    # Add CSS for terminal output styling
    ui.add_css('''
        .terminal-output textarea,
        .terminal-output .q-field__native {
            background-color: #1e1e1e !important;
            color: #ff9800 !important;
            font-family: monospace !important;
        }

        /* Override Quasar's native textarea styles */
        .terminal-output textarea.q-field__native {
            background-color: #1e1e1e !important;
            color: #ff9800 !important;
        }

        /* Force color even when readonly */
        .terminal-output textarea[readonly] {
            color: #ff9800 !important;
            opacity: 1 !important;
        }
    ''')

    def open_tab(tab_name):
        """Navigate to the specified tab."""
        if tabs:
            tabs.set_value(tab_name)

    # Terminal icon refs for each tool
    terminal_icons = {'PyPaCER': None, 'Leksell': None}

    def show_terminal_popup(tool_name: str):
        """Show terminal output in a popup dialog."""
        with ui.dialog().props('persistent') as term_dialog, ui.card().classes('w-full max-w-4xl'):
            # Header with title and buttons
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label(f'{tool_name} Terminal Output').classes('text-h6')
                with ui.row().classes('gap-1'):
                    # Output files dropdown button
                    def show_output_files():
                        output_dir = _running_processes[tool_name].get('output_dir', '')
                        if not output_dir or not Path(output_dir).exists():
                            ui.notify('No output directory found', type='warning')
                            return

                        # Find output files
                        output_path = Path(output_dir)
                        json_files = list(output_path.glob('*.json'))
                        html_files = list(output_path.glob('*.html'))

                        if not json_files and not html_files:
                            ui.notify('No output files found yet', type='info')
                            return

                        # Show files in a menu/dialog
                        with ui.dialog() as files_dialog, ui.card():
                            ui.label('Output Files').classes('text-h6 mb-2')
                            ui.label(f'Location: {output_dir}').classes('text-xs text-grey-6 mb-3')

                            with ui.column().classes('gap-1'):
                                # JSON files
                                for f in json_files:
                                    with ui.row().classes('items-center gap-2 w-full'):
                                        ui.icon('description', size='sm')
                                        ui.label(f.name).classes('text-sm flex-1')
                                        ui.label(f'({f.stat().st_size} bytes)').classes('text-xs text-grey-6')
                                        ui.button(icon='download', on_click=lambda fp=str(f): ui.download(fp)).props('flat round dense').tooltip('Download')

                                # HTML files
                                for f in html_files:
                                    with ui.row().classes('items-center gap-2 w-full'):
                                        ui.icon('html', size='sm')
                                        ui.label(f.name).classes('text-sm flex-1')
                                        ui.label(f'({f.stat().st_size} bytes)').classes('text-xs text-grey-6')
                                        ui.button(icon='download', on_click=lambda fp=str(f): ui.download(fp)).props('flat round dense').tooltip('Download')

                            ui.button('Close', on_click=files_dialog.close).props('flat').classes('mt-4')
                        files_dialog.open()

                    ui.button(icon='folder_open', on_click=show_output_files).props('flat round dense').tooltip('View output files')

                    # Kill process button (only shown if process is active)
                    def kill_process():
                        if _running_processes[tool_name]['process']:
                            try:
                                _running_processes[tool_name]['process'].terminate()
                                ui.notify(f'{tool_name} process terminated', type='warning')
                            except Exception as e:
                                ui.notify(f'Error terminating process: {str(e)}', type='negative')
                            _running_processes[tool_name]['active'] = False
                            if terminal_icons[tool_name]:
                                terminal_icons[tool_name].classes(add='hidden')

                    kill_btn = ui.button(icon='stop', on_click=kill_process).props('flat round dense color=negative').tooltip('Kill process')
                    if not _running_processes[tool_name]['active']:
                        kill_btn.classes(add='hidden')

                    ui.button(icon='delete', on_click=lambda: (_running_processes[tool_name]['output'].clear(), popup_textarea.set_value(''))).props('flat round dense').tooltip('Clear output')
                    ui.button(icon='close', on_click=term_dialog.close).props('flat round dense').tooltip('Close')

            # Terminal output textarea
            popup_textarea = ui.textarea('').props('readonly outlined rows=20').classes('w-full font-mono text-sm')
            popup_textarea.style('background-color: #1e1e1e !important; color: #ff9800 !important;')
            popup_textarea.classes('terminal-output')

            # Set initial content
            popup_textarea.set_value('\n'.join(_running_processes[tool_name]['output']))

            # Auto-scroll to bottom
            ui.run_javascript(f'''
                setTimeout(() => {{
                    const el = document.getElementById("{popup_textarea.id}");
                    if (el) {{
                        const textarea = el.querySelector('textarea');
                        if (textarea) {{
                            textarea.scrollTop = textarea.scrollHeight;
                        }}
                    }}
                }}, 100);
            ''')

            # Update timer to refresh output
            def update_popup():
                # Update output
                popup_textarea.set_value('\n'.join(_running_processes[tool_name]['output']))
                ui.run_javascript(f'''
                    const el = document.getElementById("{popup_textarea.id}");
                    if (el) {{
                        const textarea = el.querySelector('textarea');
                        if (textarea) {{
                            textarea.scrollTop = textarea.scrollHeight;
                        }}
                    }}
                ''')

                # Update kill button visibility
                if _running_processes[tool_name]['active']:
                    kill_btn.classes(remove='hidden')
                else:
                    kill_btn.classes(add='hidden')

            ui.timer(0.5, update_popup)

        term_dialog.open()

    def auto_run_cli(tool_name: str, file_path: str, output_dir: str):
        """Run CLI tool automatically (no GUI) and capture output to terminal."""
        import os

        # Check if already running
        if _running_processes[tool_name]['active']:
            ui.notify(f'{tool_name} is already running. Check the terminal icon to view output.', type='warning')
            return

        # Mark process as active
        _running_processes[tool_name]['active'] = True
        _running_processes[tool_name]['output'] = []
        _running_processes[tool_name]['output_dir'] = output_dir

        # Show terminal icon if available
        if terminal_icons[tool_name]:
            terminal_icons[tool_name].classes(remove='hidden')

        # Build command for CLI (automatic processing)
        if tool_name == 'PyPaCER':
            cmd = [sys.executable, '-m', 'pypacer.cli.pypacer', file_path, '--output-dir', output_dir, '--html']
        elif tool_name == 'Leksell':
            # Leksell doesn't have CLI auto-run, fallback to GUI
            cmd = [sys.executable, '-m', 'leksell_frame_registration.gui.matplotlib_gui', file_path, '--output-dir', output_dir]
        else:
            return

        try:
            # Launch process with output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )

            _running_processes[tool_name]['process'] = process

            # Create queue for thread-safe output handling
            output_queue = queue.Queue()

            def read_output_thread():
                """Read output in background thread."""
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_queue.put(line)
                    output_queue.put(None)  # Signal end
                except Exception as e:
                    output_queue.put(f"Error reading output: {str(e)}\n")
                    output_queue.put(None)

            # Start reader thread
            reader_thread = threading.Thread(target=read_output_thread, daemon=True)
            reader_thread.start()

            # Poll queue and update UI
            def check_output():
                """Check for new output."""
                try:
                    while not output_queue.empty():
                        line = output_queue.get_nowait()
                        if line is None:  # End of stream
                            _running_processes[tool_name]['active'] = False
                            if terminal_icons[tool_name]:
                                terminal_icons[tool_name].classes(add='hidden')
                            ui.notify(f'{tool_name} auto-run completed. Check terminal for results.', type='positive')
                            return False
                        if line:
                            # Store in process tracker
                            _running_processes[tool_name]['output'].append(line.rstrip())
                            # Keep last 1000 lines
                            if len(_running_processes[tool_name]['output']) > 1000:
                                _running_processes[tool_name]['output'] = _running_processes[tool_name]['output'][-1000:]

                            # Print to console (terminal visibility)
                            print(f"[{tool_name}] {line}", end='')
                except queue.Empty:
                    pass
                return True  # Continue polling

            # Start polling timer
            ui.timer(0.1, check_output)

            ui.notify(f'{tool_name} auto-run started (no GUI)', type='positive')
        except Exception as e:
            _running_processes[tool_name]['active'] = False
            if terminal_icons[tool_name]:
                terminal_icons[tool_name].classes(add='hidden')
            ui.notify(f'Error launching CLI: {str(e)}', type='negative')
            print(f"Error launching {tool_name} CLI: {str(e)}")

    def launch_gui_with_output(tool_name: str, file_path: str, output_dir: str):
        """Launch GUI and capture output to both terminal and UI."""
        import os

        # Check if already running
        if _running_processes[tool_name]['active']:
            ui.notify(f'{tool_name} is already running. Check the terminal icon to view output.', type='warning')
            return

        # Mark process as active
        _running_processes[tool_name]['active'] = True
        _running_processes[tool_name]['output'] = []
        _running_processes[tool_name]['output_dir'] = output_dir

        # Show terminal icon if available
        if terminal_icons[tool_name]:
            terminal_icons[tool_name].classes(remove='hidden')

        # Build command
        if tool_name == 'PyPaCER':
            cmd = [sys.executable, '-m', 'pypacer.cli.pypacer_gui', file_path, '--output-dir', output_dir]
        elif tool_name == 'Leksell':
            cmd = [sys.executable, '-m', 'leksell_frame_registration.gui.matplotlib_gui', file_path, '--output-dir', output_dir]
        else:
            return

        try:
            # Launch process with output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )

            _running_processes[tool_name]['process'] = process

            # Create queue for thread-safe output handling
            output_queue = queue.Queue()

            def read_output_thread():
                """Read output in background thread."""
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_queue.put(line)
                    output_queue.put(None)  # Signal end
                except Exception as e:
                    output_queue.put(f"Error reading output: {str(e)}\n")
                    output_queue.put(None)

            # Start reader thread
            reader_thread = threading.Thread(target=read_output_thread, daemon=True)
            reader_thread.start()

            # Poll queue and update UI
            def check_output():
                """Check for new output."""
                try:
                    while not output_queue.empty():
                        line = output_queue.get_nowait()
                        if line is None:  # End of stream
                            _running_processes[tool_name]['active'] = False
                            if terminal_icons[tool_name]:
                                terminal_icons[tool_name].classes(add='hidden')
                            return False
                        if line:
                            # Store in process tracker
                            _running_processes[tool_name]['output'].append(line.rstrip())
                            # Keep last 1000 lines
                            if len(_running_processes[tool_name]['output']) > 1000:
                                _running_processes[tool_name]['output'] = _running_processes[tool_name]['output'][-1000:]

                            # Print to console (terminal visibility)
                            print(f"[{tool_name}] {line}", end='')
                except queue.Empty:
                    pass
                return True  # Continue polling

            # Start polling timer
            ui.timer(0.1, check_output)

            ui.notify(f'{tool_name} legacy GUI launched', type='positive')
        except Exception as e:
            _running_processes[tool_name]['active'] = False
            if terminal_icons[tool_name]:
                terminal_icons[tool_name].classes(add='hidden')
            ui.notify(f'Error launching GUI: {str(e)}', type='negative')
            print(f"Error launching {tool_name}: {str(e)}")

    def show_legacy_gui_dialog(tool_name: str):
        """Show dialog to load NIfTI file and launch legacy GUI."""
        uploaded_file_path = None
        temp_session_dir = None

        # Check if already running
        if _running_processes[tool_name]['active']:
            ui.notify(f'{tool_name} is already running. Click the terminal icon to view output.', type='warning')
            return

        with ui.dialog().props('persistent') as dialog, ui.card().classes('w-full max-w-md'):
            # Header with title and close button
            with ui.row().classes('w-full items-center justify-between mb-4'):
                ui.label(f'Launch {tool_name} Legacy GUI').classes('text-h6')
                ui.button(icon='close', on_click=dialog.close).props('flat round dense').tooltip('Close')

            # File upload
            file_status = ui.label('No file selected').classes('text-caption text-grey-7')

            async def handle_upload(e):
                nonlocal uploaded_file_path, temp_session_dir
                try:
                    # Get file object from upload event
                    uploaded_file = e.file
                    filename = uploaded_file.name

                    # Validate file extension
                    if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
                        ui.notify('Please upload a .nii or .nii.gz file', type='warning')
                        return

                    # Read file content
                    content = await uploaded_file.read()

                    # Save using temp file manager
                    file_path = save_uploaded_file(content, filename)
                    uploaded_file_path = str(file_path)

                    # Keep uploaded file in temp (will be cleared on reboot)
                    # But set output directory to user's home directory with timestamp
                    # Format: ~/dbstoolbox/YYYYMMDD_HHMMSS/pypacer or leksell-reg
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tool_subdir = 'pypacer' if tool_name == 'PyPaCER' else 'leksell-reg'
                    default_output_dir = Path.home() / 'dbstoolbox' / timestamp / tool_subdir
                    temp_session_dir = str(default_output_dir)
                    output_dir_input.value = temp_session_dir

                    file_status.text = f'File: {filename}'
                    file_status.classes('text-caption text-primary')
                    launch_button.enable()
                    # Only enable auto run button if it exists (PyPaCER only)
                    if tool_name == 'PyPaCER':
                        auto_run_button.enable()
                        # Change to red when enabled
                        auto_run_button.classes(remove='text-grey', add='text-red')
                except Exception as ex:
                    ui.notify(f'Error uploading file: {str(ex)}', type='negative')

            ui.upload(
                label='Drag & drop or click to select NIfTI file',
                on_upload=handle_upload,
                auto_upload=True
            ).classes('w-full').props('accept=".nii,.nii.gz"')

            # Output directory
            ui.label('Output Directory').classes('text-subtitle2 mt-4 mb-2')

            # Info message about default location
            with ui.row().classes('w-full items-start gap-2 p-2 bg-blue-50 rounded border border-blue-200 mb-2'):
                ui.icon('info', color='blue').classes('mt-0.5')
                with ui.column().classes('gap-1'):
                    ui.label('Uploaded files stored in /tmp (cleared on reboot)').classes('text-caption text-blue-900 font-semibold')
                    ui.label('Output results saved to ~/dbstoolbox/[timestamp]/pypacer or leksell-reg').classes('text-caption text-blue-800')

            # Create output directory input (will be populated when file is uploaded)
            output_dir_input = ui.input(
                placeholder='Will be set when file is uploaded'
            ).classes('w-full')

            # Button handlers
            def launch_gui():
                if not uploaded_file_path:
                    ui.notify('Please select a NIfTI file first', type='warning')
                    return

                # Use output directory from input (or temp dir if not changed)
                output_dir = Path(output_dir_input.value.strip()) if output_dir_input.value else Path(temp_session_dir)

                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    ui.notify(f'Error creating output directory: {str(e)}', type='negative')
                    return

                launch_gui_with_output(tool_name, uploaded_file_path, str(output_dir))
                dialog.close()

            def run_auto():
                if not uploaded_file_path:
                    ui.notify('Please select a NIfTI file first', type='warning')
                    return

                # Use output directory from input (or temp dir if not changed)
                output_dir = Path(output_dir_input.value.strip()) if output_dir_input.value else Path(temp_session_dir)

                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    ui.notify(f'Error creating output directory: {str(e)}', type='negative')
                    return

                auto_run_cli(tool_name, uploaded_file_path, str(output_dir))
                dialog.close()

            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                # Only show auto run button for PyPaCER
                if tool_name == 'PyPaCER':
                    auto_run_button = ui.button('Auto Run', icon='play_arrow', on_click=run_auto).props('outline').classes('text-grey')
                    auto_run_button.disable()  # Only enabled after file upload
                launch_button = ui.button('Launch GUI', icon='open_in_new', on_click=launch_gui).props('color=primary')
                launch_button.disable()  # Only enabled after file upload

        dialog.open()

    with ui.column().classes('w-full max-w-6xl mx-auto p-8'):
        # Hero section
        with ui.card().classes('w-full p-8 mb-8'):
            ui.label('Welcome to The DBS Toolbox').classes('text-h3 text-weight-bold mb-4')
            ui.label(
                'A modern, unified interface for custom deep brain stimulation imaging tools. '
                'This application provides easy access to PyPaCER for electrode reconstruction, '
                'Leksell Frame Registration for handling stereotactic data, and various utility tools.'
            ).classes('text-body1 text-grey-8')

            # GitHub link
            with ui.row().classes('absolute gap-2 mt-0 right-8 bottom-2 flex'):
                ui.icon('code', size='sm').classes('text-grey-6')
                ui.link('View on GitHub', 'https://github.com/mvpetersen/dbs-toolbox', new_tab=True).classes('text-sm text-grey-7')

        # Feature cards
        with ui.row().classes('w-full gap-4'):
            # PyPaCER card - legacy GUI available
            with ui.card().classes('col p-6 relative').style('min-height: 260px'):
                with ui.column().classes('items-center text-center gap-2'):
                    ui.icon('sensors', size='xl', color='primary')
                    ui.label('PyPaCER').classes('text-h6 mt-2')
                    ui.label(
                        'Automatic detection and reconstruction of DBS electrodes from post-operative CT imaging'
                    ).classes('text-body2 text-grey-7')
                    ui.badge('Legacy GUI available', color='blue').classes('mt-2')

                    # GitHub link
                    with ui.row().classes('gap-2 mt-4 items-center').on('click.stop', lambda e: None):
                        ui.icon('code', size='sm').classes('text-grey-6')
                        ui.link('View on GitHub', 'https://github.com/mvpetersen/pypacer', new_tab=True).classes('text-sm text-grey-7')

                # Buttons in lower right - always available
                with ui.element('div').classes('absolute bottom-2 right-2 flex gap-1'):
                    # Terminal icon (hidden by default, shown when process is running)
                    pypacer_term_icon = ui.button(icon='terminal',
                             on_click=lambda: show_terminal_popup('PyPaCER')
                            ).props('flat round dense color=orange').tooltip('View terminal output').classes('hidden')
                    terminal_icons['PyPaCER'] = pypacer_term_icon

                    # Legacy UI button
                    ui.button(icon='open_in_new',
                             on_click=lambda: show_legacy_gui_dialog('PyPaCER')
                            ).props('flat round dense').tooltip('Open legacy GUI')

            # Leksell card - legacy GUI available
            with ui.card().classes('col p-6 relative').style('min-height: 260px'):
                with ui.column().classes('items-center text-center gap-2'):
                    ui.icon('grid_on', size='xl', color='primary')
                    ui.label('Leksell Registration').classes('text-h6 mt-2')
                    ui.label(
                        'Fiducial detection and frame registration for Leksell stereotactic data'
                    ).classes('text-body2 text-grey-7')
                    ui.badge('Legacy GUI available', color='blue').classes('mt-2')

                    # GitHub link
                    with ui.row().classes('gap-2 mt-4 items-center').on('click.stop', lambda e: None):
                        ui.icon('code', size='sm').classes('text-grey-6')
                        ui.link('View on GitHub', 'https://github.com/mvpetersen/leksell-frame-registration', new_tab=True).classes('text-sm text-grey-7')

                # Buttons in lower right - always available
                with ui.element('div').classes('absolute bottom-2 right-2 flex gap-1'):
                    # Terminal icon (hidden by default, shown when process is running)
                    leksell_term_icon = ui.button(icon='terminal',
                             on_click=lambda: show_terminal_popup('Leksell')
                            ).props('flat round dense color=orange').tooltip('View terminal output').classes('hidden')
                    terminal_icons['Leksell'] = leksell_term_icon

                    # Legacy UI button
                    ui.button(icon='open_in_new',
                             on_click=lambda: show_legacy_gui_dialog('Leksell')
                            ).props('flat round dense').tooltip('Open legacy GUI')

            # Utils card
            with ui.card().classes('col p-6').style('min-height: 260px'):
                with ui.column().classes('items-center text-center gap-2'):
                    ui.icon('handyman', size='xl', color='primary')
                    ui.label('Utility Tools').classes('text-h6 mt-2')
                    ui.label(
                        'Coordinate transformation and data visualization for DBS workflows'
                    ).classes('text-body2 text-grey-7')
                    # Shortcut buttons
                    with ui.row().classes('gap-2 mt-4'):
                        ui.button('Transform', icon='grid_off', on_click=lambda: open_tab('transform')).props('outline color=primary')
                        ui.button('Visualize', icon='visibility', on_click=lambda: open_tab('visualize')).props('outline color=primary')