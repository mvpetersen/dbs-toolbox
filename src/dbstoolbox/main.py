"""Main entry point for DBS Toolbox."""

from nicegui import ui, app
from pathlib import Path
import shutil
import sys

from dbstoolbox.pages.home import home_page
from dbstoolbox.pages.transform_simple import SimpleTransformPage
from dbstoolbox.pages.utils import VisualizePage

# Temp directory will be initialized on startup
TEMP_DIR = None

def cleanup_temp_dir():
    """Clean up temporary directory."""
    if TEMP_DIR and TEMP_DIR.exists():
        print(f"Cleaning up temp directory: {TEMP_DIR}")
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception as e:
            print(f"Error cleaning temp directory: {e}")

def cleanup_old_temp_dirs(current_temp_dir=None):
    """Clean up old temp directories from previous runs."""
    import tempfile
    temp_root = Path(tempfile.gettempdir())

    # Find and remove old dbstoolbox temp directories
    old_dirs = list(temp_root.glob('dbstoolbox_*'))
    if old_dirs:
        print(f"Found {len(old_dirs)} old temp directories to clean up")
        for old_dir in old_dirs:
            if old_dir != current_temp_dir and old_dir.is_dir():
                try:
                    shutil.rmtree(old_dir)
                    print(f"  Removed: {old_dir}")
                except Exception as e:
                    print(f"  Failed to remove {old_dir}: {e}")

# Periodic cleanup of old files (files older than 1 hour)
async def periodic_cleanup():
    """Periodically clean up old temp files and session directories."""
    import asyncio
    import time
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        try:
            if not TEMP_DIR:
                continue

            current_time = time.time()

            # Clean up files and directories in TEMP_DIR
            for item in TEMP_DIR.iterdir():
                try:
                    item_age = current_time - item.stat().st_mtime

                    if item_age > 3600:  # 1 hour
                        if item.is_file():
                            item.unlink()
                            print(f"Cleaned up old file: {item.name} (age: {item_age/60:.1f} minutes)")
                        elif item.is_dir():
                            # Clean up session directory
                            shutil.rmtree(item)
                            print(f"Cleaned up old session directory: {item.name} (age: {item_age/60:.1f} minutes)")
                except Exception as e:
                    print(f"Failed to clean up {item}: {e}")

        except Exception as e:
            print(f"Error in periodic cleanup: {e}")


def initialize_app():
    """Initialize app resources (temp dirs, static files, etc.)."""
    global TEMP_DIR
    import tempfile
    import atexit
    import asyncio
    from dbstoolbox.utils.temp_file_manager import set_temp_dir

    # Create a temporary directory for serving files
    TEMP_DIR = Path(tempfile.mkdtemp(prefix='dbstoolbox_'))
    app.add_static_files('/temp_files', TEMP_DIR)

    # Initialize temp file manager
    set_temp_dir(TEMP_DIR)

    # Configure native window settings
    app.native.window_args = {
        'min_size': (1300, 900),  # Minimum window size
        'resizable': True,
    }

    # Clean up old directories
    cleanup_old_temp_dirs(TEMP_DIR)

    # Register cleanup on app shutdown
    atexit.register(cleanup_temp_dir)

    # Start periodic cleanup task
    app.on_startup(lambda: asyncio.create_task(periodic_cleanup()))

    # Add static files directory
    STATIC_DIR = Path(__file__).parent / 'static'
    STATIC_DIR.mkdir(exist_ok=True)
    app.add_static_files('/static', STATIC_DIR)


def setup_theme():
    """Configure the application theme."""
    ui.colors(primary='#1976D2', secondary='#424242', accent='#82B1FF')
    
    # Add custom CSS
    ui.add_head_html('''
        <style>
        .nicegui-content {
            padding: 0;
        }
        </style>
    ''')


@ui.page('/')
def index():
    """Main page of the application."""
    # Setup
    setup_theme()

    # Create page instances
    transform_page = SimpleTransformPage()
    visualize_page = VisualizePage()

    # Create header
    with ui.header().classes('items-center justify-between'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('hub', size='lg')
            ui.label('The DBS Toolbox').classes('text-h6 q-ml-sm')

        # Center section for active tool indicator
        active_tool_label = ui.label('').classes('text-h6')

        tabs = ui.tabs()
        with tabs:
            ui.tab('home', label='Home', icon='home')
            ui.tab('transform', label='Transform', icon='grid_off')
            ui.tab('visualize', label='Visualize', icon='visibility')

    # Update active tool label based on tab
    def update_active_tool():
        # Update active tool label
        tool_names = {
            'transform': 'Transform Coordinates',
            'visualize': 'Visualize Data',
        }
        active_tool_label.text = tool_names.get(tabs.value, '')

    tabs.on_value_change(update_active_tool)

    # Create tab panels (animated=False prevents transition issues)
    with ui.tab_panels(tabs, value='home', animated=False).classes('w-full') as tab_panels:
        with ui.tab_panel('home'):
            home_page(tabs)

        with ui.tab_panel('transform'):
            transform_page.create_ui(visualize_page, tabs)

        with ui.tab_panel('visualize'):
            visualize_page.create_ui()


# Initialize app resources at module level (required for reload mode)
initialize_app()


def main():
    """Main application entry point - only used when NOT in reload mode."""
    import os

    # Check command-line flags and environment variables
    # Native mode: dbstoolbox --native  OR  NICEGUI_NATIVE=true dbstoolbox
    native_mode = '--native' in sys.argv or os.environ.get('NICEGUI_NATIVE', '').lower() == 'true'

    # Print mode info
    print("🔧 Starting DBS Toolbox")
    print("   Features: Coordinate transformation and visualization tools")

    # Print additional mode info
    if native_mode:
        print("   Running in native window mode (1300x900 minimum size)")

    # Configure native window settings if in native mode
    run_kwargs = {
        'title': 'DBS Toolbox',
        'port': 8090,
        'reload': False,  # Reload is handled separately
        'dark': None,  # Auto dark mode
        'show': True,  # Auto-open browser
        'native': native_mode,  # Run as native window (requires: uv sync --extra native)
    }

    # Add window size configuration for native mode
    if native_mode:
        run_kwargs['window_size'] = (1300, 900)  # Initial window size
        # Note: min_size is set via app configuration

    ui.run(**run_kwargs)


# Handle reload mode vs normal mode
if '--reload' in sys.argv:
    # Reload mode: ui.run() must be called at module level
    import os

    # Print mode info
    print("🔧 Starting DBS Toolbox")
    print("   Features: Coordinate transformation and visualization tools")
    print("   Hot reload enabled - code changes will auto-refresh")

    native_mode = '--native' in sys.argv or os.environ.get('NICEGUI_NATIVE', '').lower() == 'true'
    if native_mode:
        print("   Running in native window mode (1300x900 minimum size)")

    run_kwargs = {
        'title': 'DBS Toolbox',
        'port': 8090,
        'reload': True,
        'dark': None,
        'show': True,
        'native': native_mode,
    }

    if native_mode:
        run_kwargs['window_size'] = (1300, 900)

    ui.run(**run_kwargs)
elif __name__ in {"__main__", "__mp_main__"}:
    # Normal mode: call main() function
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        # Cleanup is already registered with atexit, so it will be called automatically
        pass