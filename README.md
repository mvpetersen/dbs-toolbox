# The DBS Toolbox

A unified launcher and toolbox for deep brain stimulation (DBS) imaging workflows, built with NiceGUI.

## Overview

The DBS Toolbox provides a modern web-based interface for essential DBS neurosurgical tools:
- **PyPaCER**: Automatic detection and reconstruction of DBS electrodes from post-operative CT imaging
- **Leksell Frame Registration**: Fiducial detection and frame registration for stereotactic data
- **Utility Tools**: Coordinate transformation and data visualization for DBS workflows

## Features

- 🌐 Modern web-based interface accessible from any browser
- 🚀 One-click launchers for PyPaCER and Leksell Frame Registration GUIs
- 🔧 Coordinate transformation tools for stereotactic data
- 📊 Interactive 3D visualization of medical imaging data
- 💾 Support for NIFTI files

## Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/mvpetersen/dbstoolbox.git
cd dbstoolbox

# Install dependencies
uv sync
```

### Dependencies

**Core dependencies:**
- NiceGUI (>=2.0.0) - Web UI framework
- NumPy, Nibabel - Medical image processing
- Plotly - 3D visualization
- ANTsPyx - Image registration and transformation
- PyPaCER (from GitHub) - DBS electrode detection
- Leksell Frame Registration (from GitHub) - Stereotactic frame registration

## Usage

### Running the Application

```bash
# Start the application
uv run dbstoolbox

# The application will open at http://localhost:8090
```

### Available Options

**`--native`** - Run as a native desktop application (requires: `uv sync --extra native`)
```bash
uv run dbstoolbox --native
```

**`--reload`** - Enable hot reload for development (auto-refresh on code changes)
```bash
uv run dbstoolbox --reload
```

### Using the Tools

#### PyPaCER (Electrode Detection)
1. From the home page, click "Open legacy GUI" on the PyPaCER card
2. The PyPaCER interface will launch in a new window
3. Follow the PyPaCER workflow for electrode detection

#### Leksell Frame Registration
1. From the home page, click "Open legacy GUI" on the Leksell Registration card
2. The Leksell interface will launch in a new window
3. Follow the Leksell workflow for frame registration

#### Coordinate Transformation
1. Navigate to the "Transform" tab
2. Upload your coordinate data and transformation files
3. Select transformation parameters
4. Export transformed coordinates

#### Data Visualization
1. Navigate to the "Visualize" tab
2. Upload your NIfTI files
3. Use interactive 3D visualization tools

### Programmatic Usage

For advanced users who want to use the transformation utilities in their own scripts, see the [examples/](examples/) directory:

- **[transform_surgical_csv.py](examples/transform_surgical_csv.py)** - Transform surgical planning data from stereotactic space to T1W
- **[transform_electrode_json.py](examples/transform_electrode_json.py)** - Transform electrode reconstructions from postop CT to T1W (via preop CT)

See the [examples README](examples/README.md) for detailed usage instructions and API documentation.

## Project Structure

```
dbstoolbox/
├── src/
│   └── dbstoolbox/
│       ├── main.py                  # Application entry point
│       ├── pages/                   # UI pages
│       │   ├── home.py             # Landing page with GUI launchers
│       │   ├── transform_simple.py # Coordinate transformation
│       │   └── utils.py            # Visualization tools
│       ├── components/             # Reusable UI components
│       │   ├── file_upload.py
│       │   └── plotly_3d.py
│       └── utils/                  # Utility functions
│           ├── transform_coordinates.py
│           └── validate_*.py
├── examples/                       # Usage examples
│   ├── transform_surgical_csv.py
│   ├── transform_electrode_json.py
│   ├── data/                       # Sample data
│   └── README.md
├── pyproject.toml
└── README.md
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/mvpetersen/dbstoolbox.git
cd dbstoolbox

# Install with development tools
uv sync --extra dev
```

### Running in Development Mode

```bash
# Run with hot reload (recommended for development)
uv run dbstoolbox --reload

# Test native mode
uv run dbstoolbox --native
```


## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- PyPaCER by Mikkel V. Petersen
- Leksell Frame Registration by Mikkel V. Petersen
- Built with NiceGUI by Zauberzeug GmbH

