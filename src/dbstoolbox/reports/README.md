# DBS Report Generators

Generate comprehensive HTML reports for DBS electrode analysis directly from data files.

Two report types are available:

- **Surgical (Stereotactic) Report** — electrode positions relative to planned MER tracks at each contact depth
- **Clinical (Stimulation) Report** — NIfTI slice visualization along electrode trajectories with optional stimulation volume overlays

## Quick Start

### Surgical Report

```python
from dbstoolbox.reports import StereotacticReportGenerator

report_generator = StereotacticReportGenerator.from_json(
    surgical_json='surgical_plan.json',
    electrode_json='electrode_reconstruction.json'
)
report_generator.patient_id = 'AUH_0001'  # optional, overrides value in JSON

temp_path, filename = report_generator.save_and_download()
```

### Clinical Report

```python
from dbstoolbox.reports import StimulationReportGenerator

generator = StimulationReportGenerator.from_json(
    electrode_json='electrode_reconstruction.json',
    nifti_3d_path='t1.nii.gz',
    nifti_4d_path='probmap.nii.gz',    # optional
    label_path='labels.txt',            # optional
    threshold=0.5,                      # optional
)
generator.patient_id = 'AUH_0001'  # optional

temp_path, filename = generator.save_and_download()
```

### Command Line Usage

```bash
# Surgical report
python examples/generate_surgical_report.py surgical_plan.json recon.json \
    --patient-id AUH_0001 --output report.html

# Clinical report
python examples/generate_clinical_report.py recon.json \
    --nifti-3d t1.nii.gz --patient-id AUH_0001 --output report.html
```

## JSON Data Format

### Surgical Planning Data

Can be either a list of targets or a dict with a 'targets' key:

```json
[
    {
        "patient_id": "AUH_0001",
        "hemisphere": "left",
        "anatomical_target": "STN",
        "x": -12.5,
        "y": -15.2,
        "z": -5.8,
        "entry_x": -28.3,
        "entry_y": 15.4,
        "entry_z": 45.2,
        "mer_anterior_target_x": -12.5,
        "mer_anterior_target_y": -13.2,
        "mer_anterior_target_z": -5.8,
        "mer_anterior_entry_x": -28.3,
        "mer_anterior_entry_y": 17.4,
        "mer_anterior_entry_z": 45.2,
        "mer_lateral_target_x": -14.5,
        "mer_lateral_target_y": -15.2,
        "mer_lateral_target_z": -5.8,
        "mer_lateral_entry_x": -30.3,
        "mer_lateral_entry_y": 15.4,
        "mer_lateral_entry_z": 45.2,
        "ring": 90.0,
        "arc": 70.0
    }
]
```

**Required fields for transformed data (RAS coordinates):**
- `x`, `y`, `z`: Target coordinates
- `entry_x`, `entry_y`, `entry_z`: Entry point coordinates
- `mer_anterior_target_x/y/z`, `mer_anterior_entry_x/y/z`: Anterior MER track
- `mer_lateral_target_x/y/z`, `mer_lateral_entry_x/y/z`: Lateral MER track

**Required fields for untransformed data (Leksell frame):**
- `x`, `y`, `z`: Target coordinates in frame space
- `ring`, `arc`: Frame angles

**Optional fields:**
- `patient_id`: Patient identifier (defaults to "Target N")
- `hemisphere`: "left", "right", "L", or "R" (auto-detected from x coordinate if missing)
- `anatomical_target`: Target name (e.g., "STN", "GPi")

### Electrode Reconstruction Data

```json
{
    "metadata": {
        "ct_file": "/path/to/ct.nii.gz",
        "pypacer_version": "1.0.0",
        "transformed": false
    },
    "electrodes": [
        {
            "trajectory_coordinates": [
                [x1, y1, z1],
                [x2, y2, z2]
            ],
            "contact_positions_3d": [
                [x1, y1, z1],
                [x2, y2, z2],
                [x3, y3, z3],
                [x4, y4, z4]
            ]
        }
    ]
}
```

**Required fields:**
- `electrodes`: List of electrode dictionaries
- `trajectory_coordinates`: List of [x, y, z] points defining the electrode trajectory

**Optional fields:**
- `contact_positions_3d`: List of [x, y, z] positions for each contact (C0-C3)
- `metadata`: Dict with `ct_file`, `pypacer_version`, `transformed`, etc.

## Surgical Report Details

### Features

- Automatic parsing of surgical planning and electrode reconstruction data
- Contact position analysis with interactive polar charts
- MER track reference positions (Central, Anterior, Posterior, Lateral, Medial)
- Statistical tables (distance, AP offset, ML offset per contact)
- Optional brain shift analysis with secondary electrode file
- Optional NIfTI axial slice overlays
- Optional 3D Plotly figure integration

### Advanced Usage

#### Brain Shift Analysis

```python
report_generator = StereotacticReportGenerator.from_json(
    surgical_json='surgical_plan.json',
    electrode_json='recon_pre.json',
    electrode_json_2='recon_post.json',  # secondary reconstruction
)
```

#### With 3D Figure Callback

```python
import plotly.graph_objects as go

def get_3d_figure():
    fig = go.Figure(...)
    return fig

report_generator = StereotacticReportGenerator.from_json(
    surgical_json='surgical_plan.json',
    electrode_json='electrode_reconstruction.json',
    get_3d_figure_callback=get_3d_figure
)
```

## Clinical Report Details

### Features

- Axial NIfTI slice rendering at each electrode contact depth
- 3D isosurface visualization of stimulation probability maps (4D NIfTI)
- Per-volume labels, colors, and visibility from label files
- AC-PC landmark overlay support
- Configurable isosurface threshold

### Input Files

| File | Required | Description |
|------|----------|-------------|
| Electrode JSON | Yes | Reconstruction data with trajectory and contacts |
| 3D NIfTI | At least one NIfTI | Anatomical volume (T1/T2 MRI) for axial slices |
| 4D NIfTI | At least one NIfTI | Probability map for isosurface visualization |
| Label TXT | No | Volume names, colors, and visibility for 4D NIfTI |
| AC-PC CSV | No | AC, PC, and midline landmarks |

## Examples

- [examples/generate_surgical_report.py](../../examples/generate_surgical_report.py) — surgical report from command line
- [examples/generate_clinical_report.py](../../examples/generate_clinical_report.py) — clinical report from command line
