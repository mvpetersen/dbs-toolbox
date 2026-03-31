"""
Example: Generate a surgical (stereotactic) targeting report from data files.

This example shows how to use the StereotacticReportGenerator independently
of the visualization UI, allowing you to create reports from command line.

The StereotacticReportGenerator.from_json() method handles all the parsing
and pre-calculation automatically.

Usage:
    python generate_surgical_report.py <surgical_file> <electrode_file> [options]

Required:
    surgical_file       Path to surgical planning JSON
    electrode_file      Path to primary electrode reconstruction JSON

Options:
    --electrode-2 FILE  Secondary electrode JSON (for brain shift analysis)
    --nifti-3d FILE     3D NIfTI volume (e.g. T1/T2 MRI)
    --patient-id ID     Patient identifier (overrides value from surgical JSON)
    --output FILE       Output path (default: auto-generated in temp dir)

Examples:
    # Minimal
    python generate_surgical_report.py surgical.json recon.json

    # With brain shift comparison and patient ID override
    python generate_surgical_report.py surgical.json recon_pre.json \\
        --electrode-2 recon_post.json \\
        --nifti-3d t1.nii.gz \\
        --patient-id AUH_0001 \\
        --output report.html
"""

import argparse
import sys

from dbstoolbox.reports import StereotacticReportGenerator


def generate_report(
    surgical_file,
    electrode_file,
    electrode_file_2=None,
    nifti_3d_path=None,
    patient_id='',
    output_path=None,
):
    """
    Generate a stereotactic targeting report from file paths.

    Args:
        surgical_file: Path to surgical planning JSON
        electrode_file: Path to primary electrode reconstruction JSON
        electrode_file_2: Optional secondary electrode JSON (brain shift)
        nifti_3d_path: Optional path to 3D NIfTI volume
        patient_id: Optional patient identifier (overrides surgical JSON value)
        output_path: Optional output path (if None, saves to temp directory)

    Returns:
        Path to generated HTML report
    """
    report_generator = StereotacticReportGenerator.from_json(
        surgical_json=surgical_file,
        electrode_json=electrode_file,
        electrode_json_2=electrode_file_2,
    )

    if patient_id:
        report_generator.patient_id = patient_id

    # Load NIfTI if provided
    if nifti_3d_path:
        import nibabel as nib
        from pathlib import Path

        print(f"Loading NIfTI file: {nifti_3d_path}")
        img = nib.load(nifti_3d_path)
        report_generator.nifti_files = [{
            'data': img.get_fdata(),
            'affine': img.affine,
            'filename': Path(nifti_3d_path).name,
        }]

    if output_path:
        html_content = report_generator.generate_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report saved to: {output_path}")
        return output_path
    else:
        temp_path, filename = report_generator.save_and_download()
        print(f"Report saved to: {temp_path}")
        return temp_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a DBS surgical (stereotactic) targeting report from data files."
    )
    parser.add_argument("surgical_file", help="Path to surgical planning JSON")
    parser.add_argument("electrode_file", help="Path to primary electrode reconstruction JSON")
    parser.add_argument("--electrode-2", dest="electrode_2", help="Secondary electrode JSON (brain shift analysis)")
    parser.add_argument("--nifti-3d", dest="nifti_3d", help="3D NIfTI volume (e.g. T1/T2 MRI)")
    parser.add_argument("--patient-id", dest="patient_id", default="", help="Patient identifier (overrides surgical JSON)")
    parser.add_argument("--output", help="Output HTML path (default: auto-generated)")

    args = parser.parse_args()

    print("Generating surgical report from:")
    print(f"  Surgical data:  {args.surgical_file}")
    print(f"  Electrode data: {args.electrode_file}")
    if args.electrode_2:
        print(f"  Electrode 2:    {args.electrode_2}")
    if args.nifti_3d:
        print(f"  3D NIfTI:       {args.nifti_3d}")
    if args.patient_id:
        print(f"  Patient ID:     {args.patient_id}")

    try:
        report_path = generate_report(
            surgical_file=args.surgical_file,
            electrode_file=args.electrode_file,
            electrode_file_2=args.electrode_2,
            nifti_3d_path=args.nifti_3d,
            patient_id=args.patient_id,
            output_path=args.output,
        )
        print(f"\nSuccess! Report generated: {report_path}")
    except Exception as e:
        print(f"\nError generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
