"""
Example: Generate a clinical (stimulation) report directly from files.

This example shows how to use the StimulationReportGenerator independently
of the visualization UI, allowing you to create reports from command line.

The StimulationReportGenerator.from_json() method handles all the parsing
automatically, making it easy to generate reports from raw data files.

Usage:
    python generate_clinical_report.py <electrode_file> [options]

Required:
    electrode_file      Path to electrode reconstruction JSON

Options:
    --nifti-3d FILE     3D NIfTI volume (e.g. T1/T2 MRI)
    --nifti-4d FILE     4D NIfTI probability map
    --labels FILE       Volume label file (TXT) for 4D NIfTI
    --acpc FILE         AC-PC landmarks CSV
    --patient-id ID     Patient identifier for the report header
    --threshold VAL     Isosurface threshold for 4D volumes (default: 0.5)
    --output FILE       Output path (default: auto-generated in temp dir)

Examples:
    # Minimal: electrode + 3D NIfTI
    python generate_clinical_report.py recon.json --nifti-3d t1.nii.gz

    # Full: all inputs with patient ID
    python generate_clinical_report.py recon.json \\
        --nifti-3d t1.nii.gz \\
        --nifti-4d probmap.nii.gz \\
        --labels labels.txt \\
        --patient-id AUH_0001 \\
        --threshold 0.3 \\
        --output report.html
"""

import argparse
import sys

from dbstoolbox.reports import StimulationReportGenerator


def generate_report(
    electrode_file,
    nifti_3d_path=None,
    nifti_4d_path=None,
    label_path=None,
    acpc_path=None,
    patient_id='',
    threshold=0.5,
    output_path=None,
):
    """
    Generate a clinical report from file paths.

    Args:
        electrode_file: Path to electrode reconstruction JSON
        nifti_3d_path: Optional path to 3D NIfTI volume
        nifti_4d_path: Optional path to 4D NIfTI probability map
        label_path: Optional path to volume label TXT file
        acpc_path: Optional path to AC-PC landmarks CSV
        patient_id: Optional patient identifier for the report header
        threshold: Isosurface threshold for 4D volumes (default: 0.5)
        output_path: Optional output path (if None, saves to temp directory)

    Returns:
        Path to generated HTML report
    """
    generator = StimulationReportGenerator.from_json(
        electrode_json=electrode_file,
        nifti_3d_path=nifti_3d_path,
        nifti_4d_path=nifti_4d_path,
        label_path=label_path,
        acpc_path=acpc_path,
        threshold=threshold,
    )

    if patient_id:
        generator.patient_id = patient_id

    if output_path:
        html_content = generator.generate_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report saved to: {output_path}")
        return output_path
    else:
        temp_path, filename = generator.save_and_download()
        print(f"Report saved to: {temp_path}")
        return temp_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a DBS clinical (stimulation) report from data files."
    )
    parser.add_argument("electrode_file", help="Path to electrode reconstruction JSON")
    parser.add_argument("--nifti-3d", dest="nifti_3d", help="3D NIfTI volume (e.g. T1/T2 MRI)")
    parser.add_argument("--nifti-4d", dest="nifti_4d", help="4D NIfTI probability map")
    parser.add_argument("--labels", help="Volume label file (TXT) for 4D NIfTI")
    parser.add_argument("--acpc", help="AC-PC landmarks CSV")
    parser.add_argument("--patient-id", dest="patient_id", default="", help="Patient identifier")
    parser.add_argument("--threshold", type=float, default=0.5, help="Isosurface threshold (default: 0.5)")
    parser.add_argument("--output", help="Output HTML path (default: auto-generated)")

    args = parser.parse_args()

    if not args.nifti_3d and not args.nifti_4d:
        print("Error: At least one of --nifti-3d or --nifti-4d is required.", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    print("Generating clinical report from:")
    print(f"  Electrode data: {args.electrode_file}")
    if args.nifti_3d:
        print(f"  3D NIfTI:       {args.nifti_3d}")
    if args.nifti_4d:
        print(f"  4D NIfTI:       {args.nifti_4d}")
    if args.labels:
        print(f"  Labels:         {args.labels}")
    if args.acpc:
        print(f"  AC-PC:          {args.acpc}")
    if args.patient_id:
        print(f"  Patient ID:     {args.patient_id}")
    print(f"  Threshold:      {args.threshold}")

    try:
        report_path = generate_report(
            electrode_file=args.electrode_file,
            nifti_3d_path=args.nifti_3d,
            nifti_4d_path=args.nifti_4d,
            label_path=args.labels,
            acpc_path=args.acpc,
            patient_id=args.patient_id,
            threshold=args.threshold,
            output_path=args.output,
        )
        print(f"\nSuccess! Report generated: {report_path}")
    except Exception as e:
        print(f"\nError generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
