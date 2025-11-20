#!/usr/bin/env python3
"""
Example: Transform surgical data from stereotactic (frame) space to T1W space

This example demonstrates how to programmatically transform surgical planning
coordinates from Leksell stereotactic frame space to T1-weighted MRI space.

Typical workflow:
1. Surgical planning is done in frame space (stereotactic coordinates)
2. Frame registration provides transformation from frame -> CT
3. ANTs registration provides transformation from CT -> T1W
4. Both transforms are applied in sequence

Input: surgical_data.csv with columns:
    - patient_id, hemisphere, target_structure, x, y, z, ring, arc
    - Optional: clinical_track, clinical_depth, research_track, research_depth

Output: surgical_data_transformed.csv with additional columns:
    - Transformed coordinates (x, y, z in T1W space)
    - Original coordinates preserved as x_original, y_original, z_original
    - Virtual entry points (entry_x, entry_y, entry_z)
    - MER track positions if provided
"""

import sys
from pathlib import Path
import csv
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dbstoolbox.utils.transform_coordinates import transform_surgical_csv


def load_csv(csv_path: Path) -> list[dict]:
    """Load CSV file into list of dictionaries."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(csv_data: list[dict], output_path: Path):
    """Save list of dictionaries to CSV file."""
    if not csv_data:
        print("Warning: No data to save")
        return

    # Get all unique fieldnames from all rows
    fieldnames = []
    for row in csv_data:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"✓ Saved transformed data to: {output_path}")


def main():
    """
    Transform surgical planning data from frame space to T1W space.

    Usage:
        python transform_surgical_csv.py \\
            surgical_data.csv \\
            frame_registration.json \\
            ct_to_t1w_0GenericAffine.mat \\
            surgical_data_transformed.csv
    """
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python transform_surgical_csv.py <csv_file> <frame_reg> <ct_to_t1w> <output_csv>")
        print("\nExample:")
        print("  python transform_surgical_csv.py \\")
        print("      data/surgical_plan.csv \\")
        print("      data/frame_registration.json \\")
        print("      data/ct_to_t1w_0GenericAffine.mat \\")
        print("      data/surgical_plan_t1w.csv")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    frame_reg_file = Path(sys.argv[2])
    ct_to_t1w_file = Path(sys.argv[3])
    output_file = Path(sys.argv[4])

    # Verify input files exist
    for file in [csv_file, frame_reg_file, ct_to_t1w_file]:
        if not file.exists():
            print(f"Error: File not found: {file}")
            sys.exit(1)

    print("=" * 60)
    print("Transforming Surgical Data: Frame Space → T1W Space")
    print("=" * 60)
    print(f"\nInput CSV:           {csv_file}")
    print(f"Frame Registration:  {frame_reg_file}")
    print(f"CT → T1W Transform:  {ct_to_t1w_file}")
    print(f"Output CSV:          {output_file}")
    print()

    # Load surgical data
    print("Loading surgical data...")
    csv_data = load_csv(csv_file)
    print(f"  Found {len(csv_data)} surgical targets")

    # Display first row as example
    if csv_data:
        print(f"\n  Example row:")
        for key, value in list(csv_data[0].items())[:8]:  # Show first 8 columns
            print(f"    {key}: {value}")

    # Configure transforms
    # Order matters! Apply frame registration first, then ANTs transform
    transform_files = [frame_reg_file, ct_to_t1w_file]

    # Invert flags:
    # - Frame registration: False (forward transform from frame → CT)
    # - CT → T1W: True (inverse transform from CT → T1W)
    invert_flags = [False, True]

    # Transform types
    transform_types = ['frame_registration', 'ants']

    print("\n" + "=" * 60)
    print("Applying Transforms")
    print("=" * 60)
    print(f"1. Frame → CT:      {frame_reg_file.name} (forward)")
    print(f"2. CT → T1W:        {ct_to_t1w_file.name} (inverse)")
    print()

    # Transform the data
    transformed_data = transform_surgical_csv(
        csv_data=csv_data,
        transform_files=transform_files,
        invert_flags=invert_flags,
        transform_types=transform_types
    )

    print("✓ Transformation complete!")

    # Display transformation results
    if transformed_data:
        print("\n" + "=" * 60)
        print("Transformation Results")
        print("=" * 60)

        row = transformed_data[0]
        print(f"\nExample target (first row):")
        print(f"  Original (frame):  ({row.get('x_original', 'N/A')}, "
              f"{row.get('y_original', 'N/A')}, {row.get('z_original', 'N/A')})")
        print(f"  Transformed (T1W): ({row['x']}, {row['y']}, {row['z']})")
        print(f"  Entry point:       ({row.get('entry_x', 'N/A')}, "
              f"{row.get('entry_y', 'N/A')}, {row.get('entry_z', 'N/A')})")

        # Show MER tracks if present
        if 'clinical_target_x' in row:
            print(f"  Clinical MER:      ({row['clinical_target_x']}, "
                  f"{row['clinical_target_y']}, {row['clinical_target_z']})")

        if 'research_target_x' in row:
            print(f"  Research MER:      ({row['research_target_x']}, "
                  f"{row['research_target_y']}, {row['research_target_z']})")

    # Save transformed data
    print("\n" + "=" * 60)
    save_csv(transformed_data, output_file)
    print("=" * 60)

    print("\n✓ Done! Transformed coordinates are now in T1W space.")
    print(f"\nNew columns added:")
    print("  - entry_x, entry_y, entry_z: Virtual entry points")
    print("  - x_original, y_original, z_original: Original frame coordinates")
    print("  - clinical_target_*: Clinical MER recording positions (if provided)")
    print("  - research_target_*: Research MER recording positions (if provided)")
    print("  - mer_*_target/entry_*: MER track trajectories (if provided)")


if __name__ == "__main__":
    main()
