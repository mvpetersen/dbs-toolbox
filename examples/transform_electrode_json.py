#!/usr/bin/env python3
"""
Example: Transform electrode reconstruction from postop CT space to T1W space

This example demonstrates how to programmatically transform PyPaCER electrode
reconstruction data from postoperative CT space to T1-weighted MRI space.

Typical workflow:
1. PyPaCER detects electrodes in postop CT → coordinates in postop CT space
2. ANTs registration: postop CT → preop CT
3. ANTs registration: preop CT → T1W
4. Both transforms are applied in sequence to get electrodes into T1W space

Input: electrode_reconstruction.json from PyPaCER with structure:
    {
        "electrodes": [
            {
                "side": "left",
                "contact_positions_3d": [[x1, y1, z1], [x2, y2, z2], ...],
                "tip_position": [x, y, z],
                "entry_position": [x, y, z],
                "trajectory_coordinates": [[x1, y1, z1], ...],
                ...
            }
        ],
        "metadata": {...}
    }

Output: electrode_reconstruction_t1w.json with:
    - All coordinates transformed to T1W space
    - Original coordinates preserved as *_original fields
    - Updated metadata with transformation info
"""

import sys
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dbstoolbox.utils.transform_coordinates import transform_pypacer_reconstruction


def load_json(json_path: Path) -> dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: dict, output_path: Path):
    """Save data to JSON file with pretty formatting."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved transformed data to: {output_path}")


def main():
    """
    Transform PyPaCER electrode reconstruction from postop CT space to T1W space.

    Usage:
        python transform_electrode_json.py \\
            electrode_reconstruction.json \\
            postop_to_preop_0GenericAffine.mat \\
            preop_to_t1w_0GenericAffine.mat \\
            electrode_reconstruction_t1w.json

    Note: Frame registration transforms are NOT used for electrode data
          (PyPaCER already outputs coordinates in CT space)
    """
    # Parse command line arguments
    if len(sys.argv) < 5:
        print("Usage: python transform_electrode_json.py <input_json> <postop_to_preop> <preop_to_t1w> <output_json>")
        print("\nExample:")
        print("  python transform_electrode_json.py \\")
        print("      data/electrode_reconstruction.json \\")
        print("      data/postop_to_preop_0GenericAffine.mat \\")
        print("      data/preop_to_t1w_0GenericAffine.mat \\")
        print("      data/electrode_reconstruction_t1w.json")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    postop_to_preop_file = Path(sys.argv[2])
    preop_to_t1w_file = Path(sys.argv[3])
    output_file = Path(sys.argv[4])

    # Verify input files exist
    for file in [input_file, postop_to_preop_file, preop_to_t1w_file]:
        if not file.exists():
            print(f"Error: File not found: {file}")
            sys.exit(1)

    print("=" * 70)
    print("Transforming Electrode Reconstruction: Postop CT → Preop CT → T1W")
    print("=" * 70)
    print(f"\nInput JSON:               {input_file}")
    print(f"Postop → Preop Transform: {postop_to_preop_file}")
    print(f"Preop → T1W Transform:    {preop_to_t1w_file}")
    print(f"Output JSON:              {output_file}")
    print()

    # Load electrode data
    print("Loading electrode reconstruction...")
    reconstruction = load_json(input_file)

    num_electrodes = len(reconstruction.get("electrodes", []))
    print(f"  Found {num_electrodes} electrode(s)")

    # Display summary
    for i, electrode in enumerate(reconstruction.get("electrodes", [])):
        side = electrode.get("side", "unknown")
        num_contacts = len(electrode.get("contact_positions_3d", []))
        print(f"  Electrode {i+1}: {side} side, {num_contacts} contacts")

        # Show first contact position
        if num_contacts > 0:
            first_contact = electrode["contact_positions_3d"][0]
            print(f"    First contact (CT): ({first_contact[0]:.2f}, "
                  f"{first_contact[1]:.2f}, {first_contact[2]:.2f})")

    # Configure transforms
    # For PyPaCER data, only ANTs transforms are needed
    # (data is already in postop CT space)
    # Order matters! Apply postop→preop first, then preop→T1W
    transform_files = [postop_to_preop_file, preop_to_t1w_file]

    # Invert flags:
    # - Postop CT → Preop CT: True (inverse transform from moving → fixed)
    # - Preop CT → T1W: True (inverse transform from moving → fixed)
    invert_flags = [True, True]

    # Transform types (all ANTs for electrode data)
    transform_types = ['ants', 'ants']

    print("\n" + "=" * 70)
    print("Applying Transforms")
    print("=" * 70)
    print(f"1. Postop → Preop: {postop_to_preop_file.name} (inverse)")
    print(f"2. Preop → T1W:    {preop_to_t1w_file.name} (inverse)")
    print()

    # Transform the data
    transformed_data, total_contacts = transform_pypacer_reconstruction(
        reconstruction_data=reconstruction,
        transform_files=transform_files,
        invert_flags=invert_flags,
        transform_types=transform_types
    )

    print(f"✓ Transformation complete! Transformed {total_contacts} contacts total.")

    # Display transformation results
    if transformed_data.get("electrodes"):
        print("\n" + "=" * 60)
        print("Transformation Results")
        print("=" * 60)

        for i, electrode in enumerate(transformed_data["electrodes"]):
            side = electrode.get("side", "unknown")
            print(f"\nElectrode {i+1} ({side} side):")

            # Show first contact transformation
            if electrode.get("contact_positions_3d"):
                original = electrode.get("contact_positions_3d_original", [[None]*3])[0]
                transformed = electrode["contact_positions_3d"][0]
                print(f"  First contact:")
                print(f"    Original (Postop CT): ({original[0]:.2f}, {original[1]:.2f}, {original[2]:.2f})")
                print(f"    Transformed (T1W):    ({transformed[0]:.2f}, {transformed[1]:.2f}, {transformed[2]:.2f})")

            # Show tip transformation
            if electrode.get("tip_position"):
                original_tip = electrode.get("tip_position_original", [None]*3)
                transformed_tip = electrode["tip_position"]
                print(f"  Tip position:")
                print(f"    Original (Postop CT): ({original_tip[0]:.2f}, {original_tip[1]:.2f}, {original_tip[2]:.2f})")
                print(f"    Transformed (T1W):    ({transformed_tip[0]:.2f}, {transformed_tip[1]:.2f}, {transformed_tip[2]:.2f})")

    # Save transformed data
    print("\n" + "=" * 60)
    save_json(transformed_data, output_file)
    print("=" * 60)

    print("\n✓ Done! Electrode coordinates are now in T1W space.")
    print(f"\nAll electrode components transformed:")
    print("  - contact_positions_3d: All contact coordinates")
    print("  - tip_position: Electrode tip")
    print("  - entry_position: Electrode entry point")
    print("  - trajectory_coordinates: Full trajectory")
    print("\nOriginal coordinates preserved as *_original fields")


if __name__ == "__main__":
    main()
