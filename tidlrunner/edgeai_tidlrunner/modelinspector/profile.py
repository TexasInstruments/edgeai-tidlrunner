#!/usr/bin/env python3
# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.

"""
Compiler V6 - Wrapper Script (Backward Compatibility)

This script provides backward compatibility by calling both data_extractor.py
and html_generator.py sequentially.

Usage:
    python compiler.py <work_dirs/> <template.html> <output.html>

Example:
    python compiler.py work_dirs/ template.html output.html

This is equivalent to running:
    python data_extractor.py work_dirs/ temp_data.json.gz
    python html_generator.py temp_data.json.gz template.html output.html
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path


def main():
    """Main wrapper function"""
    if len(sys.argv) < 4:
        print("=" * 70)
        print("Compiler V6 - TIDL Profiler HTML Generator")
        print("=" * 70)
        print("\nUsage: python compiler.py <work_dirs/> <template.html> <output.html>")
        print("\nArguments:")
        print("  work_dirs/     - Work directory containing all model files")
        print("  template.html  - HTML template file")
        print("  output.html    - Output HTML file path")
        print("\nExample:")
        print("  python compiler.py work_dirs/ template.html output.html")
        print("\nThis script orchestrates:")
        print("  1. Data extraction (data_extractor.py) - Slow, run once")
        print("  2. HTML generation (html_generator.py) - Fast, run multiple times")
        print("\nFor better workflow, you can also run the scripts separately:")
        print("  Step 1: python data_extractor.py work_dirs/ data.json.gz")
        print("  Step 2: python html_generator.py data.json.gz template.html output.html")
        print("=" * 70)
        sys.exit(1)

    work_dirs_path = sys.argv[1]
    template_path = sys.argv[2]
    output_html_path = sys.argv[3]

    # Validate inputs
    if not os.path.exists(work_dirs_path):
        print(f"ERROR: Work directory not found: {work_dirs_path}")
        sys.exit(1)

    if not os.path.isdir(work_dirs_path):
        print(f"ERROR: {work_dirs_path} is not a directory")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    # Get script directory
    script_dir = Path(__file__).parent

    # Create temporary JSON file for intermediate data
    output_dir = Path(output_html_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    temp_json_path = output_dir / "temp_extracted_data.json.gz"

    
    print(f"Work Directory: {work_dirs_path}")
    print(f"Template:       {template_path}")
    print(f"Output HTML:    {output_html_path}")
    print(f"Temp JSON:      {temp_json_path}")
    print("=" * 70)

    try:
        # Step 1: Extract data to JSON
        print("\n" + "=" * 70)
        print("STEP 1/2: Extracting artifact data to JSON")
        print("=" * 70)

        data_extractor_script = script_dir / "data_extractor.py"
        cmd1 = [sys.executable, str(data_extractor_script), work_dirs_path, str(temp_json_path)]

        print(f"Running: {' '.join(cmd1)}\n")
        result1 = subprocess.run(cmd1, check=True)

        if result1.returncode != 0:
            print(f"\nERROR: Data extraction failed with code {result1.returncode}")
            sys.exit(1)

        # Step 2: Generate HTML from JSON
        print("\n" + "=" * 70)
        print("STEP 2/2: Generating HTML from JSON")
        print("=" * 70)

        html_generator_script = script_dir / "html_generator.py"
        cmd2 = [sys.executable, str(html_generator_script), str(temp_json_path), template_path, output_html_path]

        print(f"Running: {' '.join(cmd2)}\n")
        result2 = subprocess.run(cmd2, check=True)

        if result2.returncode != 0:
            print(f"\nERROR: HTML generation failed with code {result2.returncode}")
            sys.exit(1)

        # Clean up temporary JSON file (optional - keep for debugging)
        print(f"\nKeeping intermediate JSON file: {temp_json_path}")
        print("  (You can delete this file manually if not needed)")

        print("\n" + "=" * 70)
        print("SUCCESS! Complete pipeline finished.")
        print("=" * 70)
        print(f"\nGenerated files:")
        print(f"  - Intermediate JSON: {temp_json_path}")
        print(f"  - Final HTML:        {output_html_path}")
        print(f"\nOpen the HTML file in your browser:")
        print(f"  {os.path.abspath(output_html_path)}")


    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed at step with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
