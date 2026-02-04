# TIDL Profiler HTML Generator - V6

Modular architecture separating data extraction from HTML generation for improved maintainability and faster iteration.

## Architecture

The system consists of two independent components:

1. **Data Extractor** - Parses TIDL artifacts and outputs compressed JSON
2. **HTML Generator** - Fuses JSON data into HTML template

## Core Files

- `data_extractor.py` - Extracts all TIDL artifacts to compressed JSON
- `html_generator.py` - Generates HTML from JSON and template
- `profile.py` - Wrapper script for backward compatibility


## Usage

### Two-Step Workflow (Recommended)

Extract data once, generate HTML multiple times:

```bash
# Step 1: Extract data (runs once, takes 2-5 minutes)
python data_extractor.py work_dirs/ model_data.json

# Step 2: Generate HTML (runs multiple times, takes seconds)
python html_generator.py model_data.json.gz template .html output.html
```

### Single Command (Backward Compatible)

```bash
python  profile.py work_dirs/ template.html output.html
```

## Data Extractor

**Input:** TIDL compilation artifacts directory (`work_dirs/`)

**Output:** Compressed JSON (`.json.gz`) containing:
- Model structure (ONNX)
- Subgraph partitioning
- TIDL layer details
- Activation comparisons
- Accuracy metrics
- Performance data (processing time, cycles, memory)

**Auto-Discovery:** Automatically locates all required files in the work directory.

## HTML Generator

**Input:** JSON file (`.json` or `.json.gz`) and HTML template

**Output:** Self-contained HTML visualization

**Compression:** Activation data is compressed using gzip+base64 for 5-10x size reduction.

## Dependencies

Required:
```bash
pip install onnx numpy beautifulsoup4 openpyxl pyyaml
```

Optional (for optimized ONNX parsing):
```bash
pip install onnx-graphsurgeon
```

## Benefits

- Faster iteration: regenerate HTML without re-parsing artifacts
- Better debugging: inspect intermediate JSON output
- Code reusability: JSON can be used by other tools
- Cleaner architecture: single responsibility per component


