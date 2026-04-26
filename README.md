<p align="center">
  <img src="assets/mosaic_header.png" alt="Mosaic" width="1000">
</p>

<p align="center">
  <em>Redistricting for Python</em>
</p>

---

Mosaic is a redistricting toolkit using recombination (ReCom) algorithms with simulated annealing. It includes a live GUI for map visualization and optimization.

## Quick Start

**Option 1: Run the launcher (no Python required)**

1. **Download the project**: Click the green **Code** button at the top of this page, then select **Download ZIP**
2. **Extract the ZIP**: Right-click the downloaded file and choose "Extract All"
3. **Run the launcher**:
   - **Windows**: Double-click `run_mosaic.bat`
   - **Mac/Linux**: Open a terminal in the folder and run `chmod +x run_mosaic.sh && ./run_mosaic.sh`

The launcher auto-installs dependencies on first run (requires internet).

**Option 2: Install with pip**

```bash
pip install -e .
mosaic
```

## Requirements

- Python >= 3.10
- Windows 10+ / macOS 12+ / Linux

## Sample Data

The `shapefiles/` folder includes North Carolina precinct data as a ready-to-use starting point.

## Background

This is a Python port of [Mosaic for R](https://github.com/matt-mohn/Mosaic). Some features are implemented differently or still in progress.

## License

MIT License - see [LICENSE](LICENSE)

---

A project of Matt Mohn ([@mattmxhn](https://x.com/mattmxhn)). The ReCom algorithm is derived from the one developed by the MGGG Redistricting Lab.
