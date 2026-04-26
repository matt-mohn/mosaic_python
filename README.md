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

```bash
# Windows
run_mosaic.bat

# Mac/Linux
chmod +x run_mosaic.sh && ./run_mosaic.sh
```

The launcher auto-installs dependencies on first run.

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

## License

MIT License - see [LICENSE](LICENSE)

---

A project of Matt Mohn ([@mattmxhn](https://x.com/mattmxhn)). The ReCom algorithm is derived from the one developed by the MGGG Redistricting Lab.
