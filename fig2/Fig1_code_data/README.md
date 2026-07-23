Fig1 code and data package

Files included:
- plot_fig1.py: Python script for reproducing Fig1.
- data/: original source data and derived CSV summary files used by the script.
- outputs/Fig1_scientific_composite.png: final PNG output.
- outputs/Fig1_scientific_composite.svg: SVG file that visually matches the PNG exactly.
- outputs/Fig1_scientific_composite_vector.svg: optional vector SVG exported directly from Matplotlib.
- requirements.txt: Python dependencies.
- run_windows.bat: Windows one-click run script.
- run_mac_linux.sh: macOS/Linux run script.

Revision notes:
- Panel labels are lowercase a/b/c.
- Panel C title/subtitle spacing was adjusted to remove text overlap.
- Data files were not modified.

How to run:
1. Install dependencies: pip install -r requirements.txt
2. Run: python plot_fig1.py
3. New figures will be saved in the outputs folder.
