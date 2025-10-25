# Data Directory

This folder stores intermediate and derived data for the MIDI Bassline project.

## Regenerating
To reproduce the full dataset:
1. Run `extract_basslines.py` to produce `output_basslines/`.
2. Run `build_tokens.py` to generate `tokens.txt` and `features.csv`.
3. Run `1_assign_labels.py` to produce `labels.csv`.

These files are large and not included in the repository by default.

## Example
A minimal example (2–3 songs) is included under `data/sample/` for testing the pipeline.
