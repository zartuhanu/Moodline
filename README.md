# Moodline: Inferring Musical Mood from Symbolic MIDI Basslines

This repository contains the code for:

**“Moodline: Inferring Musical Mood from Symbolic MIDI Basslines”**

The project implements an end-to-end pipeline for predicting musical mood using **symbolic MIDI basslines only**, without audio or lyrics.

---

## Dataset

This project uses the **Lakh MIDI Dataset (Clean MIDI Subset)**.

You can refer to the dataset as in the paper, or access it here:
(https://colinraffel.com/projects/lmd/)
Place the dataset in the `data/` directory (or adjust paths in configs accordingly).

---

## Pipeline Overview

The entire pipeline is sequential.
You must run scripts in order, as each step depends on outputs from the previous stage.

---

## Execution Order

The pipeline is strictly sequential.

Run the scripts in order of the folders:

```
00_deduplication → 01_extraction → 02_tokenization → 03_labels → 04_model
```

Within each folder, execute the scripts in their numbered order.

Each stage depends on the outputs of the previous one, so skipping or reordering steps will break the pipeline.

---

## Note on Labeling

The labeling stage includes an optional manual refinement step.

If you prefer a fully automatic pipeline, you may skip the manual labeling scripts.

In that case:

* continue executing the remaining scripts in order
* adjust configuration files if needed to account for the absence of human-labeled data

Skipping this step may slightly affect label quality and final model performance.

