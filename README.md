# CWRU bearing signals

Rolling-bearing fault diagnosis on the public [Case Western Reserve University (CWRU) bearing dataset](https://engineering.case.edu/bearingdatacenter), worked through in one notebook: [`CWRU_bearing_signals.ipynb`](CWRU_bearing_signals.ipynb).

The notebook answers three questions:

1. **Can classical signal processing recover the fault frequency predicted by bearing geometry?** Raw waveforms, amplitude distributions, per-recording statistics, PSDs, spectrograms, spectral kurtosis and an envelope order map, compared across fault type, fault size and load, plus measured vs nominal shaft speed.
2. **How much of a model's score survives when the test data comes from conditions it never saw?** One random forest under three test protocols: random windows, unseen motor load, unseen fault size.
3. **Which lightweight classifier family generalises best, and at what cost?** Logistic regression, k-NN, SVM, random forest, gradient boosting, MLP and a small 1D-CNN on raw vibration, evaluated on the same folds, plus model size and latency.

All figures are rendered inside the notebook; open it on GitHub to see them.

## Data

- 12 kHz drive-end accelerometer, SKF 6205 drive-end bearing.
- Classes: Normal, Inner race, Outer race (6 o'clock), Ball. Fault diameters of 0.007, 0.014 and 0.021 in. Motor loads of 0 to 3 HP.
- 40 recordings cut into 500 non-overlapping 1 s windows (140 Normal, 120 per fault class).
- The `.mat` files (~140 MB) are **not** versioned. `bearings.download()` fetches them into `data/cwru/` the first time the notebook runs.

## Main results

Balanced accuracy, mean over 5 seeds (1D-CNN: 3 seeds). Chance is 0.25.

**What the split does to the score** (random forest, 17 features: 8 time statistics + 9 envelope features)

| Features | Random windows | Unseen load | Unseen damage |
|---|---|---|---|
| Time statistics | 0.985 | 0.954 | 0.557 |
| Envelope physics | 0.964 | 0.951 | 0.631 |
| Both | 0.992 | 0.984 | 0.713 |

**Model families** (same 17 features, except the 1D-CNN, which reads the raw window)

| Model | Family | Random windows | Unseen load | Unseen damage | Size (kB) | End-to-end ms / window |
|---|---|---|---|---|---|---|
| Gradient boosting | Boosted trees | 0.986 | 0.946 | **0.751** | 565 | 30.6* |
| Random forest | Bagged trees | 0.992 | 0.984 | 0.713 | 736 | 15.2* |
| k-NN | Instance-based | 0.965 | 0.930 | 0.671 | 72 | 3.3 |
| MLP | Neural net (features) | 0.902 | 0.888 | 0.651 | 89 | 3.1 |
| 1D-CNN | Neural net (raw signal) | **1.000** | **0.986** | 0.616 | 99 | **1.9** |
| Logistic regression | Linear | 0.945 | 0.902 | 0.581 | 2 | 3.1 |
| SVM (RBF) | Kernel | 0.949 | 0.907 | 0.502 | 29 | 3.1 |

\* Tree latency in scikit-learn is mostly per-call Python overhead; feature extraction takes ~2.9 ms of each feature-based figure. Latency was measured on a laptop CPU.

**Takeaways**

- Across conditions, **load barely changes the signal** (waveform, level, spectrum), while **each fault size behaves like a different machine**: the 0.014 in inner race fault is the most impulsive, the 0.014 in outer race fault looks almost healthy, and no ball fault shows a BSF line.
- Holding out a load barely changes the score. Holding out a fault size does, and **both the feature ranking and the model ranking flip** between random and unseen-damage splits. A random split would select the wrong features and the wrong model.
- Every model scores about 0.5 when the 0.014 in fault size is held out, because that outer race fault has almost no BPFO signature in the data. That part of the drop is a data limit, not a modelling failure.
- Time statistics (especially kurtosis, crest and impulse factor) identify the recording far better than the fault class (η² ≈ 0.9 vs ≈ 0.27). BPFI_h1 is the only feature where the two are close.
- **Recommended lightweight model:** gradient boosting on time and envelope features. It has the best score on unseen fault sizes, is the only model that recovers ball faults, and stays under 600 kB. The 1D-CNN tops the easy protocols but learns recording-specific waveforms and labels unseen outer race faults as ball faults.

Full tables are in [`results/`](results/).

## Repository layout

```
CWRU_bearing_signals.ipynb   the analysis (outputs included)
bearings.py                  data download/reading, bearing geometry, spectra, envelope, features
models.py                    model zoo (one per family), 1D-CNN, size/latency helpers
results/                     CSV tables written by the notebook
pyproject.toml, uv.lock      environment (Python 3.12, managed with uv)
```

## How to run

```bash
uv sync                      # creates .venv with numpy, scipy, pandas, scikit-learn, matplotlib, torch
uv run --with jupyterlab jupyter lab   # or open the notebook in VS Code with the .venv kernel
```

A full run downloads the data on first use and takes about 25 minutes on a laptop CPU, mostly 1D-CNN training in §6.

## Limitations

One test rig, one healthy bearing, seeded faults; untuned models with fixed hyperparameters; a fixed 2 to 5.5 kHz demodulation band; windows from the same recording are not independent. See §7 of the notebook.

## References

- Case Western Reserve University Bearing Data Center, https://engineering.case.edu/bearingdatacenter
- W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study", *Mechanical Systems and Signal Processing*, 64–65, 2015.
- J. Antoni, "The spectral kurtosis: a useful tool for characterising non-stationary signals", *Mechanical Systems and Signal Processing*, 20(2), 2006.
- W. Zhang et al., "A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals" (WDCNN), *Sensors*, 17(2), 2017.
