# Predicting Nuclear Binding Energies: Liquid Drop Model vs Machine Learning

Compares the physics-based semi-empirical mass formula (Bethe-Weizsäcker) with machine learning models for predicting nuclear binding energies, using the AME2020 Atomic Mass Evaluation.

## Why this matters
Binding energies and masses of neutron-rich nuclei feed directly into r-process nucleosynthesis calculations, and most of those nuclei have never been measured. Models therefore have to **extrapolate**, which is why this project tests both interpolation and extrapolation.

## Data
- AME2020 mass table (IAEA AMDC), measured nuclei only (extrapolated `#` entries removed)
- Nuclei with A < 16 excluded, since the liquid drop model is not valid there

## Models
1. **Liquid drop**: 5 physics terms (volume, surface, Coulomb, asymmetry, pairing), coefficients fitted by least squares
2. **Random forest** on Z, N, A and the physics terms
3. **Gradient boosting** on the same features
4. **Hybrid**: liquid drop + gradient boosting trained on its residuals

## Experiments
- **Random 80/20 split** (interpolation)
- **Extrapolation**: train on A < 180, test on A ≥ 180

## Results
Fill in from `results_random_split.csv` and `results_extrapolation.csv` after running:

| Model | Random split MAE (MeV) | Extrapolation MAE (MeV) |
|---|---|---|
| Liquid drop | | |
| Random forest | | |
| Gradient boosting | | |
| Liquid drop + GB residual | | |

## Key findings
Write 3 to 4 sentences here from your own plots: which model wins on each test, how the errors behave near magic numbers (see `figures/residuals_*.png`), and how tree-based models behave when extrapolating.

## Run it
```bash
pip install pandas numpy scikit-learn matplotlib
python binding_energy.py
```

## Figures
- `figures/chart_of_nuclides.png`
- `figures/residuals_random.png`
- `figures/residuals_extrapolation.png`
- `figures/pred_vs_true_random.png`

## Possible extensions
Add shell-correction features (distance to nearest magic number), try a neural network, or compare with the Duflo-Zuker model.
