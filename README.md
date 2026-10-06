# Predicting Nuclear Binding Energies: Liquid Drop Model vs Machine Learning

Compares the physics-based semi-empirical mass formula (Bethe-Weizsäcker) with machine learning models for predicting nuclear binding energies, using the AME2020 Atomic Mass Evaluation.

## Why this matters
Binding energies and masses of neutron-rich nuclei feed directly into r-process nucleosynthesis calculations, and most of those nuclei have never been measured. Models therefore have to **extrapolate**, which is why this project tests both interpolation and extrapolation.

## Data
- AME2020 mass table (IAEA AMDC), measured nuclei only (extrapolated `#` entries removed)
- Nuclei with A < 16 excluded, since the liquid drop model is not valid there
- Clean dataset: 2,484 nuclei, A from 16 to 270

## Models
1. **Liquid drop**: 5 physics terms (volume, surface, Coulomb, asymmetry, pairing), coefficients fitted by least squares
2. **Random forest** on Z, N, A and the physics terms
3. **Gradient boosting** on the same features
4. **Hybrid**: liquid drop + gradient boosting trained on its residuals

## Experiments
- **Random 80/20 split** (interpolation): 1,987 train / 497 test nuclei
- **Extrapolation**: train on A < 180, test on A >= 180 (1,790 train / 694 test nuclei)

## Results (MeV)

| Model | Random split RMSE | Random split MAE | Extrapolation RMSE | Extrapolation MAE |
|---|---|---|---|---|
| Liquid drop | 3.353 | 2.356 | 8.535 | 6.693 |
| Random forest | 4.646 | 3.050 | 254.696 | 215.312 |
| Gradient boosting | 3.723 | 2.855 | 258.709 | 217.863 |
| **Liquid drop + GB residual** | **0.514** | **0.370** | **5.973** | **4.407** |

## Key findings
- The hybrid model is the best on both tests: RMSE drops by about 85% on the random split (3.35 to 0.51 MeV) and about 30% on the extrapolation test (8.54 to 5.97 MeV).
- Standalone random forest and gradient boosting do worse than the liquid-drop formula on the random split, so the physics baseline is hard to beat without it.
- Tree-based models fail badly when extrapolating (errors above 200 MeV), because they cannot predict outside the range of values seen in training. The hybrid avoids this since the liquid-drop term supplies the trend.

## Run it
```bash
pip install -r requirements.txt
python binding_energy.py
```
Download `mass_1.mas20.txt` from https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt and place it in the `data/` folder first.

## Repository contents
- `binding_energy.py`: full pipeline
- `results_random_split.csv`, `results_extrapolation.csv`: score tables
- `figures/`: `chart_of_nuclides.png`, `residuals_random.png`, `residuals_extrapolation.png`, `pred_vs_true_random.png`
