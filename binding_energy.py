"""
Nuclear Binding Energy Prediction
Liquid-drop model (physics) vs machine learning (data science).

Run:  python binding_energy.py
Needs: pip install pandas numpy scikit-learn matplotlib
"""

# ---------- 1. IMPORTS ----------
import os                                   # for making folders and checking if files exist
import urllib.request                       # for downloading the dataset from the internet
import numpy as np                          # numerical arrays and math
import pandas as pd                         # tables (DataFrames) for data handling
import matplotlib.pyplot as plt             # plotting
from sklearn.linear_model import LinearRegression            # used to fit the liquid-drop formula
from sklearn.ensemble import RandomForestRegressor           # ML model 1
from sklearn.ensemble import GradientBoostingRegressor       # ML model 2
from sklearn.model_selection import train_test_split         # random train/test split
from sklearn.metrics import mean_absolute_error, mean_squared_error  # error metrics

# ---------- 2. SETTINGS ----------
DATA_URL = "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"  # official AME2020 mass table
DATA_PATH = "data/mass_1.mas20.txt"         # where we save the downloaded file
FIG_DIR = "figures"                         # folder where plots are saved
RANDOM_STATE = 42                           # fixed seed so results are reproducible
MAGIC = [2, 8, 20, 28, 50, 82, 126]         # magic numbers (closed nuclear shells)

os.makedirs("data", exist_ok=True)          # create data folder if it does not exist
os.makedirs(FIG_DIR, exist_ok=True)         # create figures folder if it does not exist


# ---------- 3. LOAD DATA ----------
def download_data():
    """Download the AME2020 file once and reuse it afterwards."""
    if not os.path.exists(DATA_PATH):                       # only download if file is missing
        print("Downloading AME2020 mass table...")          # tell the user what is happening
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)     # download and save to disk
    else:
        print("Data file already present, skipping download.")  # nothing to do


def load_ame(path):
    """Read the fixed-width AME2020 text file into a pandas DataFrame."""
    # Width of each column in characters, taken from the file's Fortran format description
    widths = [1, 3, 5, 5, 5, 1, 3, 4, 1, 14, 12, 13, 1, 10, 1, 2, 13, 11, 1, 3, 1, 13, 12]
    # Names for each column ("skip" columns are blank separators we throw away)
    names = ["skip1", "N_minus_Z", "N", "Z", "A", "skip2", "element", "origin", "skip3",
             "mass_excess_keV", "mass_excess_unc", "BE_per_A_keV", "skip4", "BE_per_A_unc",
             "skip5", "beta_type", "beta_energy", "beta_unc", "skip6", "A_atomic",
             "skip7", "atomic_mass_micro_u", "atomic_mass_unc"]
    # read_fwf = "read fixed width file"; the first 36 lines are the file header text
    df = pd.read_fwf(path, widths=widths, names=names, skiprows=36)
    return df                                               # return the raw table


# ---------- 4. CLEAN DATA ----------
def clean(df):
    """Keep only experimentally measured nuclei with valid binding energies."""
    df = df[["N", "Z", "A", "element", "BE_per_A_keV"]].copy()   # keep only the columns we need
    df["BE_per_A_keV"] = df["BE_per_A_keV"].astype(str)          # treat as text so we can inspect '#'
    # In AME a '#' marks values estimated from trends (not measured). We drop them.
    df = df[~df["BE_per_A_keV"].str.contains("#")]               # remove extrapolated rows
    # '*' means "not calculable"; to_numeric turns any bad entries into NaN (missing)
    df["BE_per_A_keV"] = pd.to_numeric(df["BE_per_A_keV"], errors="coerce")
    for col in ["N", "Z", "A"]:                                  # make sure N, Z, A are numbers
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()                                             # drop every row with missing data
    df = df.drop_duplicates(subset=["N", "Z"])                   # one row per nucleus
    df["BE_MeV"] = df["BE_per_A_keV"] * df["A"] / 1000.0         # total binding energy in MeV
    df = df[df["A"] >= 16].reset_index(drop=True)                # liquid drop fails for very light nuclei
    # Sanity checks: if these fail, the file was parsed wrongly
    assert (df["N"] + df["Z"] == df["A"]).all(), "N + Z should equal A"
    assert df["BE_per_A_keV"].between(1000, 9000).all(), "BE/A should be 1-9 MeV"
    return df                                                    # clean table


# ---------- 5. FEATURE ENGINEERING ----------
def liquid_drop_features(df):
    """Build the terms of the semi-empirical mass formula (Bethe-Weizsacker)."""
    A, Z, N = df["A"].values, df["Z"].values, df["N"].values     # pull columns out as numpy arrays
    # Pairing: +1 for even-even nuclei, -1 for odd-odd, 0 for odd-A
    pairing = np.where((Z % 2 == 0) & (N % 2 == 0), 1.0,
                       np.where((Z % 2 == 1) & (N % 2 == 1), -1.0, 0.0))
    X = np.column_stack([
        A,                                  # volume term:    a_V * A
        -A ** (2 / 3),                      # surface term:  -a_S * A^(2/3)
        -Z * (Z - 1) / A ** (1 / 3),        # Coulomb term:  -a_C * Z(Z-1) / A^(1/3)
        -(N - Z) ** 2 / A,                  # asymmetry:     -a_A * (N-Z)^2 / A
        pairing / np.sqrt(A),               # pairing:       +delta / sqrt(A)
    ])
    return X                                # the matrix has one column per physics term


class LiquidDrop:
    """Liquid drop model: B = X @ coefficients. Linear, so least squares fits it."""
    def __init__(self):
        self.model = LinearRegression(fit_intercept=False)   # no intercept: formula has none

    def fit(self, df):
        self.model.fit(liquid_drop_features(df), df["BE_MeV"])   # find best a_V, a_S, a_C, a_A, a_P
        return self                                              # allow chaining

    def predict(self, df):
        return self.model.predict(liquid_drop_features(df))      # predicted binding energy in MeV


def ml_features(df):
    """Simple features for the ML models: Z, N, A and the physics terms."""
    base = np.column_stack([df["Z"], df["N"], df["A"]])          # raw proton/neutron/mass numbers
    return np.hstack([base, liquid_drop_features(df)])           # stack with the physics features


# ---------- 6. METRICS ----------
def report(name, y_true, y_pred, results):
    """Compute MAE and RMSE (in MeV) and store them in the results list."""
    mae = mean_absolute_error(y_true, y_pred)                    # average absolute error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))           # root mean squared error (punishes big misses)
    results.append({"model": name, "MAE_MeV": mae, "RMSE_MeV": rmse})   # save for the summary table
    print(f"{name:35s} MAE = {mae:7.3f} MeV   RMSE = {rmse:7.3f} MeV")   # print one line


# ---------- 7. TRAIN + EVALUATE ON A SPLIT ----------
def run_experiment(train, test, label):
    """Train all models on `train`, evaluate on `test`, return predictions and results."""
    print(f"\n=== {label}: {len(train)} train / {len(test)} test nuclei ===")   # header
    results, preds = [], {}                                      # containers for outputs

    ld = LiquidDrop().fit(train)                                 # model A: pure physics
    preds["Liquid drop"] = ld.predict(test)                      # predict on unseen nuclei
    report("Liquid drop", test["BE_MeV"], preds["Liquid drop"], results)

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)  # model B
    rf.fit(ml_features(train), train["BE_MeV"])                  # learn from the training nuclei
    preds["Random forest"] = rf.predict(ml_features(test))       # predict on the test nuclei
    report("Random forest", test["BE_MeV"], preds["Random forest"], results)

    gb = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                   random_state=RANDOM_STATE)    # model C
    gb.fit(ml_features(train), train["BE_MeV"])                  # learn from the training nuclei
    preds["Gradient boosting"] = gb.predict(ml_features(test))   # predict
    report("Gradient boosting", test["BE_MeV"], preds["Gradient boosting"], results)

    # Model D (hybrid): physics first, then ML learns what physics gets wrong (the residual)
    resid_train = train["BE_MeV"].values - ld.predict(train)     # error of liquid drop on training set
    hy = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                   random_state=RANDOM_STATE)    # ML model for the residual
    hy.fit(ml_features(train), resid_train)                      # learn the physics model's mistakes
    preds["Liquid drop + GB residual"] = ld.predict(test) + hy.predict(ml_features(test))  # add correction
    report("Liquid drop + GB residual", test["BE_MeV"], preds["Liquid drop + GB residual"], results)

    return pd.DataFrame(results), preds                          # table of scores + raw predictions


# ---------- 8. PLOTS ----------
def plot_residuals(test, preds, fname, title):
    """Plot prediction error vs neutron number; magic numbers should stand out."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)   # two side-by-side panels
    for ax, name in zip(axes, ["Liquid drop", "Liquid drop + GB residual"]):   # one panel per model
        resid = test["BE_MeV"].values - preds[name]                  # error = true - predicted
        ax.scatter(test["N"], resid, s=8, alpha=0.6)                 # one dot per nucleus
        for m in MAGIC:                                              # draw a line at every magic number
            ax.axvline(m, color="red", ls="--", lw=0.6, alpha=0.6)
        ax.axhline(0, color="black", lw=0.8)                         # zero-error reference
        ax.set_title(name)                                           # panel title
        ax.set_xlabel("Neutron number N")                            # x label
    axes[0].set_ylabel("Residual (MeV)  [true - predicted]")         # y label on the left panel
    fig.suptitle(title)                                              # overall title
    fig.tight_layout()                                               # avoid overlapping text
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150)                       # save as image for the README
    plt.close(fig)                                                   # free memory


def plot_pred_vs_true(test, preds, fname):
    """Scatter of predicted vs true binding energy per nucleon for each model."""
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))   # one panel per model
    for ax, (name, p) in zip(axes, preds.items()):                   # loop over models
        ax.scatter(test["BE_MeV"] / test["A"], p / test["A"], s=6, alpha=0.5)   # B/A true vs predicted
        lo, hi = 6.5, 9.0                                            # typical B/A range in MeV
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)                   # perfect-prediction line
        ax.set_title(name, fontsize=9)                               # panel title
        ax.set_xlabel("True B/A (MeV)")                              # x label
    axes[0].set_ylabel("Predicted B/A (MeV)")                        # y label
    fig.tight_layout()                                               # tidy layout
    fig.savefig(f"{FIG_DIR}/{fname}", dpi=150)                       # save
    plt.close(fig)                                                   # free memory


def plot_nuclear_chart(df):
    """Exploratory plot: binding energy per nucleon across the chart of nuclides."""
    fig, ax = plt.subplots(figsize=(8, 6))                           # one big panel
    sc = ax.scatter(df["N"], df["Z"], c=df["BE_per_A_keV"] / 1000, s=6, cmap="viridis")  # colour = B/A
    fig.colorbar(sc, label="B/A (MeV)")                              # colour legend
    ax.set_xlabel("N")                                               # x label
    ax.set_ylabel("Z")                                               # y label
    ax.set_title("Chart of nuclides coloured by binding energy per nucleon")   # title
    fig.savefig(f"{FIG_DIR}/chart_of_nuclides.png", dpi=150)         # save
    plt.close(fig)                                                   # free memory


# ---------- 9. MAIN ----------
def main():
    #download_data()                                                  # step 1: get data
    raw = load_ame(DATA_PATH)                                        # step 2: parse file
    df = clean(raw)                                                  # step 3: clean
    print(f"Clean dataset: {len(df)} nuclei, A from {df['A'].min()} to {df['A'].max()}")  # quick summary
    print(df.describe().round(2))                                    # basic statistics (EDA)
    plot_nuclear_chart(df)                                           # EDA plot

    # Experiment 1: random split (interpolation: test nuclei sit among training nuclei)
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)   # 80/20 split
    res1, preds1 = run_experiment(train, test, "Random split")       # train + evaluate
    plot_residuals(test, preds1, "residuals_random.png", "Random split residuals")
    plot_pred_vs_true(test, preds1, "pred_vs_true_random.png")

    # Experiment 2: extrapolation (train on light/medium nuclei, test on heavy ones A >= 180)
    train2, test2 = df[df["A"] < 180], df[df["A"] >= 180]            # split by mass number
    res2, preds2 = run_experiment(train2, test2, "Extrapolation (test A >= 180)")
    plot_residuals(test2, preds2, "residuals_extrapolation.png", "Extrapolation residuals (A >= 180)")

    res1.to_csv("results_random_split.csv", index=False)             # save score tables for the README
    res2.to_csv("results_extrapolation.csv", index=False)            # save second table
    print("\nDone. Figures saved in ./figures, scores saved as CSV files.")   # finish message


if __name__ == "__main__":                                           # run main() only when executed directly
    main()
