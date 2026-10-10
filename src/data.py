"""Synthetic telecom customer-churn dataset generator.

Produces a realistic, reproducible CSV with a mix of numeric and categorical
features plus a binary churn target. Everything is generated locally with
scikit-learn so the project has zero external data dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges",
                    "support_calls", "data_usage_gb"]
CATEGORICAL_FEATURES = ["contract", "payment_method", "internet_service",
                        "tech_support"]
TARGET = "churn"

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["electronic-check", "mailed-check", "bank-transfer", "credit-card"]
INTERNET = ["dsl", "fiber-optic", "none"]
SUPPORT = ["yes", "no"]


def generate(n_samples: int = 5000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate the synthetic churn dataset as a DataFrame."""
    rng = np.random.default_rng(random_state)

    # Core signal: 5 informative numeric features driving churn.
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        weights=[0.73, 0.27],          # ~27% churn rate, like real telecom data
        flip_y=0.03,
        random_state=random_state,
    )
    df = pd.DataFrame(X_num, columns=[
        "tenure_months", "monthly_charges", "total_charges",
        "support_calls", "data_usage_gb",
    ])

    # Rescale numerics into plausible business ranges.
    df["tenure_months"] = (df["tenure_months"] - df["tenure_months"].min())
    df["tenure_months"] = (df["tenure_months"] / df["tenure_months"].max() * 71 + 1).round()
    df["monthly_charges"] = (df["monthly_charges"] - df["monthly_charges"].min())
    df["monthly_charges"] = (df["monthly_charges"] / df["monthly_charges"].max() * 100 + 20).round(2)
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"]
                           * rng.uniform(0.85, 1.15, n_samples)).round(2)
    df["support_calls"] = np.abs(df["support_calls"] * 2).round().astype(int).clip(0, 12)
    df["data_usage_gb"] = np.abs(df["data_usage_gb"] * 40 + 120).round(1)

    # Categorical features, correlated with churn for realism.
    churn = y.astype(bool)
    df["contract"] = np.where(
        churn,
        rng.choice(CONTRACTS, n_samples, p=[0.70, 0.20, 0.10]),
        rng.choice(CONTRACTS, n_samples, p=[0.35, 0.35, 0.30]),
    )
    df["payment_method"] = np.where(
        churn,
        rng.choice(PAYMENTS, n_samples, p=[0.55, 0.20, 0.15, 0.10]),
        rng.choice(PAYMENTS, n_samples, p=[0.25, 0.25, 0.25, 0.25]),
    )
    df["internet_service"] = rng.choice(INTERNET, n_samples, p=[0.40, 0.45, 0.15])
    df["tech_support"] = np.where(
        churn,
        rng.choice(SUPPORT, n_samples, p=[0.25, 0.75]),
        rng.choice(SUPPORT, n_samples, p=[0.55, 0.45]),
    )
    df[TARGET] = y.astype(int)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the churn dataset.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/churn.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n_samples=args.n_samples)
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df.to_csv(out.parent / "train.csv", index=False)
    test_df.to_csv(out.parent / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train / {len(test_df)} test rows to {out.parent}/")


if __name__ == "__main__":
    main()
