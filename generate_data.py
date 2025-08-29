"""Generate a synthetic insurance customer dataset for churn prediction.

This script produces a CSV file with the following columns:

- ``customer_id``: an integer identifier
- ``tenure``: number of years the customer has been with the insurer
- ``premium``: yearly premium amount in dollars
- ``num_claims``: number of claims filed
- ``region``: one of ``North``, ``South``, ``East`` or ``West``
- ``churn``: binary flag indicating whether the customer churned

The distributions used for the synthetic data are loosely based on
common insurance scenarios.  The churn variable is generated using a
logistic function of the other features to introduce some predictive
signal.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np


def generate(n: int, random_state: int | None = None) -> List[dict]:
    """Generate a list of synthetic customer records.

    Parameters
    ----------
    n:
        Number of records to generate.
    random_state:
        Optional random seed for reproducibility.

    Returns
    -------
    list of dict
        A list of dictionaries containing customer features and churn label.
    """
    rng = np.random.default_rng(random_state)
    customers = []
    regions = np.array(["North", "South", "East", "West"])
    for cid in range(n):
        tenure = rng.integers(1, 11)  # years between 1 and 10
        premium = rng.normal(1000, 300)  # mean $1000, std $300
        premium = max(100.0, premium)  # ensure positive
        num_claims = rng.poisson(1.0)
        region = rng.choice(regions)
        # Compute churn probability using a logistic model
        # Customers with shorter tenure, higher premiums and more claims are more likely to churn
        logit = (
            -0.5 * tenure
            + 0.003 * premium
            + 0.8 * num_claims
            + (0.2 if region == "South" else 0.0)
        )
        prob = 1 / (1 + np.exp(-logit))
        churn = rng.random() < prob
        customers.append(
            {
                "customer_id": cid,
                "tenure": int(tenure),
                "premium": float(round(premium, 2)),
                "num_claims": int(num_claims),
                "region": str(region),
                "churn": int(churn),
            }
        )
    return customers


def write_csv(customers: List[dict], output_path: Path) -> None:
    """Write a list of customer dictionaries to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["customer_id", "tenure", "premium", "num_claims", "region", "churn"],
        )
        writer.writeheader()
        for row in customers:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic insurance churn dataset.")
    parser.add_argument("--n", type=int, default=5000, help="Number of records to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "customers.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    customers = generate(args.n, random_state=args.seed)
    write_csv(customers, Path(args.output))
    print(f"Generated {len(customers)} records to {args.output}")


if __name__ == "__main__":
    main()