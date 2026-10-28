"""Estimate a house price with a LinearRegression model and sample data."""

import argparse

from sklearn.linear_model import LinearRegression


FEATURES = [
    [850, 2, 1, 25],
    [950, 2, 1, 18],
    [1100, 2, 2, 15],
    [1250, 3, 2, 12],
    [1400, 3, 2, 10],
    [1550, 3, 2, 8],
    [1700, 3, 2, 6],
    [1850, 4, 2, 5],
    [2000, 4, 3, 4],
    [2200, 4, 3, 3],
    [2400, 4, 3, 2],
    [2600, 5, 3, 1],
]

# Illustrative prices in thousands of dollars; replace with local sales data
# for meaningful estimates.
PRICES = [180, 205, 235, 265, 300, 330, 365, 395, 435, 475, 515, 555]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a house price using a model trained on sample data."
    )
    parser.add_argument("--sqft", type=float, required=True, help="Living area in square feet")
    parser.add_argument("--bedrooms", type=int, required=True, help="Number of bedrooms")
    parser.add_argument("--bathrooms", type=float, required=True, help="Number of bathrooms")
    parser.add_argument("--age", type=float, required=True, help="House age in years")
    args = parser.parse_args()

    if args.sqft <= 0 or args.bedrooms <= 0 or args.bathrooms <= 0 or args.age < 0:
        parser.error("sqft, bedrooms, and bathrooms must be positive; age cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    model = LinearRegression()
    model.fit(FEATURES, PRICES)

    estimated_price = model.predict(
        [[args.sqft, args.bedrooms, args.bathrooms, args.age]]
    )[0]
    print(f"Estimated house price: ${estimated_price:,.2f} thousand")
    print("Note: This estimate uses illustrative sample data, not local market data.")


if __name__ == "__main__":
    main()
