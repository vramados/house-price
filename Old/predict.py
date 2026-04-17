"""
House Price Prediction — CLI Prediction Script
===============================================
Loads the trained model and predicts house prices from command-line inputs.

Usage:
    python predict.py                          (interactive mode)
    python predict.py --medinc 8.3 --houseage 41 --averooms 7 --avebedrms 1 --population 322 --aveoccup 2.5 --latitude 37.88 --longitude -122.23

Inputs  : 8 features (see below)
Output  : Predicted median house price in USD
"""

import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn

# ── Model Definition (must match train.py exactly) ───────────────────────────
class HousePriceModel(nn.Module):
    def __init__(self, input_size=8, hidden_sizes=None):
        super(HousePriceModel, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.1))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Load Model & Scaler ───────────────────────────────────────────────────────
def load_model(model_path="models/house_price_model.pt",
               scaler_path="models/scaler.pkl"):
    """Load trained model and scaler from disk."""
    # Load PyTorch model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    arch       = checkpoint["model_architecture"]
    model      = HousePriceModel(
        input_size   = arch["input_size"],
        hidden_sizes = arch["hidden_sizes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, checkpoint


# ── Prediction Function ───────────────────────────────────────────────────────
def predict(features: dict, model, scaler) -> float:
    """
    Predict house price given a dictionary of features.

    Args:
        features: dict with keys MedInc, HouseAge, AveRooms, AveBedrms,
                  Population, AveOccup, Latitude, Longitude
        model:    trained PyTorch model
        scaler:   fitted StandardScaler

    Returns:
        Predicted price in USD
    """
    # Build feature array in the correct order
    feature_order = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                     "Population", "AveOccup", "Latitude", "Longitude"]

    X = np.array([[features[k] for k in feature_order]], dtype=np.float32)

    # Scale using the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # Run inference
    with torch.no_grad():
        X_tensor    = torch.tensor(X_scaled, dtype=torch.float32)
        prediction  = model(X_tensor)
        price_100k  = prediction.item()

    # Convert from $100k units to full USD
    price_usd = price_100k * 100_000
    return price_usd


# ── CLI Argument Parser ───────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict California house prices using trained PyTorch model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature descriptions:
  --medinc      Median income in block group ($10,000s). E.g. 8.3 = $83,000/yr
  --houseage    Median house age in years (1-52)
  --averooms    Average number of rooms per household (typically 3-10)
  --avebedrms   Average number of bedrooms per household (typically 0.8-3)
  --population  Block group population (typically 100-35,000)
  --aveoccup    Average household members (typically 1.5-5)
  --latitude    Block group latitude (32.5-42 for California)
  --longitude   Block group longitude (-124 to -114 for California)

Examples:
  San Francisco (high end):
    python predict.py --medinc 8.3 --houseage 41 --averooms 7 --avebedrms 1 --population 322 --aveoccup 2.5 --latitude 37.88 --longitude -122.23

  Inland California (mid range):
    python predict.py --medinc 3.0 --houseage 20 --averooms 5 --avebedrms 1.1 --population 800 --aveoccup 2.8 --latitude 36.5 --longitude -119.5
        """
    )

    parser.add_argument("--medinc",     type=float, help="Median income ($10k units)")
    parser.add_argument("--houseage",   type=float, help="Median house age (years)")
    parser.add_argument("--averooms",   type=float, help="Average rooms per household")
    parser.add_argument("--avebedrms",  type=float, help="Average bedrooms per household")
    parser.add_argument("--population", type=float, help="Block group population")
    parser.add_argument("--aveoccup",   type=float, help="Average household occupants")
    parser.add_argument("--latitude",   type=float, help="Block group latitude")
    parser.add_argument("--longitude",  type=float, help="Block group longitude")
    parser.add_argument("--model",      type=str,   default="models/house_price_model.pt",
                        help="Path to model file (default: models/house_price_model.pt)")
    parser.add_argument("--scaler",     type=str,   default="models/scaler.pkl",
                        help="Path to scaler file (default: models/scaler.pkl)")

    return parser.parse_args()


# ── Interactive Mode ──────────────────────────────────────────────────────────
def interactive_mode(model, scaler):
    """Ask user for inputs one by one."""
    print("\n" + "=" * 55)
    print("  House Price Predictor — Interactive Mode")
    print("=" * 55)
    print("  Enter values for each feature below.")
    print("  Press Ctrl+C to exit.\n")

    prompts = [
        ("MedInc",     "Median income in $10,000s (e.g. 8.3 = $83k/yr)", 5.0),
        ("HouseAge",   "Median house age in years (e.g. 25)",             20.0),
        ("AveRooms",   "Average rooms per household (e.g. 6.0)",          5.0),
        ("AveBedrms",  "Average bedrooms per household (e.g. 1.0)",       1.0),
        ("Population", "Block group population (e.g. 500)",               800.0),
        ("AveOccup",   "Average household members (e.g. 2.5)",            2.8),
        ("Latitude",   "Latitude (California: 32.5 to 42.0)",             36.5),
        ("Longitude",  "Longitude (California: -124 to -114)",           -119.5),
    ]

    features = {}
    for key, prompt, default in prompts:
        while True:
            try:
                val = input(f"  {key} — {prompt}\n  > ").strip()
                if val == "":
                    val = default
                    print(f"  (using default: {default})")
                features[key] = float(val)
                break
            except ValueError:
                print("  Please enter a valid number.")

    return features


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Load model
    print("\nLoading model from:", args.model)
    try:
        model, scaler, checkpoint = load_model(args.model, args.scaler)
    except FileNotFoundError:
        print("\nERROR: Model file not found.")
        print("Run 'python train.py' first to train and save the model.")
        return

    # Load metrics if available
    try:
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        print(f"Model loaded | R2={metrics['r2_score']} | RMSE={metrics['rmse']} | "
              f"Trained on {metrics['train_samples']:,} samples")
    except FileNotFoundError:
        print("Model loaded successfully.")

    # Determine if CLI args were provided
    all_args = [args.medinc, args.houseage, args.averooms, args.avebedrms,
                args.population, args.aveoccup, args.latitude, args.longitude]

    if any(v is not None for v in all_args):
        # CLI mode — use provided args
        # Fill in defaults for any missing args
        features = {
            "MedInc"    : args.medinc     or 5.0,
            "HouseAge"  : args.houseage   or 20.0,
            "AveRooms"  : args.averooms   or 5.0,
            "AveBedrms" : args.avebedrms  or 1.0,
            "Population": args.population or 800.0,
            "AveOccup"  : args.aveoccup   or 2.8,
            "Latitude"  : args.latitude   or 36.5,
            "Longitude" : args.longitude  or -119.5,
        }
    else:
        # Interactive mode
        features = interactive_mode(model, scaler)

    # Run prediction
    price = predict(features, model, scaler)

    # Display result
    print("\n" + "=" * 55)
    print("  INPUTS")
    print("=" * 55)
    print(f"  Median income    : ${features['MedInc']*10:.0f}k/year")
    print(f"  House age        : {features['HouseAge']:.0f} years")
    print(f"  Avg rooms        : {features['AveRooms']:.1f}")
    print(f"  Avg bedrooms     : {features['AveBedrms']:.1f}")
    print(f"  Population       : {features['Population']:.0f}")
    print(f"  Avg occupancy    : {features['AveOccup']:.1f} people")
    print(f"  Location         : {features['Latitude']:.2f}°N, {features['Longitude']:.2f}°W")
    print("=" * 55)
    print(f"  PREDICTED PRICE  : ${price:,.0f}")
    print("=" * 55)
    print(f"\n  Note: Model accuracy ±$50-70k (RMSE) on California 1990s data.")
    print(f"  This is a learning project — not for real estate use!")


if __name__ == "__main__":
    main()
