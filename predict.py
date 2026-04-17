"""
House Price Prediction — CLI Prediction Script
===============================================
Usage:
    python predict.py                          (interactive mode)
    python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 \
                      --num_beds 3 --num_baths 2.0 --property_type 1

Inputs:
    zipcode        : US zip code (50 markets supported)
    house_sqft     : Living area in square feet
    lot_sqft       : Lot size in square feet (0 for condos)
    num_beds       : Number of bedrooms
    num_baths      : Number of bathrooms (1.0, 1.5, 2.0 ...)
    property_type  : 1=Single Family  2=Condo/Multifamily

Output:
    Predicted price in USD
"""

import argparse, pickle, json
import numpy as np
import torch
import torch.nn as nn

# ── Model (must match train.py) ───────────────────────────────────────────────
class HousePriceModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []; in_sz = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_sz,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.15)]
            in_sz = h
        layers.append(nn.Linear(in_sz, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ── Load ──────────────────────────────────────────────────────────────────────
def load_artifacts(model_dir="models"):
    ckpt   = torch.load(f"{model_dir}/house_price_model.pt", map_location="cpu", weights_only=False)
    cfg    = ckpt["model_config"]
    model  = HousePriceModel(cfg["input_size"], cfg["hidden_sizes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with open(f"{model_dir}/scaler.pkl","rb") as f: scaler = pickle.load(f)
    with open(f"{model_dir}/zip_rates.json") as f: zip_rates = {int(k):v for k,v in json.load(f).items()}
    try:
        with open(f"{model_dir}/metrics.json") as f: metrics = json.load(f)
    except FileNotFoundError: metrics = {}
    return model, scaler, zip_rates, metrics


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(zipcode, house_sqft, lot_sqft, num_beds, num_baths, property_type,
            model, scaler, zip_rates):
    if zipcode not in zip_rates:
        raise ValueError(f"Zipcode {zipcode} not supported.\nSupported: {sorted(zip_rates)}")
    market_rate = zip_rates[zipcode]
    # Feature order MUST match FEATURE_COLS in train.py:
    # ['market_rate_per_sqft','house_sqft','lot_sqft','num_beds','num_baths','property_type']
    X = np.array([[market_rate, house_sqft, lot_sqft, num_beds, num_baths, property_type]],
                 dtype=np.float32)
    X_s = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(X_s, dtype=torch.float32)).item()
    return pred * 100_000   # convert from $100k units back to USD


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict house price",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Property types:  1 = Single Family    2 = Condo / Multifamily

Examples:
  San Francisco single family:
    python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1

  NYC condo:
    python predict.py --zipcode 10003 --house_sqft 850 --lot_sqft 0 --num_beds 1 --num_baths 1.0 --property_type 2

  Phoenix starter home:
    python predict.py --zipcode 85001 --house_sqft 1800 --lot_sqft 6000 --num_beds 3 --num_baths 2.0 --property_type 1

Zip codes by city:
  SF:       94102 94103 94107 94110 94117    ($720–810/sqft)
  Portland: 97201 97202 97205 97210 97214    ($285–320/sqft)
  LA:       90210 90025 90035 90045 90064    ($620–950/sqft)
  Seattle:  98101 98102 98103 98105 98115    ($680–730/sqft)
  Phoenix:  85001 85004 85006 85012 85016    ($185–240/sqft)
  Austin:   78701 78702 78703 78704 78705    ($360–420/sqft)
  Atlanta:  30301 30306 30308 30312 30316    ($195–240/sqft)
  Chicago:  60601 60605 60607 60611 60614    ($340–420/sqft)
  NYC:      10001 10003 10011 10014 10021    ($1050–1450/sqft)
  Houston:  77001 77002 77006 77007 77019    ($155–220/sqft)
        """)
    p.add_argument("--zipcode",       type=int)
    p.add_argument("--house_sqft",    type=float)
    p.add_argument("--lot_sqft",      type=float)
    p.add_argument("--num_beds",      type=int)
    p.add_argument("--num_baths",     type=float)
    p.add_argument("--property_type", type=int, choices=[1,2])
    p.add_argument("--model_dir",     type=str, default="models")
    return p.parse_args()


def interactive(zip_rates):
    known = sorted(zip_rates.keys())
    print("\n" + "="*55)
    print("  House Price Predictor — Interactive Mode")
    print("="*55)
    print(f"\n  Supported zip codes:\n  {known}\n")
    def ask(prompt, cast, valid=None, default=None):
        while True:
            try:
                raw = input(f"  {prompt} ").strip()
                if raw == "" and default is not None:
                    print(f"  (default: {default})")
                    return default
                val = cast(raw)
                if valid and val not in valid:
                    print(f"  Must be one of: {valid}")
                    continue
                return val
            except ValueError:
                print("  Enter a valid number.")
    z  = ask("Zip code:",                           int,   valid=known)
    hs = ask("House sqft (e.g. 1500):",             float)
    ls = ask("Lot sqft — enter 0 for condo:",       float, default=0)
    nb = ask("Number of beds (e.g. 3):",            int)
    ba = ask("Number of baths (e.g. 2.0):",         float)
    pt = ask("Property type 1=SingleFamily 2=Condo:",int, valid=[1,2])
    return z, hs, ls, nb, ba, pt


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"\nLoading model from: {args.model_dir}/")
    try:
        model, scaler, zip_rates, metrics = load_artifacts(args.model_dir)
    except FileNotFoundError:
        print("\nERROR: Model files not found. Run  python train.py  first.")
        return
    if metrics:
        print(f"Model ready | R2={metrics.get('r2_score','?')} | "
              f"RMSE=±${metrics.get('rmse_usd',0):,} | "
              f"Trained on {metrics.get('train_samples','?')} samples")

    all_given = all(v is not None for v in [
        args.zipcode, args.house_sqft, args.lot_sqft,
        args.num_beds, args.num_baths, args.property_type])

    if all_given:
        z,hs,ls,nb,ba,pt = args.zipcode, args.house_sqft, args.lot_sqft, \
                           args.num_beds, args.num_baths, args.property_type
    else:
        z,hs,ls,nb,ba,pt = interactive(zip_rates)

    try:
        price = predict(z, hs, ls, nb, ba, pt, model, scaler, zip_rates)
    except ValueError as e:
        print(f"\nERROR: {e}"); return

    prop_label = "Single Family" if pt==1 else "Condo / Multifamily"
    mrate = zip_rates[z]

    print("\n" + "="*55)
    print("  INPUTS")
    print("="*55)
    print(f"  Zip code        : {z}  (market rate: ${mrate}/sqft)")
    print(f"  House size      : {hs:,.0f} sqft")
    print(f"  Lot size        : {ls:,.0f} sqft" if ls>0 else "  Lot size        : N/A (condo)")
    print(f"  Bedrooms        : {nb}")
    print(f"  Bathrooms       : {ba}")
    print(f"  Property type   : {pt} ({prop_label})")
    print("="*55)
    print(f"  PREDICTED PRICE : ${price:,.0f}")
    print("="*55)
    if metrics.get('rmse_usd'):
        print(f"\n  Accuracy: ±${metrics['rmse_usd']:,} RMSE  |  R2={metrics.get('r2_score','?')}")

if __name__ == "__main__":
    main()
