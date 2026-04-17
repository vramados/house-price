"""
House Price Prediction — Training Script
==========================================
Dataset  : house_data.csv (500 real-ish US house listings)
Features : zipcode, house_sqft, lot_sqft, num_beds, num_baths, property_type
Target   : price_usd

Model    : PyTorch Neural Network (Dense MLP — Layer 3 Deep Learning)
Outputs  :
    models/house_price_model.pt   PyTorch model weights
    models/house_price_model.pkl  Pickle model (same model)
    models/scaler.pkl             Feature scaler (required at inference)
    models/zip_encoder.pkl        Zipcode encoder (required at inference)
    models/metrics.json           RMSE, R2, MAE

Usage:
    python train.py
"""

import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 150
BATCH_SIZE  = 64
LR          = 0.001
HIDDEN      = [256, 128, 64]
DATA_FILE   = "house_data.csv"
MODEL_DIR   = "models"

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("  House Price Prediction — PyTorch Training")
print("=" * 60)


# ── 1. Load CSV ────────────────────────────────────────────────────────────────
print(f"\n[1/6] Loading {DATA_FILE}...")

df = pd.read_csv(DATA_FILE)
print(f"      Rows    : {len(df):,}")
print(f"      Columns : {list(df.columns)}")
print(f"\n      Preview:")
print(df.head(5).to_string(index=False))

# Quick data check
print(f"\n      Price range : ${df['price_usd'].min():,.0f} — ${df['price_usd'].max():,.0f}")
print(f"      Avg price   : ${df['price_usd'].mean():,.0f}")
print(f"      Prop types  : 1=Single Family: {(df['property_type']==1).sum()}  "
      f"2=Condo/Multi: {(df['property_type']==2).sum()}")


# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("\n[2/6] Preparing features...")

# Encode zipcode as integer index (zip codes are categorical, not numeric)
# 94102 ≠ 2x47051 — the number itself means nothing
unique_zips = sorted(df['zipcode'].unique())
zip_to_idx  = {z: i for i, z in enumerate(unique_zips)}
df['zip_idx'] = df['zipcode'].map(zip_to_idx)

# Save zip encoder for inference
with open(f"{MODEL_DIR}/zip_encoder.pkl", "wb") as f:
    pickle.dump(zip_to_idx, f)
print(f"      Unique zipcodes : {len(unique_zips)}")
print(f"      Zip encoder saved → {MODEL_DIR}/zip_encoder.pkl")

# Feature matrix — using zip_idx instead of raw zipcode
FEATURE_COLS = ['market_rate_per_sqft', 'house_sqft', 'lot_sqft', 'num_beds', 'num_baths', 'property_type']
TARGET_COL   = 'price_usd'

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

# Scale target to $100k units for easier training (same as California Housing convention)
y_scaled = y / 100_000.0

print(f"\n      Feature columns : {FEATURE_COLS}")
print(f"      X shape         : {X.shape}")
print(f"      y shape         : {y_scaled.shape}")
print(f"\n      Feature stats:")
for i, col in enumerate(FEATURE_COLS):
    print(f"      {col:15s}: min={X[:,i].min():.1f}  max={X[:,i].max():.1f}  mean={X[:,i].mean():.1f}")


# ── 3. Split & Scale ──────────────────────────────────────────────────────────
print("\n[3/6] Train/test split (80/20) and normalize...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.20, random_state=42
)
print(f"      Train : {len(X_train)} samples")
print(f"      Test  : {len(X_test)} samples")

scaler      = StandardScaler()
X_train_s   = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_s    = scaler.transform(X_test)         # transform test with same scaler

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"      Scaler saved → {MODEL_DIR}/scaler.pkl")

# Tensors
Xtr = torch.tensor(X_train_s, dtype=torch.float32)
ytr = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
Xte = torch.tensor(X_test_s,  dtype=torch.float32)
yte = torch.tensor(y_test,    dtype=torch.float32).unsqueeze(1)

loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)


# ── 4. Define Model ───────────────────────────────────────────────────────────
print("\n[4/6] Building neural network...")

class HousePriceModel(nn.Module):
    """
    Dense MLP — Layer 3 Deep Learning (Dense architecture family).
    Input: 6 features → hidden layers → 1 output (price in $100k units)
    """
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        in_sz  = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_sz, h), nn.ReLU(),
                       nn.BatchNorm1d(h), nn.Dropout(0.15)]
            in_sz = h
        layers.append(nn.Linear(in_sz, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model    = HousePriceModel(len(FEATURE_COLS), HIDDEN)
loss_fn  = nn.MSELoss()
opt      = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sched    = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10, factor=0.5)

params = sum(p.numel() for p in model.parameters())
print(f"      Architecture : {len(FEATURE_COLS)} → {' → '.join(map(str,HIDDEN))} → 1")
print(f"      Parameters   : {params:,}")
print(f"      Loss fn      : MSE (Mean Squared Error)")
print(f"      Optimizer    : Adam  lr={LR}  weight_decay=1e-5")
print(f"      Scheduler    : ReduceLROnPlateau (patience=10)")
print(f"      Epochs       : {EPOCHS}  Batch size: {BATCH_SIZE}")


# ── 5. Train ──────────────────────────────────────────────────────────────────
print(f"\n[5/6] Training {EPOCHS} epochs...")
print(f"      {'Epoch':>6}  {'TrainLoss':>12}  {'ValLoss':>10}  {'LR':>10}")
print("      " + "-" * 46)

best_val_loss = float('inf')
best_state    = None

for ep in range(1, EPOCHS + 1):
    # Training pass
    model.train()
    batch_losses = []
    for Xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(Xb), yb)
        loss.backward()
        opt.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    # Validation pass
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xte), yte).item()

    sched.step(val_loss)
    current_lr = opt.param_groups[0]['lr']

    # Track best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    if ep % 15 == 0 or ep == 1 or ep == EPOCHS:
        print(f"      {ep:6d}  {train_loss:12.4f}  {val_loss:10.4f}  {current_lr:10.6f}")

# Restore best weights
model.load_state_dict(best_state)
print(f"\n      Best val loss : {best_val_loss:.4f}")


# ── 6. Evaluate & Save ────────────────────────────────────────────────────────
print("\n[6/6] Evaluating on test set and saving files...")

model.eval()
with torch.no_grad():
    y_pred_scaled = model(Xte).numpy().flatten()

# Convert back to full USD for metrics
y_pred_usd = y_pred_scaled * 100_000
y_test_usd = y_test * 100_000

rmse = float(np.sqrt(mean_squared_error(y_test_usd, y_pred_usd)))
r2   = float(r2_score(y_test_usd, y_pred_usd))
mae  = float(np.mean(np.abs(y_test_usd - y_pred_usd)))

print(f"\n      ── Performance Metrics ──────────────────")
print(f"      RMSE     : ${rmse:,.0f}   (avg prediction error)")
print(f"      R2 Score : {r2:.4f}    (model explains {r2*100:.1f}% of variance)")
print(f"      MAE      : ${mae:,.0f}   (median absolute error)")

# Save .pt
torch.save({
    "model_state_dict":   model.state_dict(),
    "model_config":       {"input_size": len(FEATURE_COLS), "hidden_sizes": HIDDEN},
    "feature_cols":       FEATURE_COLS,
    "target_unit":        "price_usd / 100000",
    "epochs_trained":     EPOCHS,
    "best_val_loss":      best_val_loss,
    "metrics":            {"rmse_usd": round(rmse), "r2": round(r2,4), "mae_usd": round(mae)},
}, f"{MODEL_DIR}/house_price_model.pt")
print(f"\n      Saved → {MODEL_DIR}/house_price_model.pt  (PyTorch)")

# Save .pkl
with open(f"{MODEL_DIR}/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"      Saved → {MODEL_DIR}/house_price_model.pkl  (Pickle)")

# Save metrics
metrics_out = {
    "rmse_usd":       round(rmse),
    "r2_score":       round(r2, 4),
    "mae_usd":        round(mae),
    "epochs":         EPOCHS,
    "train_samples":  len(X_train),
    "test_samples":   len(X_test),
    "feature_cols":   FEATURE_COLS,
    "target":         "price_usd",
    "property_type":  {"1": "Single Family", "2": "Condo / Multifamily"},
}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"      Saved → {MODEL_DIR}/metrics.json")

print("\n" + "=" * 60)
print("  Training complete!")
print(f"  Model accuracy: ±${rmse:,.0f} RMSE  |  R2={r2:.3f}")
print("  Run: python predict.py --help")
print("=" * 60)
