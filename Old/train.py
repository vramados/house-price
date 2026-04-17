"""
House Price Prediction — Training Script
=========================================
Dataset   : California Housing (tries sklearn built-in first,
            falls back to synthetic data if network unavailable)
Model     : PyTorch Neural Network (Dense / MLP — Layer 3 Deep Learning)
Outputs   :
    models/house_price_model.pt   (PyTorch format)
    models/house_price_model.pkl  (Pickle format)
    models/scaler.pkl             (feature normalizer)
    models/metrics.json           (RMSE, R2, MAE)

Run:
    python train.py
"""

import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
torch.manual_seed(42); np.random.seed(42)

EPOCHS = 100; BATCH_SIZE = 256; LR = 0.001
HIDDEN = [128, 64, 32]
FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

print("="*60)
print("  House Price Prediction — PyTorch Training")
print("="*60)

# 1. Load dataset
print("\n[1/6] Loading dataset...")
def load_data():
    try:
        from sklearn.datasets import fetch_california_housing
        h = fetch_california_housing()
        print("      Source: California Housing (sklearn built-in)")
        return h.data, h.target
    except Exception:
        print("      Source: Synthetic California-style data (20,640 rows)")
        n = 20640; np.random.seed(42)
        MI = np.random.lognormal(1.5,0.6,n).clip(0.5,15)
        HA = np.random.uniform(1,52,n)
        AR = np.random.lognormal(1.8,0.4,n).clip(1,20)
        AB = np.random.lognormal(0.1,0.2,n).clip(0.5,5)
        PO = np.random.lognormal(6.5,0.8,n).clip(3,35682)
        AO = np.random.lognormal(0.9,0.4,n).clip(0.5,10)
        LA = np.random.uniform(32.5,42,n)
        LO = np.random.uniform(-124,-114,n)
        y = (0.45*MI+0.08*HA+0.03*AR-2.5*np.abs(LA-37)-1.2*np.abs(LO+120)+np.random.normal(0,0.4,n)).clip(0.5,5)
        X = np.column_stack([MI,HA,AR,AB,PO,AO,LA,LO]).astype(np.float32)
        return X, y.astype(np.float32)

X, y = load_data()
print(f"      Rows: {X.shape[0]:,}  Features: {X.shape[1]}  Price: ${y.min()*100:.0f}k-${y.max()*100:.0f}k")

# 2. Feature summary
print("\n[2/6] Feature summary:")
for i,n in enumerate(FEATURES):
    print(f"      {n:12s}: min={X[:,i].min():.2f}  max={X[:,i].max():.2f}  mean={X[:,i].mean():.2f}")

# 3. Split & scale
print("\n[3/6] Split 80/20 and normalize...")
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)
with open("models/scaler.pkl","wb") as f: pickle.dump(scaler,f)
print(f"      Train: {len(Xtr):,}  Test: {len(Xte):,}  Scaler saved.")

Xtr_t = torch.tensor(Xtr_s,dtype=torch.float32)
ytr_t = torch.tensor(ytr,dtype=torch.float32).unsqueeze(1)
Xte_t = torch.tensor(Xte_s,dtype=torch.float32)
yte_t = torch.tensor(yte,dtype=torch.float32).unsqueeze(1)
loader = DataLoader(TensorDataset(Xtr_t,ytr_t),batch_size=BATCH_SIZE,shuffle=True)

# 4. Build model
print("\n[4/6] Building MLP neural network...")
class HousePriceModel(nn.Module):
    def __init__(self,input_size=8,hidden=[128,64,32]):
        super().__init__()
        layers=[]; ins=input_size
        for h in hidden:
            layers+=[nn.Linear(ins,h),nn.ReLU(),nn.BatchNorm1d(h),nn.Dropout(0.1)]
            ins=h
        layers.append(nn.Linear(ins,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

model = HousePriceModel(8,HIDDEN)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(),lr=LR)
sched = optim.lr_scheduler.StepLR(opt,30,0.5)
print(f"      Architecture: 8 → {' → '.join(map(str,HIDDEN))} → 1")
print(f"      Params: {sum(p.numel() for p in model.parameters()):,}  Epochs: {EPOCHS}  LR: {LR}")

# 5. Train
print(f"\n[5/6] Training {EPOCHS} epochs...")
print(f"      {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}")
print("      "+"-"*32)
vl=0
for ep in range(1,EPOCHS+1):
    model.train(); bl=[]
    for Xb,yb in loader:
        opt.zero_grad(); l=loss_fn(model(Xb),yb); l.backward(); opt.step(); bl.append(l.item())
    tl=np.mean(bl)
    model.eval()
    with torch.no_grad(): vl=loss_fn(model(Xte_t),yte_t).item()
    sched.step()
    if ep%10==0 or ep==1: print(f"      {ep:6d}  {tl:10.4f}  {vl:10.4f}")

# 6. Evaluate & save
print("\n[6/6] Evaluating and saving...")
model.eval()
with torch.no_grad(): yp=model(Xte_t).numpy().flatten()
rmse=float(np.sqrt(mean_squared_error(yte,yp)))
r2=float(r2_score(yte,yp))
mae=float(np.mean(np.abs(yte-yp)))
print(f"\n      RMSE     : {rmse:.4f}  (~${rmse*100:.0f}k avg error)")
print(f"      R2 Score : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"      MAE      : {mae:.4f}  (~${mae*100:.0f}k median error)")

# Save .pt
torch.save({"model_state_dict":model.state_dict(),
            "model_architecture":{"input_size":8,"hidden_sizes":HIDDEN},
            "feature_names":FEATURES,"epochs":EPOCHS,
            "metrics":{"rmse":rmse,"r2":r2,"mae":mae}},
           "models/house_price_model.pt")
print("\n      Saved → models/house_price_model.pt  (PyTorch)")

# Save .pkl
with open("models/house_price_model.pkl","wb") as f: pickle.dump(model,f)
print("      Saved → models/house_price_model.pkl  (Pickle)")

# Save metrics
with open("models/metrics.json","w") as f:
    json.dump({"rmse":round(rmse,4),"r2_score":round(r2,4),"mae":round(mae,4),
               "epochs":EPOCHS,"train_samples":len(Xtr),"test_samples":len(Xte),
               "feature_names":FEATURES},f,indent=2)
print("      Saved → models/metrics.json")

print("\n"+"="*60)
print("  Training complete! Run: python predict.py --help")
print("="*60)
