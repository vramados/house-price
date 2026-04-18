# Machine Learning Training Lifecycle: House Price Prediction

**Perspective:** MLOps and SRE-focused development guide

This document defines the six-step flow for building a machine learning model, mapped directly to your PyTorch implementation.

---

## 1. Data Foundation and Loading

### Description
Ingest raw data from the storage layer (SQL, S3, or local CSV) and perform initial sanity checks.

For this project, you load `house_data.csv` and inspect row count, column shape, and price range.

### Code
```python
df = pd.read_csv("house_data.csv")
print(f"Rows: {len(df):,}")
print(f"Price range: ${df['price_usd'].min()} - ${df['price_usd'].max()}")
```

## 2. Preprocessing vs. Feature Engineering

### Description
Preprocessing handles technical formatting so the model can train (for example, converting categorical values to numeric indices).

Feature engineering adds useful signal (for example, creating domain-driven ratios or transformations).

SRE note: For image workloads, teams often focus more on preprocessing (resize/crop/normalize) than manual feature engineering.

### Code
```python
# Preprocessing: encode categorical zip codes
zip_to_idx = {z: i for i, z in enumerate(unique_zips)}
df["zip_idx"] = df["zipcode"].map(zip_to_idx)

# Target scaling for stable training
y_scaled = y / 100_000.0
```

## 3. Data Splitting and Normalization

### Description
Split data into training and test sets so you can validate generalization on unseen examples.

Normalization puts features on a comparable scale (for example, square footage vs. bedroom count).

### Code
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.20
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # Fit only on training data
X_test_s = scaler.transform(X_test)
```

## 4. Model Selection and Hyperparameter Choice

### Description
This project uses a **PyTorch Dense MLP regressor** named `HousePriceModel`.

In simple terms, this is a feed-forward neural network for tabular regression.
It learns non-linear relationships between home attributes and final sale price.

Model details from `train.py`:
- Input features: 6 (`market_rate_per_sqft`, `house_sqft`, `lot_sqft`, `num_beds`, `num_baths`, `property_type`)
- Hidden layers: `[256, 128, 64]`
- Activation: `ReLU` after each hidden linear layer
- Regularization: `BatchNorm1d` + `Dropout(0.15)` after each hidden layer
- Output layer: 1 neuron (predicts scaled price)
- Target scaling: `price_usd / 100000`
- Total trainable parameters: ~43.9K

Training setup used with this model:
- Loss: `MSELoss`
- Optimizer: `Adam(lr=0.001, weight_decay=1e-5)`
- LR scheduler: `ReduceLROnPlateau` with `patience=10`, `factor=0.5`
- Epochs: `150`, batch size: `64`

### Why this model is a good fit
- House pricing is usually non-linear (for example, price does not increase at a constant rate with square footage), and MLPs can model these curves.
- The dataset is structured/tabular with mixed signals (size, lot, bedrooms, bathrooms, market context), which dense networks handle well.
- Batch normalization and dropout improve generalization and reduce overfitting risk on medium-size datasets.
- Adam + learning-rate scheduling provides stable training and faster convergence without heavy manual tuning.
- The architecture is expressive enough to capture interactions, while still small enough to train quickly and deploy easily.

### Trade-offs and limitations
- MLPs are less interpretable than linear models or tree models.
- Prediction quality depends heavily on data quality and feature coverage (for example, missing neighborhood attributes can limit accuracy).
- Performance may improve further with cross-validation and comparison against tree-based baselines (XGBoost/LightGBM/RandomForest).

### Code
```python
# Hyperparameters
EPOCHS = 150
LR = 0.001
HIDDEN = [256, 128, 64]

# Model definition used in training
class HousePriceModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        in_sz = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_sz, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.15)
            ]
            in_sz = h
        layers.append(nn.Linear(in_sz, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = HousePriceModel(input_size=6, hidden_sizes=HIDDEN)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sched = optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", patience=10, factor=0.5
)
```

## 5. Model Training and Optimization

### Description
Run iterative training with forward pass, loss computation, backward pass, and optimizer step.

In MLOps pipelines, this stage is repeated across different hyperparameter sets to find the best run.

### Code
```python
for ep in range(1, EPOCHS + 1):
    model.train()

    # Forward pass
    loss = loss_fn(model(Xb), yb)

    # Backward pass and update
    opt.zero_grad()
    loss.backward()
    opt.step()
```

## 6. Evaluation and Artifact Management

### Description
Evaluate model quality with metrics such as RMSE and R2, then persist deployment artifacts.

Artifacts include model weights, encoders/scalers, and metrics for registry or CI/CD promotion.

### Code
```python
rmse = float(np.sqrt(mean_squared_error(y_test_usd, y_pred_usd)))

torch.save(
    {"model_state_dict": model.state_dict()},
    "models/house_price_model.pt"
)

with open("models/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
```
