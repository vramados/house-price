Markdown
# Machine Learning Training Lifecycle: House Price Prediction
**Perspective:** MLOps & SRE-Focused Development Guide

This document defines the 6-step universal flow for building a machine learning model, mapped directly to your PyTorch implementation.

---

## 1. Data Foundation & Loading
**Concept:** Ingesting raw data from the storage layer (SQL, S3, or local CSV) and performing initial sanity checks.
**House Price Example:** Loading `house_data.csv` and checking rows, columns, and price ranges.

```python
# [CODE PORTION]
df = pd.read_csv("house_data.csv")
print(f"Rows: {len(df):,}")
print(f"Price range: ${df['price_usd'].min()} — ${df['price_usd'].max()}")
2. Preprocessing vs. Feature Engineering
Concept: * Preprocessing: Technical formatting (e.g., changing text to numeric indices so the math works).

Feature Engineering: Adding "Intelligence" (e.g., creating land-to-house ratios).

SRE Note: For images, humans usually skip "Engineering" and focus on Preprocessing (resizing/cropping).

Python
# [CODE PORTION]
# Preprocessing: Encoding Categorical Zipcodes
zip_to_idx = {z: i for i, z in enumerate(unique_zips)}
df['zip_idx'] = df['zipcode'].map(zip_to_idx)

# Target Scaling: Normalizing the target for stable training
y_scaled = y / 100_000.0
3. Data Splitting & Normalization
Concept: Isolating a "Test Set" (usually 20%) to prove the model works on unseen data. Normalization ensures all features (like SqFt vs. Beds) are on the same numeric scale.

Python
# [CODE PORTION]
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.20)

# Scaler (Normalization)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train) # Fit ONLY on training data
4. Model Selection & Hyperparameter Choice (The "Knobs")
Concept: Choosing the algorithm and defining the "Knobs" (Hyperparameters) before training. This is the "Planning" phase where you set the search space.

Python
# [CODE PORTION]
# The "Knobs" (Hyperparameters)
EPOCHS = 150
LR = 0.001
HIDDEN = [256, 128, 64] # Architecture choice

# Model Definition
class HousePriceModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        # Layers are built based on the HIDDEN hyperparameter
5. Model Training & Optimization
Concept: The execution phase. In an MLOps pipeline, you run this step with different "Knob" values (Hyperparameter Optimization) to find the winning configuration.

Python
# [CODE PORTION]
for ep in range(1, EPOCHS + 1):
    model.train()
    # Forward pass: Generate predictions
    loss = loss_fn(model(Xb), yb)
    
    # Backward pass: Update weights to reduce the error
    opt.zero_grad()
    loss.backward()
    opt.step()
6. Evaluation & Artifact Management
Concept: Measuring performance (RMSE/R2) and saving the model "Artifacts" (weights, encoders, and metrics) for the production registry.

Python
# [CODE PORTION]
# Performance Metrics (RMSE = Avg prediction error in dollars)
rmse = float(np.sqrt(mean_squared_error(y_test_usd, y_pred_usd)))

# Save Artifacts for Deployment
torch.save({"model_state_dict": model.state_dict()}, "models/house_price_model.pt")
with open("models/metrics.json", "w") as f: 
    json.dump(metrics_out, f, indent=2)
