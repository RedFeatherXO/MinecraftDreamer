# 🚀 KAGGLE TRAINING RESULTS

## 📊 Training Summary
- **Total Models Trained:** 5
- **Models Saved:** Top 3
- **Best Model:** BS512_LR0.0005_LD256_adamw
- **Best Validation Loss:** 299.361228

## 📁 Package Contents

### /models
Die Top 3 Modelle nach Validation Loss sortiert.
Gesamt: 315.32 MB

### /visualizations
Alle generierten Visualisierungen und Plots.

### /tensorboard_logs.tar.gz
Komprimierte TensorBoard Logs.

Entpacken mit:
```bash
tar -xzf tensorboard_logs.tar.gz
tensorboard --logdir runs/
```

### training_summary.json
Detaillierte Statistiken ALLER 5 trainierten Modelle.

## 🔧 Modell Laden

```python
import torch
import torch.nn as nn

# Model Definition (aus dem Training Code)
class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Modell laden
model = WorldModel()
checkpoint = torch.load('models/rank1_BS512_LR0.0005_LD256_adamw.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Validation Loss abrufen
print(f"Loaded model with validation loss: {checkpoint['val_loss']:.6f}")
```

## 📈 Top 5 Modelle

| Rank | Model | Val Loss | Epoch | Saved |
|------|-------|----------|-------|-------|
| 1 | BS512_LR0.0005_LD256_adamw | 299.361228 | 48 | ✅ |
| 2 | BS512_LR0.0001_LD512_adamw | 303.396467 | 46 | ✅ |
| 3 | BS512_LR0.0005_LD512_adamw | 303.676605 | 49 | ✅ |
| 4 | BS512_LR0.0005_LD1024_adamw | 307.669708 | 46 | ❌ |
| 5 | BS512_LR0.0001_LD1024_adamw | 353.180860 | 25 | ❌ |
