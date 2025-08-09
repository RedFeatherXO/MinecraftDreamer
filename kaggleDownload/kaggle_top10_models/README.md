# 🚀 KAGGLE TRAINING RESULTS

## 📊 Training Summary
- **Total Models Trained:** 13
- **Models Saved:** Top 10
- **Best Model:** BS64_LR0.001_adamw
- **Best Validation Loss:** 0.000957

## 📁 Package Contents

### /models
Die Top 10 Modelle nach Validation Loss sortiert.
Gesamt: 55.34 MB

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
Detaillierte Statistiken ALLER 13 trainierten Modelle.

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
checkpoint = torch.load('models/rank1_BS64_LR0.001_adamw.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Validation Loss abrufen
print(f"Loaded model with validation loss: {checkpoint['val_loss']:.6f}")
```

## 📈 Top 10 Modelle

| Rank | Model | Val Loss | Epoch | Saved |
|------|-------|----------|-------|-------|
| 1 | BS64_LR0.001_adamw | 0.000957 | 38 | ✅ |
| 2 | BS64_LR0.001_adam | 0.000965 | 40 | ✅ |
| 3 | BS64_LR0.0005_adam | 0.000976 | 40 | ✅ |
| 4 | BS64_LR0.0005_adamw | 0.000980 | 40 | ✅ |
| 5 | BS128_LR0.001_adam | 0.001003 | 38 | ✅ |
| 6 | BS128_LR0.001_adamw | 0.001015 | 39 | ✅ |
| 7 | BS128_LR0.0005_adamw | 0.001136 | 40 | ✅ |
| 8 | BS128_LR0.0005_adam | 0.001140 | 40 | ✅ |
| 9 | BS64_LR0.0001_adamw | 0.001310 | 40 | ✅ |
| 10 | BS64_LR0.0001_adam | 0.001348 | 40 | ✅ |

*Zeigt nur Top 10 von 13 Modellen. Siehe training_summary.json für alle.*
