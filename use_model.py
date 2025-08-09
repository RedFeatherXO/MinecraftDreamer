import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Schritt 1: Exakte Modelldefinition einfügen ---
# Kopiere hier die genaue `WorldModel`-Klasse aus deinem Trainings-Notebook.
# Die Architektur muss 100% identisch sein.
IMG_CHANNELS = 3
class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 64, 4, 2, 1), nn.ReLU(),
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
            nn.ConvTranspose2d(64, IMG_CHANNELS, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Schritt 2: Modell laden und Gewichte zuweisen ---
# Passe den Pfad zu deiner heruntergeladenen Datei an
MODEL_PATH = "/home/meik/Downloads/Malmo-0.37.0-Linux-Ubuntu-18.04-64bit_withBoost_Python3.6/Python_Examples/kaggleDownload/kaggle_top10_models/models/rank1_loss0.000957_BS64_LR0.001_adamw.pth" # z.B. "/home/user/Downloads/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Erstelle eine leere Instanz des Modells
model = WorldModel().to(device)

# Lade die gespeicherten Gewichte
# Wichtig: model.load_state_dict() erwartet nur die Gewichte (das state_dict)
# Falls du das ganze Dictionary gespeichert hast, musst du es zuerst extrahieren.
try:
    # Lade das gesamte Checkpoint-Dictionary (falls du so gespeichert hast)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        # Extrahiere nur die Modellgewichte
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Modellgewichte aus Checkpoint-Dictionary geladen.")
    else:
        # Falls die Datei nur das state_dict enthält
        model.load_state_dict(checkpoint)
        print("✅ Modellgewichte direkt geladen.")

except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    print("Stelle sicher, dass der Pfad korrekt ist und die Datei nicht korrupt ist.")
    exit()

# Setze das Modell in den Evaluationsmodus. Sehr wichtig!
# Das deaktiviert z.B. Dropout und sorgt für konsistente Ergebnisse.
model.eval()
print(f"Modell ist auf '{device}' und im Evaluationsmodus.")

# --- Schritt 3: Ein Testbild vorbereiten und Inferenz durchführen ---
# Hier laden wir einfach ein zufälliges Bild. Besser wäre ein echtes Bild
# aus deinem HDF5-Datensatz, um eine realistische Rekonstruktion zu sehen.
# Wichtig ist das Format: (Batch, Channels, Höhe, Breite) -> (1, 3, 64, 64)
input_image = torch.rand(1, 3, 64, 64).to(device)

# Führe die Inferenz ohne Gradientenberechnung durch (spart Speicher und Zeit)
with torch.no_grad():
    reconstructed_image = model(input_image)

print("Inferenz erfolgreich!")

# --- Schritt 4: Ergebnis visualisieren ---
# Konvertiere die Tensoren zurück zu numpy-Bildern für die Anzeige
input_np = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
recon_np = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(input_np)
axes[0].set_title("Original (Zufallsbild)")
axes[0].axis("off")

axes[1].imshow(recon_np)
axes[1].set_title("Rekonstruktion")
axes[1].axis("off")

plt.suptitle("Modell-Inferenz Test")
plt.show()