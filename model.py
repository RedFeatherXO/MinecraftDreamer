# model.py
# Enthält alle neuronalen Netzwerke für die Dreamer-Architektur.

import torch
import torch.nn as nn
import random
from pathlib import Path

# --- Konstanten (könnten auch aus einer Config kommen) ---
IMG_CHANNELS = 3
TARGET_HEIGHT = 64
TARGET_WIDTH = 64

# --- WorldModel Klasse (unverändert) ---
# Diese Definition muss exakt mit der übereinstimmen, mit der du trainiert hast.
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

class Actor(nn.Module):
    """Lernt eine Policy, um Aktionen auszuwählen."""
    def __init__(self, config):
        super().__init__()
        # TODO: Definiere hier dein Actor-Netzwerk.
        pass

    def forward(self, state):
        # TODO: Implementiere den Forward-Pass für den Actor.
        pass

class Critic(nn.Module):
    """Lernt, den erwarteten zukünftigen Reward vorherzusagen."""
    def __init__(self, config):
        super().__init__()
        # TODO: Definiere hier dein Critic-Netzwerk.
        pass

    def forward(self, state):
        # TODO: Implementiere den Forward-Pass für den Critic.
        pass


# --- DreamerAgent Klasse (ANGEPASST) ---
# Dient als Hülle für das WorldModel und zukünftige Komponenten (Actor/Critic)
class DreamerAgent:
    """Der Haupt-Agent, der alle Modelle zusammenhält."""

    def __init__(self, config, pretrained_world_model_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model = WorldModel().to(self.device)

        # --- NEU: Lade die vortrainierten Gewichte, falls ein Pfad angegeben ist ---
        if pretrained_world_model_path:
            model_path = Path(pretrained_world_model_path)
            if model_path.exists():
                print(f"🔄 Lade vortrainiertes WorldModel von: {model_path}")
                try:
                    # Lade das Checkpoint-Dictionary
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    # Prüfe, ob die Gewichte in einem 'model_state_dict' Schlüssel liegen
                    if 'model_state_dict' in checkpoint:
                        self.world_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # Falls die Datei nur die Gewichte enthält
                        self.world_model.load_state_dict(checkpoint)
                    
                    # Setze das Modell in den Evaluationsmodus (wichtig für Inferenz!)
                    self.world_model.eval()
                    print(f"✅ WorldModel erfolgreich geladen und im eval()-Modus.")

                except Exception as e:
                    print(f"❌ FEHLER beim Laden des Modells: {e}")
            else:
                print(f"⚠️ WARNUNG: Modelldatei nicht gefunden unter {model_path}")


    def get_action(self, obs):
        """Wählt eine Aktion basierend auf der Beobachtung."""
        
        # --- ZUKÜNFTIGE VERWENDUNG DES ENCODERS ---
        # Hier würdest du den Encoder deines World Models verwenden, um das Bild
        # in einen kompakten Zustand (latent state) umzuwandeln.
        with torch.no_grad():
            # Konvertiere die Beobachtung (obs) in einen Tensor mit Batch-Dimension
            # und sende ihn an das richtige Gerät (CPU/GPU).
            input_tensor = obs.unsqueeze(0).to(self.device)
            
            # Hier wird das Bild durch den Encoder geschickt.
            latent_state = self.world_model.encoder(input_tensor)
        
        # Der 'latent_state' würde dann an die nächsten Teile des Agenten
        # (z.B. ein RNN und einen Actor) weitergegeben, um eine intelligente
        # Aktion zu bestimmen.
        # print(f"Latent State Shape: {latent_state.shape}") # Zum Debuggen
        
        # Da wir noch keinen Actor haben, geben wir vorerst eine zufällige Aktion zurück.
        num_actions = len(self.config.DISCRETE_ACTIONS)
        return random.randint(0, num_actions - 1)