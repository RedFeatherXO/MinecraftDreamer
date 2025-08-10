# model.py - KORRIGIERT

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path

# --- Konstanten (könnten auch aus einer Config kommen) ---
IMG_CHANNELS = 3
# HINWEIS: Die Bildgröße muss mit der Größe übereinstimmen, mit der trainiert wurde.
# Laut deinem Kaggle-Notebook ist das 64x64.
TARGET_HEIGHT = 64
TARGET_WIDTH = 64

# --- WorldModel Klasse (ANGEPASST) ---
# Jetzt mit flexiblem latent_dim
class WorldModel(nn.Module):
    # ==============================================================================
    # ÄNDERUNG: latent_dim wird jetzt als Parameter übergeben
    def __init__(self, img_channels=3, latent_dim=128):
    # ==============================================================================
        super(WorldModel, self).__init__()
        # Der Encoder- und Decoder-Teil ist identisch zum Kaggle-Notebook
        self.encoder_conv  = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        # Die Größe von 1024 * 2 * 2 ergibt sich aus dem Encoder bei 64x64 Input
        self.fc_mu = nn.Linear(1024 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 2 * 2, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 1024 * 2 * 2)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), 
            nn.Sigmoid()
        )
        

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 1024, 2, 2)
        return self.decoder_conv(x)

    def forward(self, x):
        # HINWEIS: forward() gibt jetzt 3 Werte zurück, passend zum Training
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# --- DreamerAgent Klasse (ANGEPASST) ---
# Muss den latent_dim an das WorldModel weitergeben
class DreamerAgent:
    def __init__(self, config, pretrained_world_model_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ==============================================================================
        # ÄNDERUNG: Extrahiere latent_dim aus der Config und übergebe es
        # (Wir geben einen Standardwert von 128 an, falls es nicht in der Config steht)
        latent_dim = getattr(config, "LATENT_DIM", 128)
        self.world_model = WorldModel(latent_dim=latent_dim).to(self.device)
        # ==============================================================================

        if pretrained_world_model_path:
            model_path = Path(pretrained_world_model_path)
            if model_path.exists():
                print(f"🔄 Lade vortrainiertes WorldModel von: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        self.world_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.world_model.load_state_dict(checkpoint)
                    
                    self.world_model.eval()
                    print(f"✅ WorldModel erfolgreich geladen und im eval()-Modus.")
                except Exception as e:
                    print(f"❌ FEHLER beim Laden des Modells: {e}")
            else:
                print(f"⚠️ WARNUNG: Modelldatei nicht gefunden unter {model_path}")

    def get_action(self, obs):
        # Die Logik hier muss angepasst werden, da encode() jetzt 2 Werte zurückgibt
        with torch.no_grad():
            input_tensor = obs.unsqueeze(0).to(self.device)
            # ==============================================================================
            # ÄNDERUNG: encode gibt mu und logvar zurück
            mu, logvar = self.world_model.encode(input_tensor)
            # Für die Aktion nehmen wir einfach den Mittelwert (mu) als latent state
            latent_state = mu
            # ==============================================================================
        
        num_actions = len(self.config.DISCRETE_ACTIONS)
        return random.randint(0, num_actions - 1)