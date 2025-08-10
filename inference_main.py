# inference_main.py
# Dieses Skript führt den Agenten aus und visualisiert die Inferenz in Echtzeit.

import time
import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Importiere deine Projekt-Module
import MalmoPython
from environment import MalmoEnvironment
from model import DreamerAgent # Wir brauchen den Agenten als Hülle für das WorldModel
from utils import Config

# --- NEU: Konfiguration ---
# Passe diesen Pfad zu deiner heruntergeladenen Modelldatei an
PRETRAINED_MODEL_PATH = "kaggle_top1_models/models/rank1_loss351.414028_BS512_LR0.0001_LD512_adamw.pth"

# --- NEU: Visualisierungs-Funktion ---
# Diese Funktion erstellt und aktualisiert das Inferenz-Fenster
def visualize_inference(original_obs, reconstructed_obs):
    # Konvertiere Tensoren zu anzeigbaren Numpy-Bildern
    original_np = original_obs.cpu().numpy().transpose(1, 2, 0)
    reconstructed_np = reconstructed_obs.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Initialisiere die Plots, falls sie noch nicht existieren
    if not hasattr(visualize_inference, "fig"):
        plt.ion() # Interaktiven Modus aktivieren (wichtig für Live-Updates)
        visualize_inference.fig, visualize_inference.axes = plt.subplots(1, 2, figsize=(10, 5))
        visualize_inference.img_orig = visualize_inference.axes[0].imshow(original_np)
        visualize_inference.img_recon = visualize_inference.axes[1].imshow(reconstructed_np)
        visualize_inference.axes[0].set_title("Original (aus Minecraft)")
        visualize_inference.axes[1].set_title("Rekonstruktion (vom Modell)")
        visualize_inference.axes[0].axis("off")
        visualize_inference.axes[1].axis("off")
        plt.suptitle("Live-Inferenz des World Models")
    
    # Aktualisiere die Bilddaten in den bestehenden Plots
    visualize_inference.img_orig.set_data(original_np)
    visualize_inference.img_recon.set_data(reconstructed_np)
    visualize_inference.fig.canvas.draw()
    visualize_inference.fig.canvas.flush_events()
    plt.pause(0.01) # Kurze Pause, damit das Fenster aktualisiert wird


def main():
    """Führt einen Agenten für eine Episode aus und visualisiert die Inferenz."""
    
    # Setup (wie in main.py)
    logging.basicConfig(level=logging.INFO)
    config = Config()
    # ==============================================================================
    # NEU: Setze den latent_dim in der Config, passend zum Modellnamen
    # Der Modellname enthält "LD256", also setzen wir LATENT_DIM auf 256
    config.LATENT_DIM = 256
    # ==============================================================================
    
    # Mission XML laden
    try:
        with open("mission.xml", "r") as f:
            mission_xml = f.read()
    except FileNotFoundError:
        logging.error("❌ mission.xml nicht gefunden. Stelle sicher, dass sie im selben Ordner liegt.")
        return

    # Umgebung und Agent initialisieren
    env = MalmoEnvironment(config, mission_xml)
    
    # --- GEÄNDERT: Lade den Agenten mit dem vortrainierten Modell ---
    # Wir übergeben den Pfad an den Konstruktor des Agenten
    # (Dafür musst du deinen DreamerAgent anpassen, wie in meiner vorherigen Antwort gezeigt)
    agent = DreamerAgent(config, pretrained_world_model_path=PRETRAINED_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- ENTFERNT: Replay Buffer wird nicht mehr benötigt ---
    # replay_buffer = HDF5ReplayBuffer(...)

    logging.info("🚀 Starte Inferenz-Lauf für eine Episode...")
    
    try:
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            # Agent wählt eine Aktion (z.B. zufällig)
            action = agent.get_action(obs) 
            
            # Umgebung führt die Aktion aus
            next_obs, reward, done = env.step(action)
            step_count += 1
            
            if next_obs is not None:
                with torch.no_grad():
                    # Das Modell braucht einen Batch-Dimension (unsqueeze)
                    input_tensor = next_obs.unsqueeze(0).to(device)
                    reconstructed_obs, _, _ = agent.world_model(input_tensor)
                
                visualize_inference(next_obs, reconstructed_obs)
                obs = next_obs

            # --- ENTFERNT: replay_buffer.add(...) wird nicht mehr benötigt ---
            
            if step_count > 1000: # Sicherheits-Timeout
                logging.warning("Episoden-Timeout nach 1000 Schritten.")
                done = True

    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}", exc_info=True)
    finally:
        plt.ioff() # Interaktiven Modus am Ende deaktivieren
        plt.show() # Zeige das letzte Bild an, bevor das Skript endet
        logging.info("Inferenz-Lauf beendet.")


if __name__ == "__main__":
    # Stelle sicher, dass der Pfad zum Modell existiert
    if not Path(PRETRAINED_MODEL_PATH).exists():
        print(f"❌ Fehler: Modelldatei nicht gefunden unter '{PRETRAINED_MODEL_PATH}'")
        print("Bitte passe den Pfad oben im Skript an.")
    else:
        main()