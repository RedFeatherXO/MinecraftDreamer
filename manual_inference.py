# manual_inference.py
# Ermöglicht die manuelle Steuerung des Agenten mit Live-Visualisierung der Modell-Inferenz.

import time
import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from pynput import keyboard # NEU: Bibliothek zur Tastaturabfrage

# Importiere deine Projekt-Module
import MalmoPython
from environment import MalmoEnvironment
from model import DreamerAgent
from utils import Config

# --- Konfiguration ---
# Passe diesen Pfad zu deiner besten heruntergeladenen Modelldatei an
PRETRAINED_MODEL_PATH = "kaggle_top3_models/models/rank1_loss299.361228_BS512_LR0.0005_LD256_adamw.pth"

# --- NEU: Controller-Klasse für manuelle Steuerung ---
class KeyboardController:
    def __init__(self, config):
        self.config = config
        self.active_keys = set()
        
        # Mapping von Tasten zu Action-Indizes (gemäß deiner Config)
        self.key_to_action_idx = {
            'w': self.config.DISCRETE_ACTIONS.index('forward'),
            's': self.config.DISCRETE_ACTIONS.index('backward'),
            'a': self.config.DISCRETE_ACTIONS.index('strafe_left'),
            'd': self.config.DISCRETE_ACTIONS.index('strafe_right'),
            keyboard.Key.up: self.config.DISCRETE_ACTIONS.index('look_up'),
            keyboard.Key.down: self.config.DISCRETE_ACTIONS.index('look_down'),
            keyboard.Key.left: self.config.DISCRETE_ACTIONS.index('turn_left'),
            keyboard.Key.right: self.config.DISCRETE_ACTIONS.index('turn_right'),
            keyboard.Key.space: self.config.DISCRETE_ACTIONS.index('jump'),
        }
        self.idle_action_idx = self.config.DISCRETE_ACTIONS.index('idle')

        # Starte den Listener in einem separaten Thread
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True # Erlaubt dem Hauptprogramm, sich zu beenden
        listener.start()
        print("⌨️ Keyboard-Controller aktiv. Steuerung mit W,A,S,D und Pfeiltasten. Leertaste für Sprung.")

    def _on_press(self, key):
        self.active_keys.add(key)

    def _on_release(self, key):
        # Konvertiere Zeichen zu Key-Objekten falls nötig
        if hasattr(key, 'char') and key.char in ['w', 's', 'a', 'd']:
             self.active_keys.discard(key.char) # Funktioniert nicht immer, daher Sicherheits-Check
        self.active_keys.discard(key)


    def get_action_index(self):
        # Gehe die aktiven Tasten durch und gib die erste gefundene Aktion zurück
        # (Priorisierung könnte hier noch verfeinert werden)
        for key in self.active_keys:
            # Pynput gibt für Buchstabentasten ein Objekt mit .char Attribut
            key_val = key.char if hasattr(key, 'char') else key
            if key_val in self.key_to_action_idx:
                return self.key_to_action_idx[key_val]
        
        # Wenn keine relevante Taste gedrückt ist, "idle" zurückgeben
        return self.idle_action_idx

# --- Die Visualisierungs-Funktion (unverändert) ---
def visualize_inference(original_obs, reconstructed_obs):
    # ... (exakt wie im vorherigen Skript)
    original_np = original_obs.cpu().numpy().transpose(1, 2, 0)
    reconstructed_np = reconstructed_obs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    if not hasattr(visualize_inference, "fig"):
        plt.ion()
        visualize_inference.fig, visualize_inference.axes = plt.subplots(1, 2, figsize=(10, 5))
        visualize_inference.img_orig = visualize_inference.axes[0].imshow(original_np)
        visualize_inference.img_recon = visualize_inference.axes[1].imshow(reconstructed_np)
        visualize_inference.axes[0].set_title("Original (Deine Steuerung)")
        visualize_inference.axes[1].set_title("Rekonstruktion (vom Modell)")
        [ax.axis("off") for ax in visualize_inference.axes]
        plt.suptitle("Live-Inferenz mit manueller Steuerung")
    
    visualize_inference.img_orig.set_data(original_np)
    visualize_inference.img_recon.set_data(reconstructed_np)
    visualize_inference.fig.canvas.draw()
    visualize_inference.fig.canvas.flush_events()
    plt.pause(0.01)


def main():
    logging.basicConfig(level=logging.INFO)
    config = Config()
    # ==============================================================================
    # NEU: Setze den latent_dim in der Config, passend zum Modellnamen
    # Der Modellname enthält "LD256", also setzen wir LATENT_DIM auf 256
    config.LATENT_DIM = 256
    # ==============================================================================
    
    with open("/home/meik/Downloads/Malmo-0.37.0-Linux-Ubuntu-18.04-64bit_withBoost_Python3.6/Python_Examples/mission.xml", "r") as f:
        mission_xml = f.read()

    env = MalmoEnvironment(config, mission_xml)
    agent = DreamerAgent(config, pretrained_world_model_path=PRETRAINED_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- NEU: Initialisiere den Keyboard-Controller ---
    manual_controller = KeyboardController(config)

    logging.info("🚀 Starte manuellen Inferenz-Lauf...")
    
    try:
        obs = env.reset()
        done = False

        while not done:
            # --- GEÄNDERT: Aktion von der Tastatur statt vom Agenten holen ---
            action_idx = manual_controller.get_action_index()
            
            next_obs, reward, done = env.step(action_idx)
            
            if next_obs is not None:
                with torch.no_grad():
                    input_tensor = next_obs.unsqueeze(0).to(device)
                    reconstructed_obs, _, _ = agent.world_model(input_tensor)
                
                visualize_inference(next_obs, reconstructed_obs)
                obs = next_obs
            
            # Kurze Pause, um die CPU nicht voll auszulasten
            time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("Manueller Abbruch durch STRG+C.")
    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}", exc_info=True)
    finally:
        plt.ioff()
        plt.show()
        logging.info("Inferenz-Lauf beendet.")


if __name__ == "__main__":
    if not Path(PRETRAINED_MODEL_PATH).exists():
        print(f"❌ Fehler: Modelldatei nicht gefunden unter '{PRETRAINED_MODEL_PATH}'")
    else:
        main()