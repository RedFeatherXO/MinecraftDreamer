# main2.py - KORRIGIERTE VERSION
import time
import random
import logging
import traceback
import torch
from pathlib import Path
import gc  # Importiere den Garbage Collector

# Optional: numpy für bessere Obs-Summaries
try:
    import numpy as np
except Exception:
    np = None

# Importiere unsere eigenen Module
from utils import Config, HDF5ReplayBuffer
from environment import MalmoEnvironment
from model import DreamerAgent
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# ... (Funktionen summarize und short_repr bleiben unverändert) ...
def summarize(obj):
    """Kurze, sichere Zusammenfassung eines Obs/Objekts für Logs."""
    try:
        if obj is None:
            return "None"
        if np is not None and isinstance(obj, np.ndarray):
            return f"ndarray shape={obj.shape} dtype={obj.dtype}"
        if hasattr(obj, "shape"):
            return f"{type(obj).__name__} shape={getattr(obj, 'shape', 'unknown')}"
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return f"dict keys={keys[:8]}{'...' if len(keys) > 8 else ''}"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__} len={len(obj)}"
        return f"{type(obj).__name__}"
    except Exception as e:
        return f"summary-failed ({e})"

def short_repr(obj, max_len=200):
    """Kurzrepräsentation für Actions / Infos (vermeidet gigantische Logs)."""
    try:
        s = repr(obj)
        return s if len(s) <= max_len else s[:max_len] + "...(truncated)"
    except Exception:
        return f"<{type(obj).__name__}>"

if __name__ == "__main__":
    config = Config()

    # ================================================================
    # NEU: Kommandozeilen-Argumente für den Port einlesen
    parser = argparse.ArgumentParser(description="Malmo Dreamer Agent starten.")
    parser.add_argument('--port', type=int, default=10000, 
                        help='Der Port des Malmo-Clients, mit dem sich verbunden werden soll.')
    args = parser.parse_args()
    # ================================================================

    # --- Matplotlib und Logging Setup (unverändert) ---
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Agent View (Letztes Bild)")
    ax2.set_title("Reward History")
    img_plot = None
    reward_history = []
    level = logging.DEBUG if getattr(config, "DEBUG", True) else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger(f"dreamer_main_p{args.port}") 
    logger.setLevel(level)
    logger.info(f"🧭 Konfiguration: IMAGE_WIDTH={getattr(config,'IMAGE_WIDTH', 'n/a')}, "
                f"BATCH_SIZE={getattr(config,'BATCH_SIZE', 'n/a')}, "
                f"REPLAY_CAPACITY={getattr(config,'REPLAY_CAPACITY', 'n/a')}")
    logger.info(f"🔌 Verbinde mit Malmo-Client auf Port: {args.port}")

    # --- Missions-Liste (unverändert) ---
    mission_files = [
        "missions/mission_flatmap_navigation.xml",
        "missions/mission_flatmap_obstacles.xml",
        "missions/mission_flatmap_collect_diamond.xml",
        "missions/mission_flatmap_night_combat.xml",
        "missions/mission_swamp.xml",
        "missions/mission_taiga.xml",
        "missions/mission_desert.xml",
        "missions/mission_plains.xml",
        "missions/mission_mountain_climb.xml",
        "missions/mission_ocean_swim.xml",
    ]

    # --- Agent einmalig erstellen (unverändert) ---
    agent = DreamerAgent(config)

    # --- KORREKTUR: Pfad für den Buffer erstellen ---
    # Der Pfad wird einmal pro Skript-Lauf festgelegt.

    script_dir = Path(__file__).parent.resolve()
    main_buffer_dir = "replay_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_p{args.port}_{timestamp}"
    buffer_save_path = f"{script_dir}/{main_buffer_dir}/{run_folder_name}/replay_buffer.hdf5"
    logger.info(f"💾 Replay-Daten für diesen Lauf werden gespeichert unter: {buffer_save_path}")

    # --- Seed und Start-Vorbereitungen (unverändert) ---
    if getattr(config, "SEED", None):
        random.seed(config.SEED)
        if np: np.random.seed(config.SEED)
        logger.info(f"🔢 Seed gesetzt: {config.SEED}")

    logger.info("🚀 Stelle sicher, dass der Malmo-Client in einem anderen Terminal läuft!")
    time.sleep(3)

    # --- Haupt-Schleife für die Episoden ---
    total_episodes = getattr(config, "NUM_EPISODES", 10)
    episode_counter = 0

    while episode_counter < total_episodes:
        try:
            # Mission für diese Episode zufällig auswählen und laden
            relative_mission_path = random.choice(mission_files)
            mission_path = script_dir / relative_mission_path

            if not mission_path.exists():
                logger.error(f"❌ Mission-Datei nicht gefunden: {mission_path}, überspringe.")
                episode_counter += 1
                continue

            logger.info(f"📜 Lade Mission {episode_counter + 1}/{total_episodes}: {mission_path.name}")
            with open(mission_path, "r") as f:
                mission_xml = f.read()

            # Umgebung für die aktuelle Mission initialisieren
            env = MalmoEnvironment(config, mission_xml, port=args.port)
            obs = env.reset()
            done = False
            total_reward = 0.0
            step = 0
            episode_start = time.time()

            # --- KORREKTUR: 'with' Block für den Buffer INNERHALB der Schleife ---
            # So wird die Datei nach JEDER Episode sicher gespeichert.
            # Wichtig: mode='a' (append) hängt Daten an die Datei an.
            with HDF5ReplayBuffer(
                capacity=getattr(config, "REPLAY_CAPACITY", 100000),
                save_path=buffer_save_path,
                mode='a'
            ) as replay_buffer:
                while not done:
                    step += 1
                    action = agent.get_action(obs)
                    next_obs, reward, done = env.step(action)
                    reward_val = float(reward) if reward is not None else 0.0
                    total_reward += reward_val

                    if next_obs is not None:
                        replay_buffer.add(obs, action, reward_val, next_obs, done)

                    obs = next_obs
            
            # Episode Ende Log
            ep_dur = time.time() - episode_start
            reward_history.append(total_reward)
            logger.info(f"✅ Episode {episode_counter + 1} beendet — Reward: {total_reward:.2f} | Daten gespeichert.")

        except KeyboardInterrupt:
            logger.warning("🛑 Abbruch durch Nutzer.")
            break  # Schleife beenden
        except Exception:
            logger.exception(f"❗ Schwerer Fehler in Episode (Mission: {mission_path})")
        finally:
            # Aufräumen für die nächste Episode
            if 'env' in locals():
                del env
            gc.collect()
            episode_counter += 1

    logger.info("🏁 Datensammlung beendet!")
    # ================================================================
    # KORREKTUR: Plotte ALLES EINMAL am Ende
    logger.info("📈 Erstelle finalen Plot...")
    
    # Zeige das letzte gesehene Bild
    if obs is not None:
        ax1.imshow(obs.permute(1, 2, 0))

    # Zeige die gesamte Reward History
    ax2.plot(reward_history, 'r-')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.grid(True)
    
    plt.ioff() # Interaktiven Modus aus
    plt.show() # Zeige das fertige Fenster an
    # ================================================================


    