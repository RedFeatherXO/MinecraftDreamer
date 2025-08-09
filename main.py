# main.py
# Das Hauptskript, das die Umgebung, den Agenten und die Trainingsschleife initialisiert.
# + zusätzliche, freundliche Debug-Logs mit Emojis.

import time
import random
import logging
import traceback
import torch

# Optional: numpy nur für bessere Obs-Summaries, funktioniert auch ohne numpy.
try:
    import numpy as np
except Exception:
    np = None

# Importiere unsere eigenen Module
from utils import Config, HDF5ReplayBuffer
from environment import MalmoEnvironment
from model import DreamerAgent

import matplotlib.pyplot as plt



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

    # Matplotlib Setup
    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("🎮 Agent View (RGB)")
    ax2.set_title("📊 Reward History")
    img_plot = None
    reward_history = []

    # Logging konfigurieren (kontrollierbar über config.DEBUG, falls vorhanden)
    level = logging.DEBUG if getattr(config, "DEBUG", True) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("dreamer_main")
    logger.setLevel(level)

    # Kurze Config-Übersicht (sicher abrufen mit getattr)
    logger.info(f"🧭 Konfiguration: IMAGE_WIDTH={getattr(config,'IMAGE_WIDTH', 'n/a')}, "
                f"IMAGE_HEIGHT={getattr(config,'IMAGE_HEIGHT', 'n/a')}, "
                f"BATCH_SIZE={getattr(config,'BATCH_SIZE', 'n/a')}, "
                f"REPLAY_CAPACITY={getattr(config,'REPLAY_CAPACITY', 'n/a')}")

    with open("mission.xml", "r") as f:
        mission_xml = f.read()

    env = MalmoEnvironment(config, mission_xml)
    agent = DreamerAgent(config)
    replay_buffer = HDF5ReplayBuffer(getattr(config, "REPLAY_CAPACITY", 100000))

    # Optional: reproducible Runs (wenn config.SEED gesetzt ist)
    seed = getattr(config, "SEED", None)
    if seed is not None:
        random.seed(seed)
        try:
            if np is not None:
                np.random.seed(seed)
        except Exception:
            pass
        logger.info(f"🔢 Seed gesetzt: {seed}")

    logger.info("🚀 Stelle sicher, dass der Malmo-Client in einem anderen Terminal läuft!")
    time.sleep(3)

    num_episodes = getattr(config, "NUM_EPISODES", 10000)
    enable_training = getattr(config, "ENABLE_TRAINING", False)  # Standard: False, damit Verhalten nicht geändert wird
    train_every = getattr(config, "TRAIN_EVERY", 50)

    with HDF5ReplayBuffer(capacity=100000, mode='a') as replay_buffer:
        for episode in range(num_episodes):
            episode_start = time.time()
            logger.info(f"🎬 Starte Episode {episode + 1}/{num_episodes}")
            obs = env.reset()
            done = False
            total_reward = 0.0
            step = 0
            
            try:
                while not done:
                    step += 1
                    action = agent.get_action(obs)
                    next_obs, reward, done = env.step(action)
                    
                    # Safety cast
                    try:
                        reward_val = float(reward) if reward is not None else 0.0
                    except Exception:
                        reward_val = 0.0
                    total_reward += reward_val
                    
                    if next_obs is not None:
                        # NUR EINMAL hinzufügen!
                        try:
                            replay_buffer.add(obs, action, reward_val, next_obs, done)
                        except Exception as e:
                            logger.exception(f"⚠️ Fehler beim Buffer-Add: {e}")
                        
                        # Visualisierung alle 20 Steps
                        if step % 20 == 0:
                            buffer_len = len(replay_buffer)
                            
                            # ... matplotlib code ...
                            
                            logger.info(f"🧾 Buffer: {buffer_len} samples | "
                                      f"Step {step} | Reward: {total_reward:.2f}")
                    
                    obs = next_obs
                    
                    # Training (optional)
                    if enable_training and len(replay_buffer) > config.BATCH_SIZE:
                        if step % train_every == 0:
                            logger.info(f"⚙️ Training...")
                            # agent.train(replay_buffer)
                
                # Episode Ende
                ep_dur = time.time() - episode_start
                logger.info(f"✅ Episode {episode + 1} beendet — "
                          f"Reward: {total_reward:.2f} | "
                          f"Steps: {step} | Dauer: {ep_dur:.2f}s")
                
            except KeyboardInterrupt:
                logger.warning("🛑 Abgebrochen")
                break
            except Exception:
                logger.exception("❗ Fehler in Episode")
                continue
    
    # Buffer wird automatisch geschlossen durch 'with'!
    logger.info("🏁 Training beendet!")