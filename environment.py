# environment.py
import MalmoPython
import time
import torch
import numpy as np
from PIL import Image
import logging
from utils import VirtualController  # NEU!
import random

class MalmoEnvironment:
    def __init__(self, config, mission_xml, port=10000):  # NEU: port parameter
        self.config = config
        self.port = port  # NEU
        
        self.agent_host = MalmoPython.AgentHost()
        
        # NEU: Client Pool mit spezifischem Port
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", port))
        
        # --- NEUER ABSCHNITT: Random Seed einfügen ---
        world_seed = str(random.randint(-2**63, 2**63 - 1))
        # --- ENDE NEUER ABSCHNITT ---

        self.mission_spec = MalmoPython.MissionSpec(mission_xml, True)
        self.mission_spec.setWorldSeed(world_seed)
        self.mission_record = MalmoPython.MissionRecordSpec()
        
        self.controller = VirtualController(config)
        
        self.logger = logging.getLogger(f"MalmoEnv_Port{port}")  # NEU: Port im Logger
        # self.logger.setLevel(logging.DEBUG)
        self.logger.setLevel(logging.INFO)  # Setze Level auf INFO, um Debug-Ausgaben zu vermeiden
        self.logger.info(f"🎮 Environment initialized on port {port}")
        self.logger.info(f"🎮 Virtual Controller initialisiert mit {len(config.DISCRETE_ACTIONS)} Aktionen")

    def _get_observation(self, world_state):
        if world_state.number_of_video_frames_since_last_state > 0:
            frame = world_state.video_frames[-1]
            image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels))
            image = image.resize((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
            image_np = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            self.logger.debug(f"📷 Bild empfangen: {frame.width}x{frame.height} → resized auf {self.config.IMAGE_WIDTH}x{self.config.IMAGE_HEIGHT}")
            return torch.from_numpy(image_np)
        else:
            self.logger.debug("⚠️ Keine neuen Video-Frames empfangen.")
        return None

    def reset(self):
        self.logger.info("🔄 Starte neue Mission...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"📡 Versuch {attempt+1}/{max_retries} Mission zu starten...")

                # ================================================================
                # HIER IST DIE FINALE KORREKTUR:
                # Wir übergeben die Rolle (0) und eine leere Experiment-ID ("")
                self.agent_host.startMission(self.mission_spec, self.client_pool, self.mission_record, 0, "")
                # ================================================================


                # self.agent_host.startMission(self.mission_spec, self.mission_record)
                break  # Erfolg → Schleife verlassen
            except RuntimeError as e:
                self.logger.error(f"❌ Mission konnte nicht gestartet werden: {e}")
                if attempt == max_retries - 1:
                    raise
                else:
                    time.sleep(2)  # kurz warten und nochmal versuchen

        # Jetzt wie gewohnt auf Missionsbeginn warten
        world_state = self.agent_host.getWorldState()
        self.logger.debug("⏳ Warte auf Missionsbeginn...")
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            for err in world_state.errors:
                self.logger.error(f"💥 Mission Error: {err.text}")

        self.logger.info("✅ Mission gestartet!")
        obs = self._get_observation(world_state)
        while obs is None:
            self.logger.debug("⏳ Warte auf erste Observation...")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()
            obs = self._get_observation(world_state)
        self.logger.info("📥 Erste Observation empfangen.")
        return obs


    def step(self, action_idx):

        # NEU: Konvertiere diskrete Action zu Malmo Commands
        commands = self.controller.action_to_commands(action_idx)

        # Sende alle Commands
        for cmd in commands:
            self.logger.debug(f"🎮 Sende Command: {cmd}")
            self.agent_host.sendCommand(cmd)
        
        # Kurze Pause damit Action wirkt
        time.sleep(0.2)

        world_state = self.agent_host.getWorldState()
        reward = sum(r.getValue() for r in world_state.rewards)
        done = not world_state.is_mission_running
        self.logger.debug(f"🎯 Reward: {reward} | Done: {done}")

        next_obs = self._get_observation(world_state)
        
        if not done and next_obs is None:
            self.logger.debug("⚠️ Keine Observation, erneuter Versuch...")
            while next_obs is None and world_state.is_mission_running:
                time.sleep(0.1)
                world_state = self.agent_host.getWorldState()
                reward += sum(r.getValue() for r in world_state.rewards)
                done = not world_state.is_mission_running
                if done:
                    break
                next_obs = self._get_observation(world_state)

        return next_obs, reward, done
