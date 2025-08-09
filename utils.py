# utils.py - Erweiterte Version mit virtueller Tastatur/Maus

import random
from collections import deque
import numpy as np

import h5py
import torch
from pathlib import Path
import logging

class Config:
    """Eine Klasse, um alle Hyperparameter an einem Ort zu sammeln."""
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84
    
    # Discrete Action Space (wie Tastatur + Maus)
    # Der Agent wählt eine dieser diskreten Aktionen
    DISCRETE_ACTIONS = [
        "forward",       # W
        "backward",      # S
        "strafe_left",   # A
        "strafe_right",  # D
        "turn_left",     # Maus links
        "turn_right",    # Maus rechts
        "look_up",       # Maus hoch
        "look_down",     # Maus runter
        "jump",          # Spacebar
        "attack",        # Linksklick
        "use",           # Rechtsklick
        "sprint",        # Shift
        "idle"           # Nichts tun
    ]
    
    # Bewegungsgeschwindigkeiten (wie Maus-Sensitivität)
    MOVE_SPEED = 0.8        # Laufgeschwindigkeit
    TURN_SPEED = 0.5        # Maus-X Sensitivität  
    PITCH_SPEED = 0.3       # Maus-Y Sensitivität
    
    # Trainings-Parameter
    REPLAY_CAPACITY = 100000
    BATCH_SIZE = 50
    LEARNING_RATE = 1e-4
    LATENT_DIM = 50

class VirtualController:
    """Simuliert Tastatur und Maus für den Agenten - OPTIMIERT"""
    
    def __init__(self, config):
        self.config = config
        # Track was gerade aktiv ist
        self.active_states = {
            "jump": False,
            "attack": False,
            "use": False
        }
        
    def action_to_commands(self, action_idx):
        """Konvertiert diskrete Action zu Malmo Commands"""
        action_name = self.config.DISCRETE_ACTIONS[action_idx]
        commands = []
        
        # Bewegungs-Commands (werden nur gesendet wenn != 0)
        if action_name == "forward":
            commands.append(f"move {self.config.MOVE_SPEED}")
        elif action_name == "backward":
            commands.append(f"move -{self.config.MOVE_SPEED}")
        elif action_name == "strafe_left":
            commands.append(f"strafe -{self.config.MOVE_SPEED}")
        elif action_name == "strafe_right":
            commands.append(f"strafe {self.config.MOVE_SPEED}")
        elif action_name == "turn_left":
            commands.append(f"turn -{self.config.TURN_SPEED}")
        elif action_name == "turn_right":
            commands.append(f"turn {self.config.TURN_SPEED}")
        elif action_name == "look_up":
            commands.append(f"pitch -{self.config.PITCH_SPEED}")
        elif action_name == "look_down":
            commands.append(f"pitch {self.config.PITCH_SPEED}")
        
        # Toggle-Commands (nur senden wenn sich Status ändert)
        elif action_name == "jump":
            if not self.active_states["jump"]:
                commands.append("jump 1")
                self.active_states["jump"] = True
        elif action_name == "attack":
            if not self.active_states["attack"]:
                commands.append("attack 1")
                self.active_states["attack"] = True
        elif action_name == "use":
            if not self.active_states["use"]:
                commands.append("use 1")
                self.active_states["use"] = True
                
        # Sprint modifier
        elif action_name == "sprint":
            commands.append(f"move {self.config.MOVE_SPEED * 1.5}")
        
        # Bei "idle" oder anderen Actions: Toggle-States zurücksetzen
        else:
            # Reset toggle states wenn nicht mehr aktiv
            if self.active_states["jump"]:
                commands.append("jump 0")
                self.active_states["jump"] = False
            if self.active_states["attack"]:
                commands.append("attack 0")
                self.active_states["attack"] = False
            if self.active_states["use"]:
                commands.append("use 0")
                self.active_states["use"] = False
        
        return commands

# utils.py - Production-ready HDF5 Buffer mit FIX
class HDF5ReplayBuffer:
    """Effizienter Replay Buffer mit HDF5 Backend"""
    
    def __init__(self, capacity=100000, obs_shape=(3, 84, 84), 
                 save_path="replay_buffer.hdf5", mode='w'):
        """
        mode: 'w' = new file, 'a' = append to existing
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.ptr = 0
        self.size = 0
        self.save_path = Path(save_path)
        
        # NEU: Flag ob bereits geschlossen
        self._closed = False
        
        # Logger
        self.logger = logging.getLogger("HDF5Buffer")
        
        # Erstelle parent directory falls nötig
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if mode == 'a' and self.save_path.exists():
            # Lade existierenden Buffer
            self.file = h5py.File(self.save_path, 'a')
            self.size = self.file.attrs.get('size', 0)
            self.ptr = self.file.attrs.get('ptr', 0)
            self.logger.info(f"📂 Geladen: {self.size} samples von {self.save_path}")
        else:
            # Neuer Buffer
            self.file = h5py.File(self.save_path, 'w')
            
            # Datasets mit Compression!
            self.file.create_dataset(
                'obs', 
                (capacity, *obs_shape), 
                dtype='f4',  # float32
                chunks=(1, *obs_shape),  # Optimiert für einzelne Samples
                compression='gzip',
                compression_opts=1  # Level 1 = schnell
            )
            self.file.create_dataset('action', (capacity,), dtype='i4')
            self.file.create_dataset('reward', (capacity,), dtype='f4')
            self.file.create_dataset(
                'next_obs', 
                (capacity, *obs_shape), 
                dtype='f4',
                chunks=(1, *obs_shape),
                compression='gzip',
                compression_opts=1
            )
            self.file.create_dataset('done', (capacity,), dtype='bool')
            
            # Speichere Metadaten
            self.file.attrs['capacity'] = capacity
            self.file.attrs['obs_shape'] = obs_shape
            self.file.attrs['size'] = 0
            self.file.attrs['ptr'] = 0
            
            self.logger.info(f"💾 Neuer Buffer erstellt: {self.save_path}")
    
    def add(self, obs, action, reward, next_obs, done):
        """Fügt eine Transition hinzu"""
        # Check ob bereits geschlossen
        if self._closed:
            return
            
        # Konvertiere zu numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()
        
        # Schreibe auf Disk
        idx = self.ptr
        self.file['obs'][idx] = obs
        self.file['action'][idx] = action
        self.file['reward'][idx] = reward
        self.file['next_obs'][idx] = next_obs
        self.file['done'][idx] = done
        
        # Update pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Update Metadaten (wichtig für Recovery!)
        self.file.attrs['size'] = self.size
        self.file.attrs['ptr'] = self.ptr
        
        # Auto-flush alle 1000 samples für Sicherheit
        if self.ptr % 1000 == 0:
            self.file.flush()
    
    def sample(self, batch_size):
        """Sampelt einen Batch"""
        if self.size < batch_size:
            raise ValueError(f"Buffer hat nur {self.size} samples, brauche {batch_size}")
        
        # Random indices
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Lade Batch (HDF5 ist smart und cached!)
        batch = (
            torch.from_numpy(self.file['obs'][indices]),
            torch.from_numpy(np.array(self.file['action'][indices])),
            torch.from_numpy(np.array(self.file['reward'][indices])),
            torch.from_numpy(self.file['next_obs'][indices]),
            torch.from_numpy(np.array(self.file['done'][indices]))
        )
        
        return batch
    
    def get_sequential_batch(self, start_idx, length):
        """Holt sequentielle Daten (wichtig für RNN training!)"""
        end_idx = min(start_idx + length, self.size)
        indices = range(start_idx, end_idx)
        
        return (
            torch.from_numpy(self.file['obs'][indices]),
            torch.from_numpy(np.array(self.file['action'][indices])),
            torch.from_numpy(np.array(self.file['reward'][indices])),
            torch.from_numpy(self.file['next_obs'][indices]),
            torch.from_numpy(np.array(self.file['done'][indices]))
        )
    
    def save_stats(self):
        """Speichert Statistiken"""
        if self.size > 0:
            rewards = self.file['reward'][:self.size]
            self.file.attrs['mean_reward'] = np.mean(rewards)
            self.file.attrs['std_reward'] = np.std(rewards)
            self.file.attrs['min_reward'] = np.min(rewards)
            self.file.attrs['max_reward'] = np.max(rewards)
            
            self.logger.info(f"📊 Stats: Mean reward = {np.mean(rewards):.2f}")
    
    def close(self):
        """Schließt File ordentlich"""
        # NEU: Check ob bereits geschlossen
        if self._closed:
            return
            
        try:
            self.save_stats()
            self.file.flush()
            self.file.close()
            self._closed = True  # Markiere als geschlossen
            self.logger.info(f"💾 Buffer geschlossen: {self.size} samples")
        except Exception as e:
            # Falls schon geschlossen oder anderer Fehler
            self.logger.debug(f"Close fehlgeschlagen (vermutlich schon geschlossen): {e}")
            self._closed = True
    
    def __len__(self):
        return self.size
    
    def __del__(self):
        """Cleanup bei Garbage Collection"""
        # NEU: Besserer Check ob cleanup nötig
        if hasattr(self, '_closed') and not self._closed:
            if hasattr(self, 'file'):
                try:
                    # Versuche zu schließen, ignoriere alle Fehler
                    self.close()
                except:
                    pass
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - schließt den Buffer"""
        self.close()