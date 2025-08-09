# parallel_collector.py
# Sammelt Daten mit mehreren Agents parallel

import multiprocessing as mp
import time
import logging
import sys
import os
from pathlib import Path

# Importiere deine Module
import MalmoPython
from utils import Config, HDF5ReplayBuffer
from environment import MalmoEnvironment
from model import DreamerAgent

def setup_logging(agent_id):
    """Setup logging für jeden Agent separat"""
    logger = logging.getLogger(f"Agent_{agent_id}")
    logger.setLevel(logging.INFO)
    
    # File handler für jeden Agent
    fh = logging.FileHandler(f"agent_{agent_id}.log")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        f'[Agent {agent_id}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def run_agent(agent_id, port, num_episodes=100, start_episode=0):
    """
    Läuft als separater Prozess für einen Agent
    
    Args:
        agent_id: Unique ID für diesen Agent (0, 1, 2, ...)
        port: Minecraft Port (10000, 10001, ...)
        num_episodes: Anzahl Episoden die dieser Agent sammeln soll
        start_episode: Start-Nummer für Episode-Zählung
    """
    
    # Setup für diesen Prozess
    logger = setup_logging(agent_id)
    logger.info(f"🚀 Starting on port {port}")
    
    # Config und Pfade
    config = Config()
    buffer_path = f"replay_buffer_agent_{agent_id}.hdf5"
    
    # Mission XML laden
    with open("mission.xml", "r") as f:
        mission_xml = f.read()
    
    # Modifiziere Agent Name in XML
    mission_xml = mission_xml.replace(
        "<Name>Dreamer</Name>", 
        f"<Name>Dreamer_{agent_id}</Name>"
    )
    
    try:
        # Erstelle Environment mit spezifischem Port
        env = MalmoEnvironment(config, mission_xml, port=port)
        agent = DreamerAgent(config)
        
        # HDF5 Buffer für diesen Agent
        with HDF5ReplayBuffer(
            capacity=100000, 
            save_path=buffer_path, 
            mode='a'
        ) as replay_buffer:
            
            logger.info(f"📦 Buffer: {buffer_path}")
            logger.info(f"🎮 Starting {num_episodes} episodes")
            
            for episode in range(num_episodes):
                actual_episode = start_episode + episode
                episode_start = time.time()
                
                logger.info(f"🎬 Episode {actual_episode + 1}")
                
                # Reset mit Retry-Logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        obs = env.reset()
                        break
                    except Exception as e:
                        logger.warning(f"Reset failed, retry {retry+1}/{max_retries}: {e}")
                        time.sleep(5)
                else:
                    logger.error(f"Failed to reset after {max_retries} attempts")
                    continue
                
                done = False
                total_reward = 0.0
                step = 0
                
                # Episode Loop
                while not done:
                    step += 1
                    
                    # Get action
                    action = agent.get_action(obs)
                    
                    # Step
                    next_obs, reward, done = env.step(action)
                    
                    # Process reward
                    try:
                        reward_val = float(reward) if reward is not None else 0.0
                    except:
                        reward_val = 0.0
                    
                    total_reward += reward_val
                    
                    # Add to buffer
                    if next_obs is not None:
                        replay_buffer.add(obs, action, reward_val, next_obs, done)
                    
                    obs = next_obs
                    
                    # Periodic logging
                    if step % 100 == 0:
                        logger.debug(f"Step {step}, Reward: {total_reward:.2f}")
                    
                    # Safety timeout
                    if step > 10000:
                        logger.warning("Episode timeout!")
                        done = True
                
                # Episode complete
                duration = time.time() - episode_start
                logger.info(
                    f"✅ Episode {actual_episode + 1} complete: "
                    f"Reward={total_reward:.2f}, Steps={step}, "
                    f"Time={duration:.1f}s, Buffer={len(replay_buffer)}"
                )
                
                # Periodic save
                if (episode + 1) % 10 == 0:
                    replay_buffer.file.flush()
                    logger.info(f"💾 Buffer flushed")
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    finally:
        logger.info(f"🏁 Agent {agent_id} finished. Buffer: {buffer_path}")

def main():
    """Hauptfunktion für paralleles Sammeln"""
    
    # Configuration
    NUM_AGENTS = 2
    TOTAL_EPISODES = 1000
    BASE_PORT = 10000
    
    episodes_per_agent = TOTAL_EPISODES // NUM_AGENTS
    
    print("="*60)
    print(f"🚀 PARALLEL DATA COLLECTION")
    print(f"   Agents: {NUM_AGENTS}")
    print(f"   Episodes per Agent: {episodes_per_agent}")
    print(f"   Total Episodes: {TOTAL_EPISODES}")
    print(f"   Ports: {BASE_PORT} - {BASE_PORT + NUM_AGENTS - 1}")
    print("="*60)
    
    # Check if Minecraft clients are running
    print("\n⚠️  Make sure Minecraft clients are running!")
    print("   Run: ./HelperSkripte/launch_multi_minecraft.sh")
    input("   Press Enter when ready...")
    
    # Start processes
    processes = []
    for i in range(NUM_AGENTS):
        port = BASE_PORT + i
        start_episode = i * episodes_per_agent
        
        p = mp.Process(
            target=run_agent,
            args=(i, port, episodes_per_agent, start_episode)
        )
        p.start()
        processes.append(p)
        
        print(f"✅ Started Agent {i} on port {port}")
        time.sleep(2)  # Stagger starts
    
    print("\n📊 All agents running. Monitor with:")
    print("   tail -f agent_*.log")
    print("   ./HelperSkripte/check_clients.sh")
    
    try:
        # Wait for all to complete
        for i, p in enumerate(processes):
            p.join()
            print(f"✅ Agent {i} completed")
    except KeyboardInterrupt:
        print("\n🛑 Stopping all agents...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
    
    print("\n🏁 Data collection complete!")
    print("📦 Buffers created:")
    for i in range(NUM_AGENTS):
        buffer_file = f"replay_buffer_agent_{i}.hdf5"
        if Path(buffer_file).exists():
            size = Path(buffer_file).stat().st_size / (1024*1024)
            print(f"   {buffer_file}: {size:.1f} MB")
    
    print("\n💡 Next steps:")
    print("   1. Merge buffers: python merge_buffers.py")
    print("   2. Stop Minecraft: ./HelperSkripte/stop_all_minecraft.sh")

if __name__ == "__main__":
    # Setze multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()