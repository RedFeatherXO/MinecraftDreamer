# view_saved_data.py

import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('/home/meik/Downloads/Malmo-0.37.0-Linux-Ubuntu-18.04-64bit_withBoost_Python3.6/Python_Examples/replay_runs/run_p10000_20250809_234417/replay_buffer.hdf5', 'r') as f:
    size = f.attrs['size']
    print(f"🎉 Erfolgreich gespeichert: {size} samples")
    print(f"📊 Stats:")
    print(f"   - Mean Reward: {f.attrs.get('mean_reward', 'N/A'):.2f}")
    print(f"   - Std Reward: {f.attrs.get('std_reward', 'N/A'):.2f}")
    
    # Zeige Reward Distribution
    rewards = f['reward'][:size]
    plt.hist(rewards, bins=50)
    plt.title(f"Reward Distribution ({size} samples)")
    plt.show()
    
    # Zeige ein paar Bilder
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        idx = i * size // 8  # Gleichmäßig verteilt
        img = f['obs'][idx].transpose(1, 2, 0)
        print(f"📷 Sample {idx}: shape={img.shape}, dtype={img.dtype}")
        ax.imshow(img)
        ax.set_title(f"Sample {idx}")
        ax.axis('off')
    plt.show()