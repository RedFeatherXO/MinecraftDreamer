# merge_buffers.py
# Merged alle Agent-Buffer zu einem großen

import h5py
import numpy as np
from pathlib import Path

def merge_buffers():
    # Finde alle Agent Buffer
    buffer_files = sorted(Path('.').glob('replay_buffer_agent_*.hdf5'))
    
    if not buffer_files:
        print("❌ Keine Buffer gefunden!")
        return
    
    print(f"📦 Gefundene Buffer: {len(buffer_files)}")
    
    # Berechne Gesamtgröße
    total_size = 0
    for f in buffer_files:
        with h5py.File(f, 'r') as hf:
            size = hf.attrs.get('size', 0)
            print(f"   {f.name}: {size} samples")
            total_size += size
    
    print(f"\n📊 Total: {total_size} samples")
    
    # Merge
    output_file = 'replay_buffer_merged.hdf5'
    print(f"\n🔄 Merging to {output_file}...")
    
    with h5py.File(output_file, 'w') as out:
        # Create datasets
        obs_shape = (3, 84, 84)  # From config
        out.create_dataset('obs', (total_size, *obs_shape), dtype='f4')
        out.create_dataset('action', (total_size,), dtype='i4')
        out.create_dataset('reward', (total_size,), dtype='f4')
        out.create_dataset('next_obs', (total_size, *obs_shape), dtype='f4')
        out.create_dataset('done', (total_size,), dtype='bool')
        
        # Copy data
        offset = 0
        for f in buffer_files:
            with h5py.File(f, 'r') as inp:
                size = inp.attrs.get('size', 0)
                if size > 0:
                    out['obs'][offset:offset+size] = inp['obs'][:size]
                    out['action'][offset:offset+size] = inp['action'][:size]
                    out['reward'][offset:offset+size] = inp['reward'][:size]
                    out['next_obs'][offset:offset+size] = inp['next_obs'][:size]
                    out['done'][offset:offset+size] = inp['done'][:size]
                    offset += size
                    print(f"   ✅ Copied {size} samples from {f.name}")
        
        # Set metadata
        out.attrs['size'] = total_size
        out.attrs['obs_shape'] = obs_shape
    
    print(f"\n✅ Merged buffer saved: {output_file}")
    print(f"   Size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    merge_buffers()