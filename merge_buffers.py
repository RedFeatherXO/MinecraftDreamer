# merge_buffers.py - Verbessert mit Verifizierung

import h5py
import numpy as np
from pathlib import Path
import argparse
import logging

# Konfiguriere ein sauberes Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

def verify_merge(source_files, merged_file_path):
    """Überprüft, ob das Zusammenführen erfolgreich und korrekt war."""
    logging.info("🕵️  Starte Verifizierung des zusammengeführten Buffers...")
    
    total_source_samples = 0
    for f_path in source_files:
        with h5py.File(f_path, 'r') as hf:
            total_source_samples += hf.attrs.get('size', 0)

    with h5py.File(merged_file_path, 'r') as merged_f:
        merged_samples = merged_f.attrs.get('size', 0)

        # 1. Überprüfung: Stimmt die Gesamtanzahl der Samples?
        if total_source_samples == merged_samples:
            logging.info(f"✅ Sample-Anzahl korrekt: {merged_samples} Samples.")
        else:
            logging.error(f"❌ Fehler bei Sample-Anzahl! Quelle: {total_source_samples}, Ziel: {merged_samples}")
            return False

        # 2. Überprüfung: Stichprobenartiger Datenvergleich
        if source_files:
            logging.info(" Stichprobenartiger Datenvergleich...")
            # Nimm die erste Quelldatei als Referenz
            with h5py.File(source_files[0], 'r') as source_f:
                source_size = source_f.attrs.get('size', 0)
                if source_size > 0:
                    # Wähle einen zufälligen Index
                    idx = np.random.randint(0, source_size)
                    
                    # Vergleiche die Daten an diesem Index
                    source_obs = source_f['obs'][idx]
                    merged_obs = merged_f['obs'][idx]
                    
                    if np.array_equal(source_obs, merged_obs):
                        logging.info("✅ Stichprobe (obs) erfolgreich überprüft.")
                    else:
                        logging.error("❌ Fehler bei der Daten-Stichprobe!")
                        return False
    
    logging.info("✅ Verifizierung erfolgreich abgeschlossen!")
    return True


def merge_buffers(source_dir, output_path):
    source_dir = Path(source_dir)
    output_path = Path(output_path)

    # KORREKTUR: Suche rekursiv in Unterordnern nach der richtigen Datei
    buffer_files = sorted(list(source_dir.glob('**/replay_buffer.hdf5')))
    
    if not buffer_files:
        logging.warning(f"Keine 'replay_buffer.hdf5' Dateien im Verzeichnis '{source_dir}' gefunden.")
        return
    
    logging.info(f"📦 {len(buffer_files)} Buffer-Dateien zum Zusammenführen gefunden.")
    
    total_size = 0
    obs_shape = None
    for f_path in buffer_files:
        with h5py.File(f_path, 'r') as hf:
            size = hf.attrs.get('size', 0)
            if size > 0 and obs_shape is None:
                obs_shape = hf.attrs.get('obs_shape', (3, 84, 84))
            logging.info(f"   - {f_path.parent.name}/{f_path.name}: {size} samples")
            total_size += size
    
    logging.info(f"\n📊 Gesamtanzahl zu schreibender Samples: {total_size}")
    
    if total_size == 0:
        logging.warning("Keine Samples zum Zusammenführen vorhanden. Breche ab.")
        return

    logging.info(f"\n🔄 Führe Daten in '{output_path}' zusammen...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as out:
        # Datensätze erstellen
        out.create_dataset('obs', (total_size, *obs_shape), dtype='f4', compression='gzip')
        out.create_dataset('action', (total_size,), dtype='i4')
        out.create_dataset('reward', (total_size,), dtype='f4')
        out.create_dataset('next_obs', (total_size, *obs_shape), dtype='f4', compression='gzip')
        out.create_dataset('done', (total_size,), dtype='bool')
        
        offset = 0
        for f_path in buffer_files:
            with h5py.File(f_path, 'r') as inp:
                size = inp.attrs.get('size', 0)
                if size > 0:
                    out['obs'][offset:offset+size] = inp['obs'][:size]
                    out['action'][offset:offset+size] = inp['action'][:size]
                    out['reward'][offset:offset+size] = inp['reward'][:size]
                    out['next_obs'][offset:offset+size] = inp['next_obs'][:size]
                    out['done'][offset:offset+size] = inp['done'][:size]
                    offset += size
        
        out.attrs['size'] = total_size
        out.attrs['obs_shape'] = obs_shape
    
    logging.info(f"\n✅ Zusammengeführter Buffer gespeichert: {output_path}")
    logging.info(f"   Dateigröße: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Führe die neue Verifizierungsfunktion aus
    verify_merge(buffer_files, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mehrere HDF5 Replay Buffer zusammenführen.")
    parser.add_argument('--source_dir', type=str, default='replay_runs',
                        help='Verzeichnis, das die einzelnen Run-Ordner enthält.')
    parser.add_argument('--output_file', type=str, default='merged_buffers/replay_buffer_merged.hdf5',
                        help='Pfad zur neuen, zusammengeführten Ausgabedatei.')
    args = parser.parse_args()
    
    merge_buffers(args.source_dir, args.output_file)