import torch
import numpy as np
import awkward as ak
import pandas as pd
import vector
import random
from pathlib import Path
import os
import shutil
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice

# --- DUMMY DATA SETUP ---
def create_dummy_data(data_dir="dummy_data", num_files=3, samples_per_file=1000):
    """Creates a few dummy parquet files for testing."""
    data_path = Path(data_dir)
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True)

    print(f"Creating {num_files} dummy files in '{data_path}'...")
    for i in range(num_files):
        file_id = i + 1
        df = pd.DataFrame({
            # Simple float features
            'jet_pt': np.random.rand(samples_per_file).astype('float32'),
            'jet_eta': np.random.rand(samples_per_file).astype('float32'),
            'jet_phi': np.random.rand(samples_per_file).astype('float32'),
            'jet_energy': np.random.rand(samples_per_file).astype('float32'),
            'jet_sdmass': np.random.rand(samples_per_file).astype('float32'),
            # A label
            'jet_label': np.random.randint(0, 5, size=samples_per_file),
            # The important ID for tracking which file a sample came from
            'file_id': np.full(samples_per_file, file_id, dtype='int32'),
            # Mock particle data (lists of floats)
            'part_px': [np.random.rand(np.random.randint(10, 50)).astype('float32') for _ in range(samples_per_file)],
            'part_py': [np.random.rand(np.random.randint(10, 50)).astype('float32') for _ in range(samples_per_file)],
            'part_pz': [np.random.rand(np.random.randint(10, 50)).astype('float32') for _ in range(samples_per_file)],
            'part_energy': [np.random.rand(np.random.randint(10, 50)).astype('float32') for _ in range(samples_per_file)],
        })
        df.to_parquet(data_path / f"dummy_file_{file_id}.parquet")
    print("Dummy data created successfully.")
    return data_path

# --- DATALOADER CODE (with modification for testing) ---

vector.register_awkward()

# Mocking your feature lists for the test
constituent_keys = ["part_px", "part_py", "part_pz", "part_energy"]
hlf_keys = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_sdmass"]
label_key = "jet_label"

def read_file(filepath, max_num_particles=50):
    """
    Modified read_file to handle dummy data and return the file_id.
    """
    def pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, ak.Array):
            if a.ndim == 1: a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        return a # simplified for dummy data

    table = ak.from_parquet(filepath)
    labels = ak.to_numpy(table[label_key])
    file_ids = ak.to_numpy(table['file_id']) # <-- Get the file ID

    # Simplified processing for dummy data
    x_particles = np.stack([ak.to_numpy(pad(table[n], maxlen=max_num_particles)) for n in constituent_keys], axis=-1)
    x_jets = np.stack([ak.to_numpy(table[n]) for n in hlf_keys], axis=1)
    
    # We don't need all the return values for this test, just enough to run
    return x_particles, x_jets, labels, file_ids

class InterleavedJetDataset(IterableDataset):
    def __init__(self, filepaths, shuffle_files=True, max_num_particles=50, samples_per_file=10, batch_size=32):
        super(InterleavedJetDataset).__init__()
        self.filepaths = filepaths
        self.shuffle_files = shuffle_files
        self.max_num_particles = max_num_particles
        self.samples_per_file = samples_per_file
        self.batch_size = batch_size

    def parse_file(self, filepath):
        """Generator that yields single samples from a single file, including file_id."""
        try:
            x_particles, x_jets, labels, file_ids = read_file(
                filepath, max_num_particles=self.max_num_particles
            )
            for i in range(len(labels)):
                yield (
                    torch.tensor(x_particles[i], dtype=torch.float),
                    torch.tensor(x_jets[i], dtype=torch.float),
                    torch.tensor(labels[i], dtype=torch.long),
                    torch.tensor(file_ids[i], dtype=torch.long), # <-- Yield the file_id
                )
        except Exception as e:
            print(f"ERROR: Failed to process {filepath}. Reason: {e}. Skipping.")
            return

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        file_list = list(self.filepaths)
        if self.shuffle_files:
           random.shuffle(file_list)

        files_for_this_worker = file_list
        if worker_info is not None:
           files_for_this_worker = file_list[worker_info.id::worker_info.num_workers]

        if not files_for_this_worker: return

        file_generators = [self.parse_file(fp) for fp in files_for_this_worker]
        cycled_generators = cycle(file_generators)
        
        buffer_size = self.batch_size * 10
        buffer = []
        
        while len(file_generators) > 0:
            try:
                samples_from_file = list(islice(next(cycled_generators), self.samples_per_file))
                
                if not samples_from_file:
                    current_generator_to_remove = next(cycle(g for g in file_generators if g not in cycled_generators))
                    file_generators = [gen for gen in file_generators if gen != current_generator_to_remove]
                    if not file_generators: break
                    cycled_generators = cycle(file_generators)
                    continue

                buffer.extend(samples_from_file)

                if len(buffer) >= buffer_size:
                    random.shuffle(buffer)
                    for _ in range(len(buffer)):
                        yield buffer.pop(0)

            except StopIteration:
                break
                
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop(0)


# --- MAIN TEST SCRIPT ---
if __name__ == "__main__":
    DUMMY_DIR = "dummy_data"
    BATCH_SIZE = 16

    # 1. Create the dummy data
    dummy_data_path = create_dummy_data(DUMMY_DIR, num_files=3, samples_per_file=100)
    filepaths = list(dummy_data_path.glob("*.parquet"))

    # 2. Instantiate the Dataset and DataLoader
    dataset = InterleavedJetDataset(
        filepaths=filepaths,
        batch_size=BATCH_SIZE,
        samples_per_file=10 # Pull 10 samples from each file per turn
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 3. Iterate and inspect a few batches
    print("\n--- Inspecting Batches ---")
    num_batches_to_test = 5
    for i, (particles, jets, labels, file_ids) in enumerate(loader):
        if i >= num_batches_to_test:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Particle tensor shape: {particles.shape}")
        print(f"  Jets tensor shape:     {jets.shape}")
        print(f"  Labels tensor shape:   {labels.shape}")
        print(f"  File IDs in batch:     {file_ids.numpy()}")
        
        # Check for mixing
        unique_ids = torch.unique(file_ids)
        if len(unique_ids) > 1:
            print(f"  SUCCESS: Found samples from {len(unique_ids)} different files.")
        else:
            print(f"  WARNING: Only found samples from one file (ID: {unique_ids.item()}).")

    # 4. Clean up the dummy data
    print("\nCleaning up dummy data...")
    shutil.rmtree(DUMMY_DIR)
    print("Done.")