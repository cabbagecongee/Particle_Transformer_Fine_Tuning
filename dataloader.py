#reference: https://github.com/jet-universe/particle_transformer/blob/main/dataloader.py

import numpy as np
import awkward as ak
import uproot
import vector
import torch
from torch.utils.data import IterableDataset
import random
import time # Import the time module

vector.register_awkward()

constituent_keys = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi",
    "part_d0val", "part_d0err", "part_dzval", "part_dzerr",
    "part_charge",
    "part_isElectron", "part_isMuon", "part_isPhoton",
    "part_isChargedHadron", "part_isNeutralHadron",
    "part_pt", "part_eta", "part_phi"
]

hlf_keys = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_sdmass"]
label_key = "jet_label"

def read_file(
    filepath,
    particle_features,
    jet_features,
    labels,
    max_num_particles=128
):
  def pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim>=2 and a.shape[1] == maxlen:
      return a
    elif isinstance(a, ak.Array):
      if a.ndim ==1:
        a = ak.unflatten(a, 1)
      a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
      return ak.values_astype(a, dtype)
    else:
      x = (np.ones((len(a), maxlen)) * value).astype(dtype)
      for idx, s in enumerate(a):
        if not len(s):
          continue
        trunc = np.asarray(s[:maxlen], dtype=dtype)
        x[idx, :len(trunc)] = trunc
      return x

  table = ak.Array(ak.from_parquet(filepath))

  p4 = vector.zip({
      'px': table['part_px'],
      'py': table['part_py'],
      'pz': table['part_pz'],
      'E': table['part_energy']
  })

  table["part_pt"] = p4.pt
  table["part_eta"] = p4.eta
  table["part_phi"] = p4.phi

  x_particles = np.stack([ak.to_numpy(pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
  x_particles = np.transpose(x_particles, (0, 2, 1))

  x_jets = np.stack([ak.to_numpy(table[n]) for n in jet_features], axis=1)

  y = ak.to_numpy(table[label_key]).astype('int64')

  return x_particles, x_jets, y


class JetDataset(torch.utils.data.Dataset):
  def __init__(self, parquet_file, max_num_particles=128):
    self.x_particles, self.x_jets, self.labels = read_file(
        filepath=parquet_file,
        particle_features=constituent_keys,
        jet_features=hlf_keys,
        labels=label_key
        )
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return (
        torch.tensor(self.x_particles[idx], dtype=torch.float),
        torch.tensor(self.x_jets[idx], dtype=torch.float),
        torch.tensor(self.labels[idx], dtype=torch.long)
    )


class IterableJetDataset(IterableDataset):
  def __init__(self, filepaths, shuffle_files=True, max_num_particles=128):
    self.filepaths = filepaths
    self.shuffle_files = shuffle_files
    self.max_num_particles = max_num_particles

  def parse_files(self, filepath):
    start_time = time.time() 
    try:
      x_particles, x_jets, labels = read_file(
          filepath=filepath,
          particle_features=constituent_keys,
          jet_features=hlf_keys,
          labels=label_key,
          max_num_particles=self.max_num_particles,
      )
      end_time = time.time()
      print(f"[INFO] Finished reading {filepath} in {end_time - start_time:.2f} seconds. Found {len(labels)} jets.") # Print timing and number of jets
    except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
            return

    if len(labels) == 0:
      print(f"[WARN] No jets in file: {filepath}")
      return
    for i in range(len(labels)):
      yield(
          torch.tensor(x_particles[i], dtype=torch.float),
          torch.tensor(x_jets[i], dtype=torch.float),
          torch.tensor(labels[i], dtype=torch.long)
      )
  def __iter__(self):
    if self.shuffle_files:
      random.shuffle(self.filepaths)
    for filepath in self.filepaths:
      yield from self.parse_files(filepath)