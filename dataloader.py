#reference: https://github.com/jet-universe/particle_transformer/blob/main/dataloader.py

import numpy as np
import awkward as ak
import uproot
import vector
import torch
from torch.utils.data import IterableDataset
import random
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
    max_num_particles=128,
    allowed_labels=None, 
    tau_labels=None
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

  y = ak.to_numpy(table[label_key]).astype('int64')

  if allowed_labels is not None:
    keep = np.isin(y, list(allowed_labels))
    table = table[keep]
    y = y[keep]
    y = np.array([1 if label in tau_labels else 0 for label in y], dtype=np.int64)

  p4 = vector.zip({
  'px': table['part_px'],
  'py': table['part_py'],
  'pz': table['part_pz'],
  'E': table['part_energy']
  })

  table["part_pt"] = p4.pt
  table["part_eta"] = p4.eta
  table["part_phi"] = p4.phi

  # Shape: (num_events, 2, max_num_particles, 4)
  v_particles = np.stack([
      ak.to_numpy(pad(p4.px, max_num_particles)),
      ak.to_numpy(pad(p4.py, max_num_particles)),
      ak.to_numpy(pad(p4.pz, max_num_particles)),
      ak.to_numpy(pad(p4.E, max_num_particles))
  ], axis=-1)

  x_particles = np.stack([ak.to_numpy(pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=-1)
  # x_particles = np.transpose(x_particles, (0, 2, 1))

  # Mask is 1 if pt > 0 (real particle), 0 if padding
  # Shape: (num_events, max_num_particles)
  mask = (ak.to_numpy(pad(table['part_pt'], max_num_particles)) > 0).astype(bool)

  x_jets = np.stack([ak.to_numpy(table[n]) for n in jet_features], axis=1)

  return x_particles, x_jets, v_particles, mask, y


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
  def __init__(self, filepaths, buffer_size=200000, shuffle_files=True, max_num_particles=128, allowed_labels=None, tau_labels=None):
    self.filepaths = filepaths
    self.shuffle_files = shuffle_files
    self.max_num_particles = max_num_particles
    self.allowed_labels = allowed_labels
    self.tau_labels = tau_labels
    self.buffer_size = buffer_size


  def parse_files(self, filepath):
    x_particles, x_jets, v_particles, mask, labels = read_file(
      filepath=filepath,
      particle_features=constituent_keys,
      jet_features=hlf_keys,
      labels=label_key,
      max_num_particles=self.max_num_particles,
      allowed_labels=self.allowed_labels, 
      tau_labels=self.tau_labels
      )
    
    for i in range(len(labels)):
      yield(
          torch.tensor(x_particles[i], dtype=torch.float).clone(),
          torch.tensor(x_jets[i], dtype=torch.float).clone(),
          torch.tensor(v_particles[i], dtype=torch.float).clone(),
           torch.tensor(mask[i], dtype=torch.bool),
          torch.tensor(labels[i], dtype=torch.long).clone(),
      )

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()

    #Rank sharding (DDP)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
           world_size = dist.get_world_size()
           rank = dist.get_rank()
        else:
           world_size, rank = 1, 0
    except Exception:
       world_size, rank = 1, 0 
    
    file_list = list(self.filepaths)

    if self.shuffle_files:
       random.shuffle(file_list)

    #Shard by rank
    file_list = file_list[rank::world_size]

    #Shard by worker
    if worker_info is not None:
       wid = worker_info.id
       nw = worker_info.num_workers
       file_list = file_list[wid::nw]

    buffer = []
    
    # Create a generator that yields all samples from the assigned files
    def sample_generator():
        for fp in file_list:
            try:
                yield from self.parse_files(fp)
            except Exception as e:
                wid = worker_info.id if worker_info else 0
                print(f"ERROR: Worker {wid} failed to process {fp}. Reason: {e}. Skipping.")
                continue
    
    # Fill the initial buffer
    sample_gen = sample_generator()
    for sample in sample_gen:
        buffer.append(sample)
        if len(buffer) >= self.buffer_size:
            break
    
    # Shuffle the initial buffer
    random.shuffle(buffer)

    # Main loop: yield a random sample from the buffer and replace it with a new one
    for sample in sample_gen:
        # Pop a random sample from buffer to yield
        idx_to_yield = random.randint(0, len(buffer) - 1)
        yield buffer.pop(idx_to_yield)
        # Add the new sample to the buffer
        buffer.append(sample)
    
    # After all files are read, drain the rest of the buffer
    while buffer:
        idx_to_yield = random.randint(0, len(buffer) - 1)
        yield buffer.pop(idx_to_yield)