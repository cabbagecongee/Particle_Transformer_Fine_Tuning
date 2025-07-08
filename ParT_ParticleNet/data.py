'''
Handles all JetNet data loading and processing. 
'''

from __future__ import annotations
from jetnet.datasets import JetNet
import numpy as np
from jetnet.datasets.normalisations import FeaturewiseLinear
from sklearn.preprocessing import OneHotEncoder
import math
import torch


# Initialize the OneHotEncoder with all jet types
ALL_JET_TYPES = [[0], [2]] 
ENCODER = OneHotEncoder(sparse_output=False)
ENCODER.fit(np.array(ALL_JET_TYPES))

#data preparation
def OneHotEncodeType(x: np.ndarray):
    type_column = x[..., 0].astype(int).reshape(-1, 1)
    type_encoded = ENCODER.transform(type_column)
    other_features = x[..., 1:].reshape(-1, 3)
    return np.concatenate((type_encoded, other_features), axis=-1).reshape(*x.shape[:-1], -1)

data_args = {
    "jet_type": ["g", "t"],  # only selecting gluon and top quark jets
    "data_dir": "datasets/jetnet",
    # these are the default particle features, written here to be explicit
    "particle_features": ["etarel", "phirel", "ptrel", "mask"],
    "num_particles": 30,  # we retain only the 10 highest pT particles for this demo
    "jet_features": ["type", "pt", "eta", "mass"],
    # we don't want to normalise the 'mask' feature so we set that to False
    "particle_normalisation": FeaturewiseLinear(
        normal=True, normalise_features=[True, True, True, False]
    ),
    # pass our function as a transform to be applied to the jet features
    "jet_transform": OneHotEncodeType,
    "download": True,
}

#convert particle features into (N, C, P)
def get_pf_x(jets):
    pf_list = []

    for pf, _ in jets:
        pf = pf[:, :3].clone().detach().float().t()
        pf_list.append(pf)

    return torch.stack(pf_list)


def get_pf_v(jets): #output v, shape (4, P)
    pf_v = []

    for pf, _ in jets:
        vectors = []
        etarel = pf[:, 0]
        phirel = pf[:, 1]
        ptrel = pf[:, 2]

        px = ptrel * torch.cos(phirel)
        py = ptrel * torch.sin (phirel)
        pz = ptrel * torch.sinh(etarel)
        E  = torch.sqrt(px**2 + py**2 + pz**2)

        v = torch.stack((px, py, pz, E), dim=0) # (4, P)
        pf_v.append(v)

    return torch.stack(pf_v)

def get_pf_mask(jets):
    masks = []

    for pf, _ in jets:
        mask = pf[:, -1].clone().detach().float().unsqueeze(0)
        masks.append(mask)
    return torch.stack(masks)

def get_labels(jets):
    num_classes = len(data_args["jet_type"])
    labels = []
    
    for _, jf in jets:
        class_label = jf[0:num_classes].clone().detach()
        labels.append(class_label)
    
    return torch.stack(labels)


def load_data(split):
    '''
    pf_x = particle features  -> (number of jets (N), number of particle features (C), number of particles per jet (P))
    pf_v =  the 4vector (px, py, pz, E) -> (N, 4, P)
    pf_mask = mask -> (N, 1, P) where 1 is a dummy channel dimension; values are 1 if real particle and 0 if padding
    labels = label of jet from one-hot encoding (e.g. gluon = 0, top = 1) -> (N, num_classes) 
    '''

    #load JetNet dataset objects
    dataset = JetNet(**data_args, split=split)
    dataset = [dataset[i] for i in range(1000)]
    pf_x = get_pf_x(dataset)
    pf_v = get_pf_v(dataset)
    pf_mask = get_pf_mask(dataset)
    labels = get_labels(dataset)

    data = {
        "pf_x" : pf_x,
        "pf_v" : pf_v,
        "pf_mask" : pf_mask,
        "labels" : labels
    }
    
    return data

def get_data_config(data):
    input_dicts = {"jet_features" : data_args["jet_features"]}
    label_value = data_args["jet_type"]
    input_names = list(data.keys())
    input_shapes = {name: data[name].shape for name in input_names}

    data_config = {
        "input_dicts" : input_dicts,
        "label_value" : label_value,
        "input_names" : input_names,
        "input_shapes" : input_shapes
    }

    return data_config