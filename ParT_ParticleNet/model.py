import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''



class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = ParticleTransformer(**kwargs)

    #instruct TorchScript not to trace this method -> not to apply weight decay to the class token
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}
        
    def forward(self, points, features, lorentz_vectors, mask):
        return self.model(features, v=lorentz_vectors, mask=mask)
    # features = particle-level input features, lorentz vectors = 4-momentum info, mask=binary mask for padding
        
#returns an initilized model and metadata for ONNX export
def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=3, #how many features per particle
        num_classes=len(data_config["label_value"]), #e.g. 2 for binary classification

        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128], #input -> hidden -> output dim
        pair_embed_dims=[64, 64, 64],
        num_heads=8, #parT dim
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}, #dropout settings
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False
    )
    cfg.update(**kwargs) #allows overrides from command line or config file
    _logger.info('Model config: %s' % str(cfg)) #prints out the model config
    
    model = ParticleTransformerWrapper(**cfg)

    #used for ONNX model export or inference server support
    model_info = {
        'input_names': list(data_config["input_names"]),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config["input_shapes"].items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config["input_names"]}, **{'softmax': {0: 'N'}}}
    }

    return model, model_info

def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()