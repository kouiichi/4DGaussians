# GMFlow默认配置
DEFAULT_CONFIG = {
    'feature_channels': 128,
    'num_scales': 1,
    'upsample_factor': 8,
    'num_head': 1,
    'attention_type': 'swin',
    'ffn_dim_expansion': 4,
    'num_transformer_layers': 6,
    'model': './gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth'
}

class GMFlowConfig:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = DEFAULT_CONFIG.copy()
        self._config = config_dict
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._config.get(key)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def items(self):
        return self._config.items()
    
    def keys(self):
        return self._config.keys()
    
    def values(self):
        return self._config.values()

def get_cfg():
    """返回GMFlow配置对象"""
    return GMFlowConfig()