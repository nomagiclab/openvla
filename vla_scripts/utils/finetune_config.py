"""
Utility functions for loading configuration values from defaults.cfg
"""

import os
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path):
    """Load the configuration from the specified config file path"""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def get_config_as_dict(config_path):
    """Get the configuration as a flat dictionary from the specified config path"""
    config = load_config(config_path)
    result = {}
    
    # Convert values to appropriate types
    for section in config.sections():
        for key, value in config[section].items():
            # Convert empty strings to None
            if value == '':
                result[key] = None
            # Convert boolean values
            elif value.lower() in ('true', 'yes', 'on', '1'):
                result[key] = True
            elif value.lower() in ('false', 'no', 'off', '0'):
                result[key] = False
            # Convert numeric values
            elif value.replace('.', '', 1).isdigit():
                if '.' in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            # Keep strings as strings
            else:
                result[key] = value
    
    return result

# Get default values from vla_scripts/cfg/finetune_defaults.cfg
def get_finetune_defaults():
    """Get the default configuration values for fine-tuning"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg/finetune_defaults.cfg")
    return get_config_as_dict(config_path)

@dataclass
class FinetuneConfig:
    """Configuration class for fine-tuning OpenVLA models"""
    # fmt: off
    # [MODEL]
    vla_path: str = None
    run_root_dir: Path = None
    adapter_tmp_dir: Path = None
    
    # [DATASET]
    data_root_dir: Path = None
    dataset_name: str = None
    tolerance_s: float = None
    revision: str = None
    download_videos: bool = None
    local_files_only: bool = None
    
    # [TRAINING]
    batch_size: int = None
    grad_accumulation_steps: int = None
    learning_rate: float = None
    max_steps: int = None
    save_steps: int = None
    save_latest_checkpoint_only: bool = None
    image_aug: bool = None
    shuffle_buffer_size: int = None
    
    # [LORA]
    lora_rank: int = None
    lora_dropout: float = None
    use_lora: bool = None
    use_quantization: bool = None
    
    # [DISTRIBUTED]
    nproc_per_node: int = None
    
    # [WANDB]
    wandb_entity: str = None
    wandb_project: str = None
    wandb_experiment_name: str = None
    run_id_note: Optional[str] = None
    # fmt: on
    
    def __post_init__(self):
        """Initialize default values from config file if not provided"""
        defaults = get_finetune_defaults()
        
        # Set default values for all fields that are None
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                config_key = field_name.upper()
                if config_key in defaults:
                    # Handle Path objects
                    if field_name in ['run_root_dir', 'adapter_tmp_dir', 'data_root_dir']:
                        setattr(self, field_name, Path(defaults[config_key]))
                    else:
                        setattr(self, field_name, defaults[config_key]) 