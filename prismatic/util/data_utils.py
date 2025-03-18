"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import (
    Callable, 
    Dict, 
    Sequence, 
    Optional, 
    Tuple, 
    Type,
)

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


def create_tensor_compatible_image_transform(transform_fn):
    """
    Wraps a transform function that expects PIL Images to work with tensors coming from LeRobotDataset.
    
    Args:
        transform_fn: A function that takes a PIL Image and transforms it
        
    Returns:
        A function that can handle tensors from LeRobotDataset
    """
    def wrapper(tensor_image):
        if isinstance(tensor_image, torch.Tensor):
            # Convert tensor to PIL Image
            # The tensor is typically [C, H, W] with values in [0, 1]
            img_array = (tensor_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            # Apply the transform
            return transform_fn(pil_image)
        else:
            # If it's already a PIL Image or numpy array, apply transform directly
            return transform_fn(tensor_image)
    
    return wrapper


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

@dataclass
class VLACollatorForLeRobotDataset:
    """
    Collator for LeRobotDataset instances specifically for VLA training.
    
    This collator handles:
    1. Action tokenization
    2. Prompt construction
    3. Input/label tokenization and masking
    4. Proper batching with padding
    """
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    prompt_builder_fn: Type[PurePromptBuilder]
    pad_token_id: int
    model_max_length: int
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    action_norm_stats: Optional[Dict[str, np.ndarray]] = None
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Process and collate a batch of instances from LeRobotDataset."""
        batch_size = len(instances)
        
        # 1. Extract data from instances
        processed_items = []
        for item in instances:
            # Extract task/instruction
            task = item.get("task", "")
            
            # Extract and normalize actions
            action = torch.cat([
                item.get("action.pose", torch.zeros(6)),
                item.get("action.gripper", torch.zeros(1)).unsqueeze(0)
            ])
            
            # Normalize actions if stats are provided
            if self.action_norm_stats is not None:
                q01, q99 = self.action_norm_stats.get("q01"), self.action_norm_stats.get("q99")
                if q01 is not None and q99 is not None:
                    action = (2*action - torch.tensor(q01) - torch.tensor(q99)) / (torch.tensor(q99) - torch.tensor(q01))
            
            # Tokenize action
            action_tokens = self.action_tokenizer(action)
            
            # 2. Build prompt
            prompt_builder = self.prompt_builder_fn("openvla")
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {task}?"},
                {"from": "gpt", "value": action_tokens},
            ]
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            
            # 3. Tokenize
            tokenized = self.base_tokenizer(
                prompt_builder.get_prompt(), 
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_ids = tokenized.input_ids.squeeze(0)
            
            # 4. Create labels (copy input_ids)
            labels = input_ids.clone()
            
            # 5. Mask labels (only keep action tokens for loss)
            action_tokens_len = len(action_tokens)
            labels[:-action_tokens_len-1] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX
            
            # 6. Add to processed items
            processed_item = {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": item.get("pixel_values") if "pixel_values" in item else item.get(next(k for k in item if "image" in k.lower())),
                "actions": action
            }
            
            # Add dataset name if available
            if "dataset_name" in item:
                processed_item["dataset_name"] = item["dataset_name"]
            
            # Add wrist camera if used
            if self.use_wrist_image and any("wrist" in k.lower() for k in item):
                wrist_keys = [k for k in item if "wrist" in k.lower()]
                if wrist_keys:
                    processed_item["pixel_values_wrist"] = torch.stack([item[k] for k in wrist_keys])
            
            # Add proprioceptive data if used
            if self.use_proprio and "proprio" in item:
                processed_item["proprio"] = item["proprio"]
            
            processed_items.append(processed_item)
        
        # 7. Collate inputs with padding
        input_ids = [item["input_ids"] for item in processed_items]
        labels = [item["labels"] for item in processed_items]
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # Truncate if necessary
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        
        # Create attention mask based on padding
        attention_mask = input_ids.ne(self.pad_token_id)
        
        # 8. Collate images and actions
        # Stack main images
        pixel_values = torch.stack([item["pixel_values"] for item in processed_items])
        
        # If wrist images are available, combine them with the main images
        if self.use_wrist_image and all("pixel_values_wrist" in item for item in processed_items):
            pixel_values_wrist = torch.stack([item["pixel_values_wrist"] for item in processed_items])
            
            # Reshape wrist images if needed (from [B, num_wrist, C, H, W] to [B, num_wrist*C, H, W])
            if pixel_values_wrist.dim() == 5:  # [B, num_wrist, C, H, W]
                B, num_wrist, C, H, W = pixel_values_wrist.shape
                pixel_values_wrist = pixel_values_wrist.view(B, num_wrist * C, H, W)
            
            # Concatenate main and wrist images along the channel dimension
            pixel_values = torch.cat([pixel_values, pixel_values_wrist], dim=1)
        
        actions = torch.stack([item["actions"] for item in processed_items])
        
        # 9. Build final batch
        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
        }
        
        # Add wrist images if available
        if self.use_wrist_image and all("pixel_values_wrist" in item for item in processed_items):
            batch["pixel_values_wrist"] = torch.cat([item["pixel_values_wrist"] for item in processed_items], dim=1)
        
        # Add proprioceptive data if available
        if self.use_proprio and all("proprio" in item for item in processed_items):
            batch["proprio"] = torch.stack([item["proprio"] for item in processed_items])
        
        # Add dataset names if available
        if all("dataset_name" in item for item in processed_items):
            batch["dataset_names"] = [item["dataset_name"] for item in processed_items]
        
        return batch