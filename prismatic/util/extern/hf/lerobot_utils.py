"""
lerobot_utils.py

Utilities for working with LeRobotDataset v2.0.
"""

from dataclasses import dataclass
from typing import (
    Optional,
    Sequence,
    Type,
)

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


def create_action_norm_stats_dict_from_lerobot_dataset(
    dataset: LeRobotDataset,
) -> dict[str, dict[str, list[float]]]:
    """
    Get statistics for unnormalizing actions from a v2.0 LeRobotDataset.
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE != "bounds_q99":
        raise NotImplementedError(
            "For now, only q01/q99 normalization is supported "
            "for OpenVLA-OFT with LeRobotDataset v2.0"
        )
    assert (
        "action" in dataset.meta.stats
        and "q01" in dataset.meta.stats["action"]
        and "q99" in dataset.meta.stats["action"]
        and len(dataset.meta.stats["action"]["q01"]) == ACTION_DIM
        and len(dataset.meta.stats["action"]["q99"]) == ACTION_DIM
    ), "Dataset must have q01 and q99 stored for each action dimension"
    
    action_norm_stats = {
        "q01": dataset.meta.stats["action"]["q01"].tolist(),
        "q99": dataset.meta.stats["action"]["q99"].tolist(),
    }
    return action_norm_stats


def create_rlds_dataset_stats_dict_from_lerobot_dataset(
    dataset: LeRobotDataset,
    dataset_name: str,
) -> dict[str, dict[str, float | list[float] | dict]]:
    """
    Create a dictionary of statistics from a v2.0 LeRobotDataset that stores
    action normalization statistics.
    """
    
    try:
        action_norm_stats = \
            create_action_norm_stats_dict_from_lerobot_dataset(dataset)
    except Exception as e:
        raise ValueError(
            f"Couldn't retrieve action norm stats from dataset: {e}"
        ) from e
    
    dataset_stats = {
        dataset_name: {
            # Copy all action statistics
            "action": action_norm_stats,
            # Add trajectory/transition counts
            "num_trajectories": dataset.num_episodes,
            "num_transitions": dataset.num_frames
        }
    }

    # Add proprioceptive statistics if available
    if "proprio" in dataset.meta.stats:
        dataset_stats[dataset_name]["proprio"] \
            = dataset.meta.stats["proprio"]
        
    # Add any other available statistics
    for key, value in dataset.meta.stats.items():
        if key not in ["action", "proprio"]:
            dataset_stats[dataset_name][key] = value

    return dataset_stats


def create_train_val_split_from_lerobot_dataset(
    dataset: LeRobotDataset,
    split: float = 0.1,
) -> tuple[LeRobotDataset, LeRobotDataset]:
    """
    Create a trajectory-based train/val split from a LeRobotDataset.
    """

    episode_indices = list(range(dataset.num_episodes))
    np.random.shuffle(episode_indices)
    split = int(np.floor(split * len(episode_indices)))
    step_indices_by_episode = [
        np.arange(
            start=dataset.episode_data_index['from'][ep_idx],
            stop=dataset.episode_data_index['to'][ep_idx],
        )
        for ep_idx in episode_indices
    ]
    train_indices = [
        int(idx)
        for idx in np.concatenate(step_indices_by_episode[split:])
    ]
    val_indices = [
        int(idx)
        for idx in np.concatenate(step_indices_by_episode[:split])
    ]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset


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
    action_norm_stats: Optional[dict[str, np.ndarray]] = None
    
    def __call__(self, instances: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Process and collate a batch of instances from LeRobotDataset."""
        batch_size = len(instances)
        
        # 1. Extract data from instances
        processed_items = []
        for item in instances:
            # Extract task/instruction
            task = item.get("task", "")
            
            # === MODIFIED: Extract action_chunk ===
            # TODO(alan): Remove this once we have a way to get the action chunks from the dataset
            # Retrieve the pre-computed action chunk (which is already a tensor from the Dataset subclass)
            action_chunk_tensor = item.get("action_chunk")
            if action_chunk_tensor is None:
                 raise NotImplementedError(
                    "Item missing 'action_chunk'. This collator requires the dataset to be pre-processed "
                    "by a script (e.g., create_dataset_with_action_chunking.py) to add this column. "
                    "Standard LeRobotDataset does not provide chunks directly in this format."
                 )

            # Ensure the tensor has the correct dtype (float32)
            action_chunk_tensor = action_chunk_tensor.to(dtype=torch.float32)

            # === Original Sanity Check (can keep) ===
            if action_chunk_tensor.shape[1] != ACTION_DIM:
                 raise ValueError(f"Action chunk dimension {action_chunk_tensor.shape[1]} does not match ACTION_DIM {ACTION_DIM}")
            # ========================================

            # === MODIFIED: Normalize the whole chunk ===
            # Normalize actions if stats are provided
            if self.action_norm_stats is not None:
                q01 = self.action_norm_stats.get("q01")
                q99 = self.action_norm_stats.get("q99")
                # Raise error if normalization stats are expected but incomplete
                if q01 is None or q99 is None:
                    raise ValueError(
                        "'action_norm_stats' was provided, but missing "
                        "'q01' or 'q99' keys. Cannot normalize actions."
                    )
                
                # Proceed with normalization only if stats are valid
                q01 = torch.tensor(q01, dtype=action_chunk_tensor.dtype)
                q99 = torch.tensor(q99, dtype=action_chunk_tensor.dtype)
                # Apply normalization across the whole chunk tensor
                normalized_action_chunk \
                    = (2 * action_chunk_tensor - q01 - q99) / (q99 - q01)
            else:
                # Keep actions unnormalized if no stats were provided at all
                raise ValueError(
                    "Action normalization stats (`action_norm_stats`) "
                    "were not provided to the collator, but normalization "
                    "is expected."
                )
            # =========================================

            # === MODIFIED: Tokenize the flattened chunk ===
            # Tokenize the flattened action chunk sequence
            action_tokens = self.action_tokenizer(normalized_action_chunk.view(-1))
            # ============================================

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
            action_tokens_len = len(action_tokens) # Length based on action chunk
            labels[:-action_tokens_len-1] = IGNORE_INDEX
            if not self.predict_stop_token:
                labels[-1] = IGNORE_INDEX
            
            # 6. Add to processed items
            processed_item = {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": item.get("pixel_values") if "pixel_values" in item else item.get(next(k for k in item if "image" in k.lower())),
                "actions": normalized_action_chunk # Store the potentially normalized action chunk tensor
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