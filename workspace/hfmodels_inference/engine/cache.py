from typing import Optional, Dict, Union, List, Tuple, Any
import torch

from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_torchdynamo_compiling


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        # if batch_size is not None:
        #     logger.warning_once(
        #         f"The 'batch_size' argument of {self.__class__.__name__} is deprecated and will be removed in "
        #         "v4.49. Use the more precisely named 'max_batch_size' argument instead."
        #     )

        self.max_batch_size = batch_size or max_batch_size
        self.max_cache_len = (
            config.max_position_embeddings if max_cache_len is None else max_cache_len
        )

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        if getattr(config, "qk_nope_head_dim", None):
            self.khead_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            self.vhead_dim = config.v_head_dim

        else:
            self.khead_dim = self.vhead_dim = (
                config.head_dim
                if hasattr(config, "head_dim")
                else config.hidden_size // config.num_attention_heads
            )

        self.dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.layer_start_idx = config.layer_start_idx
        self.pos = [None] * config.num_hidden_layers
        self.key_cache: List[torch.Tensor] = [None] * config.num_hidden_layers
        self.value_cache: List[torch.Tensor] = [None] * config.num_hidden_layers
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        kcache_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.khead_dim,
        )
        vcache_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.vhead_dim,
        )
        for idx in range(config.layer_start_idx, config.layer_end_idx):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(
                kcache_shape, dtype=self.dtype, device=layer_device
            )
            new_layer_value_cache = torch.zeros(
                vcache_shape, dtype=self.dtype, device=layer_device
            )
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            # if not is_torchdynamo_compiling():
            #     self.register_buffer(
            #         f"key_cache_{idx}",
            #         torch.zeros(kcache_shape, dtype=dtype, device=layer_device),
            #     )
            #     self.register_buffer(
            #         f"value_cache_{idx}",
            #         torch.zeros(vcache_shape, dtype=dtype, device=layer_device),
            #     )
            #     new_layer_key_cache = getattr(self, f"key_cache_{idx}")
            #     new_layer_value_cache = getattr(self, f"value_cache_{idx}")
            #     torch._dynamo.mark_static_address(new_layer_key_cache)
            #     torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache[idx] = new_layer_key_cache
            self.value_cache[idx] = new_layer_value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        cache_position = cache_kwargs.get("cache_position")

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            # k_out.copy_(key_states)
            # v_out.copy_(value_states)
            if self.pos[layer_idx] == None:
                k_out[:, :, : key_states.shape[-2], :].copy_(key_states)
                v_out[:, :, : key_states.shape[-2], :].copy_(value_states)
                self.pos[layer_idx] = torch.tensor(
                    key_states.shape[-2], device=key_states.device
                )
            else:
                k_out.index_copy_(2, self.pos[layer_idx], key_states)
                v_out.index_copy_(2, self.pos[layer_idx], value_states)
                self.pos[layer_idx] += 1

            return (
                k_out[:, :, : self.pos[layer_idx], :],
                v_out[:, :, : self.pos[layer_idx], :],
            )
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

            return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if layer_idx:
            return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum().item()
        else:
            return (self.key_cache[self.layer_start_idx][0, 0].any(dim=-1)).sum().item()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        # logger.warning_once(
        #     f"The 'batch_size' attribute of {self.__class__.__name__} is deprecated and will be removed in "
        #     "v4.49. Use the more precisely named 'self.max_batch_size' attribute instead."
        # )
        return self.max_batch_size
