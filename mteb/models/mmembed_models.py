from __future__ import annotations

import logging
import math
import os
from functools import partial
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AutoConfig, AutoModel
from transformers.utils.versions import require_version

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# https://huggingface.co/nvidia/MM-Embed/blob/main/modeling_nvmmembed.py
from typing import List, Optional, Tuple, Union

from transformers import LlavaNextProcessor
from transformers import LlavaNextForConditionalGeneration, LlavaNextConfig
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, image_size_to_num_patches


class NVMMEmbedModel(LlavaNextForConditionalGeneration):
    def __init__(self, config: LlavaNextConfig):
        super().__init__(config)

        nvemb_config = AutoConfig.from_pretrained(config.retriever, trust_remote_code=True)
        # To load local model weights.
        nvemb_config.text_config.name_or_path = config.retriever

        nvemb_model = AutoModel.from_config(nvemb_config, trust_remote_code=True)
        self.language_model = nvemb_model.embedding_model
        self.latent_attention_model = nvemb_model.latent_attention_model

        self.preprocess_fn = LlavaNextProcessor.from_pretrained(config._name_or_path)
        self.preprocess_fn.tokenizer.padding_side = config.padding_side
        self.preprocess_fn.tokenizer.add_eos_token = config.add_eos_token
        self.global_image_patch_only = config.global_image_patch_only


    def create_pool_mask(self, attention_mask, instruction_lengths):
        pool_mask = attention_mask.clone()
        if instruction_lengths.unique().shape[0] == 1:
            length = instruction_lengths[0].item()
            pool_mask[:, :length] = 0
        else:
            for i, length in enumerate(instruction_lengths): 
                pool_mask[i, :length] = 0
        return pool_mask

    def calculate_instruction_length(self, tokenizer, prompts, prefix):
        instructions = []
        instruction_lengths = []
        for prompt in prompts:
            if prefix in prompt:
                instruction = prompt.split(prefix)[0]
                input_ids = tokenizer(instruction, return_tensors=None)['input_ids']
                instruction_length = len(input_ids)
                if '<image>' in instruction:
                    instruction_length += (576 - 1)
                instruction_lengths.append(instruction_length)
            else:
                instruction_lengths.append(0)
        return instruction_lengths

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        instruction_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        clip_global_image_feature = None

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            for_inputs_embeds_ids[(input_ids == 32001)] = 2 #We use tokenizer from Llava-Next but later replace PAD with EOS Token
            inputs_embeds = self.language_model.get_input_embeddings()(for_inputs_embeds_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                # ! infer image_num_patches from image_sizes
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.config.image_grid_pinpoints,
                        patch_size=self.config.vision_config.image_size,
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    if pixel_values.shape[1] == 1:
                        image_num_patches = [1 for imsize in image_sizes]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                clip_global_image_feature = image_features.pooler_output
                selected_image_feature = image_features.hidden_states[vision_feature_layer]
                
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                
                image_features = self.multi_modal_projector(selected_image_feature)
                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )

            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)

                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pool_mask = self.create_pool_mask(attention_mask, instruction_lengths)
        
        embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
        )


        return LlavaNextCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=embeds,
            attentions=outputs.attentions,
            image_hidden_states=clip_global_image_feature,
        )

    @torch.no_grad()
    def encode(self, inputs, is_query = False, instruction = None, max_length = 512, query_prefix = 'Query: '):
        assert type(inputs) == list, 'inputs should be a list of dictionay'
        prompts, imgs = [], []
        if is_query:
            if instruction is not None:
                prompt_template = f"Instruct: {instruction}\n{query_prefix}<image>\n<text>"
            else:
                prompt_template = f"{query_prefix}<image>\n<text>"
        else:
            prompt_template = f"<image>\n<text>"
    
        for input_ in inputs:
            if 'img' in input_:
                imgs.append(input_['img'])
                prompt = prompt_template
            else:
                prompt = prompt_template.replace('<image>\n', '')

            if ('txt' in input_) and (input_['txt'] is not None):
                prompt = prompt.replace('<text>', input_['txt'])
            else:
                prompt = prompt.replace('<text>', '')
            
            prompts.append(prompt)
        
        if len(imgs) == 0:
            imgs = None
        collated_features = self.preprocess_fn(prompts, imgs, return_tensors="pt", padding="longest", max_length=max_length, truncation=True).to(self.device)
        if self.global_image_patch_only and (imgs is not None): # we only use global image patch as default
            collated_features['pixel_values'] = collated_features['pixel_values'][:, 0:1]

        instruction_lengths = self.calculate_instruction_length(self.preprocess_fn.tokenizer, prompts, f'\n{query_prefix}')
        collated_features['instruction_lengths'] = torch.tensor(instruction_lengths).to(self.device)

        return self(**collated_features)


HF_MM_EMBED = "nvidia/MM-Embed"

# Modified from gme_v_models.py
class MMEmbed(Wrapper):
    """
    Load local weights:
    mteb.get_model('nvidia/MM-Embed', model_path='PATH_TO_MM-Embed', nvembed_path='PATH_TO_NV-Embed-v1')
    """

    def __init__(
        self,
        model_name: str = HF_MM_EMBED,
        model_path: str | None = None,
        nvembed_path: str | None = None,
        max_length: int = 4096,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        try:
            import einops as _
        except ImportError:
            raise ImportError("Please install `einops` for `MM-Embed` model.")
        # https://huggingface.co/nvidia/NV-Embed-v1/discussions/50
        require_version("transformers==4.43.4")

        # To load local model weights.
        self.model = NVMMEmbedModel.from_pretrained(
            model_path or model_name, retriever=nvembed_path, **kwargs
        )
        self.model.eval()
        self.max_length = max_length
        self.device = device

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ):
        return self.get_fused_embeddings(
            texts=sentences, task_name=task_name, prompt_type=prompt_type, **kwargs
        )

    def get_image_embeddings(self, images: list[Image.Image] | DataLoader, **kwargs):
        return self.get_fused_embeddings(images=images, **kwargs)

    def get_text_embeddings(self, texts: list[str], **kwargs):
        return self.get_fused_embeddings(texts=texts, **kwargs)

    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        tqdm_mininterval: int = 15,
        instruction=None,
        **kwargs: Any,
    ):
        if prompt_type == PromptType.passage:
            instruction = None
        elif instruction is None:
            instruction = self.get_instruction(task_name, prompt_type)
        self.model = self.model.to(self.device)

        if isinstance(images, DataLoader):
            image_loader = images
            batch_size = image_loader.batch_size
            image_loader.dataset.transform = None
        else:
            batch_size = kwargs.pop("batch_size", 32)
            if images is None:
                image_loader = None
            else:
                image_loader = DataLoader(
                    images,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=min(math.floor(os.cpu_count() / 2), 8),
                )

        if texts is None:
            assert image_loader is not None
            n_batch = len(image_loader)
        else:
            n_batch = len(texts) // batch_size + int(len(texts) % batch_size > 0)
            image_loader = image_loader or [None] * n_batch

        all_embeddings = []
        none_batch = [None] * batch_size
        show_progress_bar = kwargs.pop("show_progress_bar", True)
        pbar = tqdm(
            total=n_batch,
            disable=not show_progress_bar,
            mininterval=tqdm_mininterval,
            miniters=n_batch // 10,
            desc="encode",
        )
        input_kwargs = dict(
            instruction=instruction, max_length=self.max_length, is_query=prompt_type == PromptType.query
        )
        for n, (i, img_batch) in enumerate(
            zip(range(0, n_batch * batch_size, batch_size), image_loader)
        ):
            text_batch = none_batch if texts is None else texts[i : i + batch_size]
            img_batch = none_batch if img_batch is None else img_batch
            batch_data = list()
            for txt, img in zip(text_batch, img_batch):
                instance = dict()
                if txt:
                    instance.update(txt=txt)
                if img:
                    instance.update(img=img)
                batch_data.append(instance)
            with torch.inference_mode():
                outputs = self.model.encode(batch_data, **input_kwargs)
                embeddings = outputs['hidden_states']
            all_embeddings.append(embeddings.cpu())
            pbar.update(1)
        pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


def custom_collate_fn(batch):
    return batch



###
training_data = None


mm_embed = ModelMeta(
    loader=partial(
        MMEmbed,
        model_name=HF_MM_EMBED,
    ),
    name=HF_MM_EMBED,
    languages=["eng_Latn"],
    open_weights=True,
    revision="a2542c237b5b700ed01b829689756425441cef97",
    release_date="2024-11-05",
    modalities=["image", "text"],
    n_parameters=8_180_000_000,
    memory_usage_mb=8427,
    embed_dim=4096,
    license="cc-by-nc-4.0",
    max_tokens=32768,
    reference="https://huggingface.co/" + HF_MM_EMBED,
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
