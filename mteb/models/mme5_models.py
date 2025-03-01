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
from transformers import MllamaForConditionalGeneration, AutoProcessor

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        model_name,
        max_length=None,
        normalize=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.base = MllamaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        self.max_length = max_length
        self.normalize = normalize
        self.processor.tokenizer.padding_side = "right"

    def forward(
        self,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        last_hidden_state = self.base(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            **kwargs
        ).hidden_states[-1]  # https://huggingface.co/intfloat/mmE5-mllama-11b-instruct#transformers

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]  # TODO
        if left_padding:
            embeddings = last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            embeddings = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def embed(
        self,
        texts: list[str],
        images: list[Image.Image],
        device,
        instruction=None,
        **kwargs,
    ):
        if instruction and not instruction.endswith("\n"):
            instruction += "\n"
        # Inputs must be batched
        input_texts, input_images = [], []
        for t, i in zip(texts, images):
            if i is None:
                input_images = None  # All examples in the same batch are consistent
                input_str = f"{instruction or ''}{t.lstrip()}"
            else:
                input_images.append(i)
                if t is None:
                    if not instruction:
                        instruction = "Represent the given image.\n"
                elif not instruction:
                    # TODO: I+T doc instruction?
                    instruction = "Represent the given image with the following question: What is in the image\n"
                input_str = f"<|image|><|begin_of_text|>{instruction}{t}"
            input_texts.append(input_str)

        inputs = self.processor(
            text=input_texts,
            images=input_images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}  # TODO
        embeddings = self.forward(**inputs)
        return embeddings


HF_MME5_MLLAMA_11B = "intfloat/mmE5-mllama-11b-instruct"


# Modified from gme_v_models.py
class MME5(Wrapper):
    def __init__(
        self,
        model_name: str = HF_MME5_MLLAMA_11B,
        model_path: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        self.model = Encoder(model_path or model_name, **kwargs)
        self.model.eval()
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
        for n, (i, img_batch) in enumerate(
            zip(range(0, n_batch * batch_size, batch_size), image_loader)
        ):
            text_batch = none_batch if texts is None else texts[i : i + batch_size]
            img_batch = none_batch if img_batch is None else img_batch
            inputs = dict(
                texts=text_batch, images=img_batch, instruction=instruction, **kwargs
            )
            with torch.inference_mode():
                embeddings = self.model.embed(**inputs, device=self.device)
            all_embeddings.append(embeddings.cpu())
            pbar.update(1)
        pbar.close()
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings


def custom_collate_fn(batch):
    return batch



###
training_data = None


mme5_mllama_11b = ModelMeta(
    loader=partial(
        MME5,
        model_name=HF_MME5_MLLAMA_11B,
    ),
    name=HF_MME5_MLLAMA_11B,
    languages=["eng_Latn", "cmn-Hans"],  # TODO
    open_weights=True,
    revision="cbb328b9bf9ff5362c852c3166931903226d46f1",
    release_date="2025-02-16",
    modalities=["image", "text"],
    n_parameters=10_600_000_000,
    memory_usage_mb=8427,
    embed_dim=4096,
    license="mit",
    max_tokens=131072,
    reference="https://huggingface.co/" + HF_MME5_MLLAMA_11B,
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=True,
    public_training_code="https://github.com/haon-chen/mmE5",
    public_training_data="https://huggingface.co/datasets/intfloat/mmE5-MMEB-hardneg",
    training_datasets=training_data,
)
