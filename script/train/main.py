'''
Copyright Dongguk U.V. CSE 2020112030 KIM SEON PYO [2025/03/04 ~]

해당 코드는 DiffusersPipeline의 AudioLDM 모델을 파인튜닝하는 LoRA 가중치를 학습하는 코드입니다.
README.md 를 참고해 주세요

This code is training LoRA weight for AudioLDM in DiffusersPipeline.
you can choose Base_model for training LoRA weight and save at [AudioLDM-with-LoRA/data/LoRA_weight/~]

adapt at app.py to use
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append("AudioLDM-with-LoRA")

import logging
import numpy as np
from contextlib import nullcontext

import torch, random
import torch.nn.functional as F
import torch.nn as nn

import torchaudio
from torchaudio import transforms as AT
from torchvision import transforms as IT


from script.data.datasets import HfAudioDataset
from datasets import load_dataset
from script.train.finetune import train

# LoRA
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
    
from tqdm.auto import tqdm

from diffusers import AudioLDMPipeline
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available, convert_state_dict_to_diffusers

from diffusers.optimization import get_scheduler
from transformers import ClapTextModelWithProjection, RobertaTokenizerFast, SpeechT5HifiGan
from transformers import AutoProcessor, ClapModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import librosa

import torch, gc

gc.collect()
torch.cuda.empty_cache()

logger = get_logger(__name__)

# 시각화 툴

base_model_id = "cvssp/audioldm-s-full-v2"
dataset_hub_id = "Rofla/AudioLDM-with-LoRA-Hiphop-subgenre"
validation_prompt = "hip hop music, The subgenre of hip-hop is boom bap."
validation_epochs = 100
SCALE_FACTOR = 100

if is_wandb_available():
    import wandb

num_validation_images = 5

def main() :
    accelerator_project_config = ProjectConfiguration(project_dir="../../", logging_dir="AudioLDM-with-LoRA/log")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=None,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name="AudioLDM-with-LoRA",
        config=accelerator_project_config,
        init_kwargs={
            "wandb": {
                "entity": "kimsp0317-dongguk-university",
                "group": "gpu-exp-group-1",
                "tags": ["lora", "audioldm", "subgenre"],
                "name": "<task : r = 2, alpha = 2>"
            }
        }
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(accelerator.device)

    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
    pipe = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet)

    noise_scheduler = DDIMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    tokenizer = RobertaTokenizerFast.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = ClapTextModelWithProjection.from_pretrained(base_model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    vocoder = SpeechT5HifiGan.from_pretrained(base_model_id, subfolder="vocoder")

    # 기존 모델의 가중치는 잠금
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=2,
        lora_alpha=2,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v"],
    )

    unet = get_peft_model(unet, unet_lora_config)
    # unet.add_adapter(unet_lora_config, "default")
    # unet.set_adapter("default")

    weight_dtype = torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=1.0e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-08,
    )

    num_workers = 4
    train_batch_size = 2
    total_batch_size = train_batch_size * accelerator.num_processes
    num_train_epochs = 1000
    gradient_accumulation_steps = 1
    max_train_steps = 97000
    checkpointing_steps = 9700 * 2
    total_train_loss = 0.0
    total_steps = 0

    def collate_fn(examples):
        log_mel_spec = torch.stack([example["log_mel_spec"].unsqueeze(0) for example in examples])
        input_ids = torch.stack([example["text"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])

        return {"log_mel_spec": log_mel_spec, "input_ids": input_ids, "attention_mask" : attention_mask}

    dataset = load_dataset(dataset_hub_id, split="train")
    train_dataset = HfAudioDataset(dataset)

    ref_audios = [librosa.util.fix_length(example["audio"]["array"], size=16000*10) for example in dataset.select(range(num_validation_images))]


    # 필터링된 데이터셋으로 학습!
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )

    lr_scheduler = get_scheduler(
        "polynomial",
        optimizer=optimizer,
        num_warmup_steps = 0, #500 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    ### Train
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_root, "data", "LoRA_weight", "r2_alpha2")

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    train(accelerator, optimizer, max_train_steps,
          num_train_epochs, train_dataloader, validation_prompt,
          base_model_id, clap_model, processor, ref_audios,
          validation_epochs, weight_dtype, lr_scheduler, pipe,
          noise_scheduler, gradient_accumulation_steps,
          lora_layers, checkpointing_steps, output_dir,
          vae, text_encoder)

if __name__ == "__main__":
    main()