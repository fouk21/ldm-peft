from tqdm.auto import tqdm
import torch
from train_utils import unwrap_model
from diffusers import AudioLDMPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict
from train_utils import log_validation
from train_one_epoch import train_one_epoch

def train(accelerator, optimizer, max_train_steps, num_train_epochs, train_dataloader, validation_prompt, base_model_id, clap_model, processor, ref_audios, validation_epochs, weight_dtype, lr_scheduler, pipe, noise_scheduler, gradient_accumulation_steps, lora_layers, checkpointing_steps, output_dir, vae, text_encoder):
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    avg_clap_score = 0.0
    avg_original_clap_score = 0.0
    kad_score_lora = 0.0
    kad_score_original = 0.0
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        optimizer.zero_grad()

        train_loss = 0.0
        num_steps_per_epoch = 0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process,
        )

        global_step, train_loss, num_steps_per_epoch = train_one_epoch(progress_bar, accelerator,
                        unet, vae, text_encoder,
                        noise_scheduler, weight_dtype,
                        gradient_accumulation_steps,
                        lora_layers, optimizer, lr_scheduler,
                        checkpointing_steps, output_dir,
                        avg_clap_score, avg_original_clap_score,
                        kad_score_lora, kad_score_original,
                        max_train_steps, num_steps_per_epoch,
                        train_loss
                        )
        
        if accelerator.is_main_process and validation_prompt is not None and epoch % validation_epochs == 0:
            unwrapped_unet = unwrap_model(unet)
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unwrapped_unet, torch_dtype=weight_dtype)
            images, avg_clap_score_A, avg_original_clap_score_A, kad_score_lora_A, kad_score_original_A = log_validation(pipeline, accelerator, epoch,original_pipeline=pipe,
                clap_model=clap_model,
                clap_processor=processor, ref_audios=ref_audios
            )
            avg_clap_score = float(avg_clap_score_A)
            avg_original_clap_score = float(avg_original_clap_score_A)
            kad_score_lora = float(kad_score_lora_A)
            kad_score_original = float(kad_score_original_A)
            del pipeline

            torch.cuda.empty_cache()

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))

        if validation_prompt is not None:
            pipeline = AudioLDMPipeline.from_pretrained(base_model_id, unet=unet_lora_state_dict, torch_dtype=weight_dtype)
            images, avg_clap_score_A, avg_original_clap_score_A, kad_score_lora_A, kad_score_original_A = log_validation(pipeline, accelerator, epoch, is_final_validation=True, original_pipeline=pipe,
                clap_model=clap_model,
                clap_processor=processor, ref_audios=ref_audios
            )

    accelerator.end_training()