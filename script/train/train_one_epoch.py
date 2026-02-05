import torch
import torch.nn.functional as F
import os
from train_utils import unwrap_model
from diffusers.utils import convert_state_dict_to_diffusers
from peft.utils import get_peft_model_state_dict

def train_one_epoch(progress_bar, accelerator, unet, vae, text_encoder, noise_scheduler, weight_dtype, gradient_accumulation_steps, lora_layers, optimizer, lr_scheduler, checkpointing_steps, output_dir, avg_clap_score, avg_original_clap_score, kad_score_lora, kad_score_original, max_train_steps, num_steps_per_epoch, train_loss):
    for step, batch in progress_bar:
        num_steps_per_epoch += 1
        with accelerator.accumulate(unet):
            latents = vae.encode(batch["log_mel_spec"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            input_ids = batch["input_ids"].to(latents.device)   
            attention_mask = batch["attention_mask"].to(latents.device)
            
            # 3차원 텐서를 2차원으로 변환
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
                
            encoder_outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )

            # text_embeds 직접 사용
            prompt_embeds = encoder_outputs.text_embeds
            
            # L2 normalization 적용
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)
            
            # 배치 차원에 대해 반복하여 확장
            bs_embed, seq_len = prompt_embeds.shape
            num_waveforms_per_prompt = 1  # 학습 시에는 1로 설정
            prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)
            
            prompt_bsz = prompt_embeds.shape[0]
            latents_bsz = batch["log_mel_spec"].shape[0]

            if prompt_bsz != latents_bsz:
                repeat_factor = latents_bsz // prompt_bsz
                prompt_embeds = prompt_embeds.repeat_interleave(repeat_factor, dim=0)

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,
                class_labels=prompt_embeds,
                cross_attention_kwargs={"scale": 1.0},
                return_dict=False
            )[0]

            # 직접적인 loss 계산
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            avg_loss = accelerator.gather(loss).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps
            total_train_loss += avg_loss.item()
            total_steps += 1

            # Backpropagate
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = lora_layers
                accelerator.clip_grad_norm_(params_to_clip, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": train_loss})
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0
            global_step += 1

            if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                unwrapped_unet = unwrap_model(unet, accelerator)
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
                # unet.save_pretrained(save_path)
                # unet.save_lora_weights(save_directory=save_path, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet)), safe_serialization=True)
                # logger.info(f"Saved state to {save_path}")
                
        accelerator.log({"total_train_loss": total_train_loss / total_steps if total_steps > 0 else 0.0}, step=global_step)
        accelerator.log({
                            "avg_lora_clap_score": avg_clap_score,
                            "avg_original_clap_score": avg_original_clap_score,
                            "kad_score_lora": kad_score_lora,
                            "kad_score_original": kad_score_original
                        }, step=global_step)
        
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if global_step >= max_train_steps:
            break