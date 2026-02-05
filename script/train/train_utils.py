from matplotlib import pyplot as plt
import librosa
import librosa.display
import io
from PIL import Image
import torch
from contextlib import nullcontext
import numpy as np
import torch.nn.functional as F
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

SCALE_FACTOR = 100

if is_wandb_available():
    import wandb
    
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def plot_spectrogram_to_image(spec, title=None):
    plt.figure(figsize=(10, 4))
    # dB 스케일로 변환된 spectrogram을 사용한다고 가정
    img = librosa.display.specshow(spec, sr=16000, hop_length=512,
                                   x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(img, format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.tight_layout()

    # Plot을 BytesIO 버퍼에 PNG 형식으로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close() # Matplotlib figure 메모리 해제
    buf.seek(0)
    # 버퍼에서 PIL 이미지 로드
    image = Image.open(buf)
    return image

def log_validation(
    pipeline,
    accelerator,
    epoch,
    logger,
    num_validation_images,
    validation_prompt,
    is_final_validation=False,
    original_pipeline=None,
    clap_model=None,
    clap_processor=None,
    ref_audios=None
):
    logger.info(
        f"Running validation... \n Generating {num_validation_images} mel images with prompt:"
        f" {validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    if original_pipeline:
        original_pipeline = original_pipeline.to(accelerator.device)
        original_pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device)
    images, mel_spectrogram_images = [], []
    orig_images, orig_mel_spectrogram_images = [], []
    clap_scores, original_clap_scores = [], []
    kad_score_lora, kad_score_original=[],[]

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    def compute_clap_similarity(audio_waveform: np.ndarray, text: str) -> float:
        inputs = clap_processor(audios=audio_waveform, return_tensors="pt", sampling_rate=48000)
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_embed = clap_model.get_audio_features(**inputs)
            text_embed = clap_model.get_text_features(**clap_processor(text=text, return_tensors="pt", padding=True).to(accelerator.device))
            audio_embed = F.normalize(audio_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
            similarity = (audio_embed @ text_embed.T).item()
            return (similarity + 1) / 2

    with autocast_ctx:
        for i in range(num_validation_images):
            # 파인튜닝 모델 출력
            audio_output = pipeline(validation_prompt, num_inference_steps=50, generator=generator, audio_length_in_s=4.0)
            audio = audio_output.audios[0]
            images.append(audio)

            mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=1024, hop_length=512, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            spec_image = plot_spectrogram_to_image(mel_spec_db, title=f"Spectrogram {i}: {validation_prompt}")
            mel_spectrogram_images.append(spec_image)

            # CLAP 점수 계산 (파인튜닝 모델)
            if clap_model and clap_processor:
                print(f"# CLAP 점수 계산 (파인튜닝 모델)")
                # 48000Hz로 resample
                resampled_audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
                clap_score = compute_clap_similarity(resampled_audio, validation_prompt)
                clap_scores.append(clap_score)

            # 원본 모델 출력
            if original_pipeline:
                orig_output = original_pipeline(validation_prompt, num_inference_steps=50, generator=generator, audio_length_in_s=4.0)
                orig_audio = orig_output.audios[0]
                orig_images.append(orig_audio)

                orig_mel_spec = librosa.feature.melspectrogram(y=orig_audio, sr=16000, n_fft=1024, hop_length=512, n_mels=64)
                orig_mel_spec_db = librosa.power_to_db(orig_mel_spec, ref=np.max)
                orig_spec_image = plot_spectrogram_to_image(orig_mel_spec_db, title=f"Original Spectrogram {i}: {validation_prompt}")
                orig_mel_spectrogram_images.append(orig_spec_image)

                if clap_model and clap_processor:
                    print(f"# CLAP 점수 계산 (원본 모델)")
                    resampled_orig_audio = librosa.resample(orig_audio, orig_sr=16000, target_sr=48000)
                    original_clap_score = compute_clap_similarity(resampled_orig_audio, validation_prompt)
                    original_clap_scores.append(original_clap_score)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            if original_pipeline:
                orig_np_images = np.stack([np.asarray(img) for img in orig_images])
                tracker.writer.add_images(f"original_{phase_name}", orig_np_images, epoch, dataformats="NHWC")

        if tracker.name == "wandb":
            audio_logs = [wandb.Audio(img, sample_rate=16000, caption=f"{i}: {validation_prompt}") for i, img in enumerate(images)]
            tracker.log({phase_name: audio_logs})

            spec_logs = [wandb.Image(img, caption=f"{phase_name} Spectrogram {i}: {validation_prompt}") for i, img in enumerate(mel_spectrogram_images)]
            tracker.log({f"{phase_name}_spectrogram": spec_logs})

            if original_pipeline:
                orig_audio_logs = [wandb.Audio(img, sample_rate=16000, caption=f"Original {i}: {validation_prompt}") for i, img in enumerate(orig_images)]
                tracker.log({f"original_{phase_name}": orig_audio_logs})

                orig_spec_logs = [wandb.Image(img, caption=f"Original {phase_name} Spectrogram {i}: {validation_prompt}") for i, img in enumerate(orig_mel_spectrogram_images)]
                tracker.log({f"original_{phase_name}_spectrogram": orig_spec_logs})

            # wandb 로그
            if accelerator.is_main_process:
                print("*********TRUE!*********")
                if clap_scores:
                    avg_clap_score = np.mean(clap_scores)
                    wandb.log({f"{phase_name}_clap_score": avg_clap_score}, step=epoch + 10)
                    # tracker.log({f"{phase_name}_clap_score": avg_clap_score}, step=epoch)
                if original_clap_scores:
                    avg_original_clap_score = np.mean(original_clap_scores)
                    wandb.log({f"original_{phase_name}_clap_score": avg_original_clap_score}, step=epoch + 10)
                    # tracker.log({f"original_{phase_name}_clap_score": avg_original_clap_score}, step=epoch)

                if len(images) == len(orig_images):
                    kad_score_lora = compute_clap_kad_from_audio_lists(
                    ref_audios=ref_audios,
                    gen_audios=images,
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    device=accelerator.device
                    )

                    kad_score_original = compute_clap_kad_from_audio_lists(
                    ref_audios=ref_audios,
                    gen_audios=orig_images,
                    clap_model=clap_model,
                    clap_processor=clap_processor,
                    device=accelerator.device)

                    wandb.log({f"{phase_name}_kad_score_lora": kad_score_lora,
                               f"{phase_name}_kad_score_original": kad_score_original}, step=epoch + 10)

    return images, avg_clap_score, avg_original_clap_score, kad_score_lora, kad_score_original

def median_pairwise_distance(x, subsample=None):
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        distances = torch.pdist(x)
    return torch.median(distances).item()

def calc_kernel_audio_distance(x, y, device="cuda", bandwidth=None, kernel='gaussian', eps=1e-8):
    x = x.to(dtype=torch.float32, device=device)
    y = y.to(dtype=torch.float32, device=device)

    print(f"[KAD] x shape: {x.shape}, y shape: {y.shape}")
    print(f"[KAD] x norm: {torch.norm(x, dim=1)}")
    print(f"[KAD] y norm: {torch.norm(y, dim=1)}")

    if bandwidth is None:
        bandwidth = median_pairwise_distance(y)
        if bandwidth < 1e-6 or torch.isnan(torch.tensor(bandwidth)):
            print(f"[KAD] Warning: bandwidth too small or NaN, fallback to 1.0")
            bandwidth = 1.0  # 기본값 보정

    gamma = 1 / (2 * bandwidth**2 + eps)
    if kernel == 'gaussian':
        kernel_fn = lambda a: torch.exp(-gamma * a)
    elif kernel == 'iq':
        kernel_fn = lambda a: 1 / (1 + gamma * a)
    elif kernel == 'imq':
        kernel_fn = lambda a: 1 / torch.sqrt(1 + gamma * a)
    else:
        raise ValueError("Invalid kernel type")

    # x-x
    xx = x @ x.T
    x_sqnorms = torch.diagonal(xx)
    d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx
    k_xx = kernel_fn(d2_xx)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
    k_xx_mean = k_xx.sum() / (x.shape[0] * (x.shape[0] - 1))

    # y-y
    yy = y @ y.T
    y_sqnorms = torch.diagonal(yy)
    d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy
    k_yy = kernel_fn(d2_yy)
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
    k_yy_mean = k_yy.sum() / (y.shape[0] * (y.shape[0] - 1))

    # x-y
    xy = x @ y.T
    d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy
    k_xy = kernel_fn(d2_xy)
    k_xy_mean = k_xy.mean()

    result = k_xx_mean + k_yy_mean - 2 * k_xy_mean
    return result * SCALE_FACTOR

def compute_clap_kad_from_audio_lists(ref_audios, gen_audios, clap_model, clap_processor, device="cuda"):
    ref_embeddings = []
    gen_embeddings = []

    for idx, (ref_audio, gen_audio) in enumerate(zip(ref_audios, gen_audios)):
        ref_audio_rs = librosa.resample(ref_audio, orig_sr=16000, target_sr=48000)
        gen_audio_rs = librosa.resample(gen_audio, orig_sr=16000, target_sr=48000)

        ref_inputs = clap_processor(audios=ref_audio_rs, return_tensors="pt", sampling_rate=48000).to(device)
        gen_inputs = clap_processor(audios=gen_audio_rs, return_tensors="pt", sampling_rate=48000).to(device)

        with torch.no_grad():
            ref_embed = clap_model.get_audio_features(**ref_inputs)
            gen_embed = clap_model.get_audio_features(**gen_inputs)

            ref_embed = F.normalize(ref_embed, dim=-1)
            gen_embed = F.normalize(gen_embed, dim=-1)

        ref_embeddings.append(ref_embed.squeeze(0))
        gen_embeddings.append(gen_embed.squeeze(0))

    ref_tensor = torch.stack(ref_embeddings)  # [B, D]
    gen_tensor = torch.stack(gen_embeddings)  # [B, D]

    kad_score = calc_kernel_audio_distance(ref_tensor, gen_tensor, device=device)
    return kad_score.item()