"""SelfLift progressive-resolution sampler nodes.

Run the first sampling steps on a spatially downscaled latent, lift the clean
endpoint to the target resolution with the training-free Artifact-Aware
Consistency Lift (arXiv:2609.02036), re-noise it at the transition sigma, and
finish the schedule at full resolution.

Two front-ends share the same engine:
- SelfLiftAvatarH3Sampler: MiniMax H3 audio-video (nested AV latents; the audio stream
  has no spatial dimensions and retains its sampled state at the boundary).
  This is an experimental extension beyond the paper's image-model evaluation.
- SelfLiftAvatarImageSampler: compatible rectified-flow image models.
"""

import copy
import logging
import os
import time

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from server import PromptServer

from . import selflift
from . import avatar_masks
from .avatar_sampling import native_mask_model
from . import h3_upscaler
from . import h3_tiling
from . import h3_temporal
from . import spatial_anchor as spatial_anchor_utils
from .tiling_ui import format_plan
from . import h3_tst
from .diagnostics import log_memory

_HIGHRES_TILING_MODES = ("spatial", "temporal", "global_stream", "global", "global_ram")
_GLOBAL_TILING_MODES = frozenset(("global_stream", "global", "global_ram"))


def _global_tiled_model(model, chunk_size, mode, weight_cache_gb):
    """Build the full-sequence H3 memory-saving wrapper.

    Imported lazily so the existing spatial path still loads on installations
    without the optional comfy_kitchen/native-sparse dependencies.
    """
    try:
        from . import h3_global_tiling
    except Exception as exc:
        raise RuntimeError(
            "selflift-Avatar: global high-resolution mode is unavailable; "
            "check ComfyUI/native sparse dependencies"
        ) from exc
    return h3_global_tiling.tiled_model(
        model, chunk_size,
        use_ram=mode == "global_ram",
        use_stream=mode == "global_stream",
        weight_cache_gb=weight_cache_gb,
    )


class _StageTimer:
    def __init__(self, stage, device, resolution=None, pixel_scale=1):
        self.stage = stage
        self.resolution = resolution
        self.pixel_scale = pixel_scale
        self.device = torch.device(device)
        self.synchronize = os.environ.get("SELFLIFT_AVATAR_TIMING_SYNC", "0") == "1" and self.device.type == "cuda"
        self.started = self.previous = self._now()
        timing_mode = "CUDA-synchronized wall time" if self.synchronize else "wall time; no forced CUDA sync"
        suffix = ""
        if resolution is not None:
            pixels = tuple(int(value * self.pixel_scale) for value in resolution)
            suffix = f" latent={resolution} pixels={pixels}"
        logging.info("[selflift-Avatar timing] %s start (%s)%s", stage, timing_mode, suffix)

    def _now(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label):
        current = self._now()
        logging.info("[selflift-Avatar timing] %s %s: %.3fs", self.stage, label, current - self.previous)
        self.previous = current

    def finish(self):
        current = self._now()
        logging.info("[selflift-Avatar timing] %s total: %.3fs; after last mark: %.3fs",
                     self.stage, current - self.started, current - self.previous)


def _upscaler_input():
    models = ["none"] + h3_upscaler.list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default,
                     "tooltip": "H3 潜空间放大模型，来自 models/latent_upscale_models。默认选择检测到的第一个 H3 模型；选择 none 时使用最近邻升采样。"})


def _streams(samples):
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    if not callable(getattr(sampler, "sample", None)):
        raise ValueError("SelfLift requires a ComfyUI SAMPLER")


def _uses_euler_transition(sampler):
    return (isinstance(sampler, comfy.samplers.KSAMPLER)
            and type(sampler).sample is comfy.samplers.KSAMPLER.sample
            and sampler.sampler_function is comfy.k_diffusion.sampling.sample_euler
            and sampler.extra_options.get("s_churn", 0.0) == 0.0)


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("selflift-Avatar: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("selflift-Avatar: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"selflift-Avatar: transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("selflift-Avatar: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("selflift-Avatar: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError("selflift-Avatar: the high-resolution starting sigma must be less than 1")


def _validate_latent_input(latent_image):
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("selflift-Avatar: expected an image or video latent")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("selflift-Avatar: latent streams must be nonempty with matching batches")
    return avatar_masks.normalize_masks(latent_image.get("noise_mask"), streams)


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """Keyframe cond latents share the generation grid; resize them to the low-res one."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                if lat.ndim == 5:
                    # spatial-only resize per frame; never interpolates across time
                    batch, channels, frames = lat.shape[:3]
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, lat.shape[-2], lat.shape[-1]),
                        size=(h, w), mode="bilinear", align_corners=False
                    )
                    resized_latent = resized_latent.reshape(batch, frames, channels, h, w).permute(0, 2, 1, 3, 4)
                else:
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float(), size=(h, w), mode="bilinear", align_corners=False
                    )
                # per-channel, per-frame mean match: fixes bilinear's color drift without
                # restoring variance the low-res grid cannot represent
                source_mean = lat.float().mean(dim=(-2, -1), keepdim=True)
                resized_mean = resized_latent.mean(dim=(-2, -1), keepdim=True)
                kf["latent"] = (resized_latent + (source_mean - resized_mean)).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """Decode transition intermediates to PNGs when SELFLIFT_AVATAR_DEBUG=1."""
    if os.environ.get("SELFLIFT_AVATAR_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    logging.warning("selflift-Avatar: debug decoding enabled; intermediate video decodes can substantially increase time and memory")
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False, high_res_model=None, tiling_mode="auto", tiling_tiles=2,
                       tiling_axis="auto", on_tiling_plan=None,
                       highres_tiling_mode="spatial", highres_token_chunk=4096,
                       highres_weight_cache_gb=4.0, spatial_anchor_mode="off",
                       tiling_overlap=None, diagnostics_out=None,
                       temporal_chunk_frames=124, temporal_overlap_frames=22,
                       temporal_seam_lock=True):
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("selflift-Avatar: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("selflift-Avatar: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("selflift-Avatar: weights must satisfy 0 <= w_min <= w_max <= 1")
    noise_masks = _validate_latent_input(latent_image)
    if noise_masks is not None and (
            not isinstance(sampler, comfy.samplers.KSAMPLER)
            or type(sampler).sample is not comfy.samplers.KSAMPLER.sample):
        raise ValueError("selflift-Avatar: noise_mask requires the standard ComfyUI KSAMPLER sample interface")
    global_tiling = (highres_tiling and isinstance(highres_tiling_mode, str)
                     and highres_tiling_mode in _GLOBAL_TILING_MODES)
    if spatial_anchor_mode not in ("off", "lowres_global"):
        raise ValueError(
            "selflift-Avatar: spatial_anchor_mode must be 'off' or 'lowres_global'"
        )
    use_lowres_anchor = (highres_tiling and highres_tiling_mode == "spatial"
                         and spatial_anchor_mode == "lowres_global")
    content_state = h3_tiling.SpatialContentState() if use_lowres_anchor else None
    if spatial_anchor_mode != "off" and not use_lowres_anchor:
        logging.warning(
            "selflift-Avatar: spatial_anchor_mode is only applied in spatial tiling; "
            "ignoring it for highres_tiling_mode=%s", highres_tiling_mode
        )
    if highres_tiling:
        if highres_tiling_mode not in _HIGHRES_TILING_MODES:
            raise ValueError(
                f"selflift-Avatar: unknown highres_tiling_mode {highres_tiling_mode!r}; "
                f"expected one of {_HIGHRES_TILING_MODES}"
            )
        if highres_tiling_mode == "spatial":
            h3_tiling.validate_options(tiling_mode, tiling_tiles, tiling_axis)
            avatar_masks.validate_tiling_masks(noise_masks)
            if noise_masks is not None:
                logging.info("[selflift-Avatar tiling] video=generate, audio=fully preserved; full audio conditioning on every spatial tile")
        elif highres_tiling_mode in _GLOBAL_TILING_MODES:
            if not isinstance(highres_token_chunk, int) or not 128 <= highres_token_chunk <= 16384:
                raise ValueError("selflift-Avatar: highres_token_chunk must be an integer from 128 to 16384")
            if not 0.0 <= float(highres_weight_cache_gb) <= 64.0:
                raise ValueError("selflift-Avatar: highres_weight_cache_gb must be between 0 and 64")
            logging.info("[selflift-Avatar global] mode=%s token_chunk=%d weight_cache=%.2f GiB; "
                         "full-sequence video/audio masks enabled",
                         highres_tiling_mode, highres_token_chunk, highres_weight_cache_gb)

    model = low_res_model
    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    temporal_tiling = highres_tiling and highres_tiling_mode == "temporal"
    hires_base = high_res_model if high_res_model is not None else model
    _validate_sampling(hires_base.get_model_object("model_sampling"), sampler)
    if temporal_tiling:
        if not isinstance(hires_base.model, comfy.model_base.MiniMaxH3):
            raise ValueError("时间分块只支持 MiniMax H3 模型。")
        if not nested or len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
            raise ValueError("时间分块要求 H3 音视频潜变量。")
        temporal_plan = h3_temporal.plan_chunks(
            streams[0].shape[2], streams[1].shape[-1], temporal_chunk_frames, temporal_overlap_frames)
        h3_temporal.validate_conditioning(hires_base, positive)
        h3_temporal.validate_conditioning(hires_base, negative)
        if temporal_seam_lock and len(temporal_plan) > 1 and (
                not isinstance(sampler, comfy.samplers.KSAMPLER)
                or type(sampler).sample is not comfy.samplers.KSAMPLER.sample):
            raise ValueError("时间接缝锁定要求标准 ComfyUI KSAMPLER 接口。")
    reuse_euler = _uses_euler_transition(sampler)
    if not reuse_euler:
        logging.info("[selflift-Avatar sampler] generic transition: one extra low-resolution "
                     "prediction; sampler history restarts at the resolution boundary")
    if highres_tiling and highres_tiling_mode == "spatial":
        high_model = h3_tiling.tiled_model(hires_base, [tuple(stream.shape) for stream in streams],
                                           mode=tiling_mode, tiles=tiling_tiles, axis=tiling_axis,
                                           on_plan=on_tiling_plan, overlap=tiling_overlap,
                                           content_state=content_state)
    elif global_tiling:
        high_model = _global_tiled_model(
            hires_base, highres_token_chunk, highres_tiling_mode, highres_weight_cache_gb
        )
        if on_tiling_plan is not None:
            plan = {
                "kind": "global",
                "mode": highres_tiling_mode,
                "token_chunk": int(highres_token_chunk),
                "weight_cache_gb": float(highres_weight_cache_gb),
                "full_sequence": True,
            }
            fallback = getattr(high_model, "_selflift_global_fallback", None)
            if fallback:
                plan["fallback"] = str(fallback)
            on_tiling_plan(plan)
    else:
        high_model = hires_base
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)
    logging.info("[selflift-Avatar plan] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                 "low_steps=%d high_steps=%d sigma_prediction=%.8g sigma_resume=%.8g "
                 "rho=%.4f weights=(%.4f, %.4f) direct_lift=%s pixel_anchor=%s cfg=%.4f mask=%s hires_model=%s",
                 low_shape, tuple(streams[0].shape), H / h, W / w,
                 transition_step, sigmas.numel() - 1 - transition_step,
                 sigmas[transition_step - 1 if reuse_euler else transition_step].item(), sigmas[transition_step].item(),
                 rho, w_min, w_max,
                 "skipped" if rho == 1.0 and w_min == 1.0 else "external" if latent_lifter is not None else latent_upsample,
                 rho > 0.0 and w_max > 0.0, cfg,
                 "none" if noise_masks is None else str([tuple(m.shape) for m in noise_masks]),
                 "custom" if hires_base is not model else "same")

    device = comfy.model_management.intermediate_device()
    source_video = streams[0].to(device)
    audio_streams = [s.to(device) for s in streams[1:]]

    if video:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(t, h, w), mode="trilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    else:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(h, w), mode="bilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    m_full = m_low = None
    low_masks = None
    if noise_masks is not None:
        m_full = noise_masks[0]
        m_low = avatar_masks.resize_spatial(m_full, (h, w))
        low_masks = [m_low] + noise_masks[1:]
    video_anchor = source_video
    low_latent = _pack([low_video] + audio_streams, nested)
    low_model = native_mask_model(model, [low_video] + audio_streams, low_masks, nested)
    if not temporal_tiling:
        high_model = native_mask_model(high_model, [video_anchor] + audio_streams, noise_masks, nested)
    del source_video, low_video
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if video:  # keyframe cond latents share the generation grid on MiniMax H3
        positive_low = _resize_keyframes(positive, h, w)
        negative_low = _resize_keyframes(negative, h, w)
    else:
        positive_low, negative_low = positive, negative

    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        low_evaluations += 1
        step = low_evaluations - 1 if reuse_euler else min(max(int(step), 0), transition_step - 1)
        if reuse_euler and low_evaluations > transition_step:
            raise RuntimeError("selflift-Avatar: too many low-resolution callbacks for the Euler schedule")
        if reuse_euler and step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        # Preview decoders expect the target grid.  Decode a temporary lifted
        # preview while keeping the sampler state at its true low resolution.
        preview_x0 = x0
        if video:
            preview_streams, preview_nested = _streams(x0)
            preview_video = torch.nn.functional.interpolate(
                preview_streams[0].float(), size=(t, H, W), mode="trilinear", align_corners=False
            ).to(preview_streams[0].dtype)
            preview_x0 = _pack([preview_video] + preview_streams[1:], preview_nested)
            del preview_video, preview_streams
        elif x0.ndim == 4:
            preview_x0 = torch.nn.functional.interpolate(
                x0.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(x0.dtype)
        result = callback(step, preview_x0, preview_x0, total_steps)
        del preview_x0
        low_timer.mark(f"step {step + 1}/{transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    # Standard Euler reuses its last prediction. Other solvers finish their low-res
    # schedule before a separate prediction at the boundary, keeping their true state.
    resolution_scale = (getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1)
                        if video else getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1))
    low_timer = _StageTimer("low_resolution", low_model.load_device, (h, w), resolution_scale)
    log_memory("low_resolution start", low_model.load_device)
    low_sampled = comfy.samplers.sample(low_model, noise_low, positive_low, negative_low, cfg, low_model.load_device,
                                       sampler, sigmas[:transition_step + 1], low_model.model_options,
                                       latent_image=low_latent, callback=callback_low,
                                       disable_pbar=disable_pbar, seed=seed)
    if reuse_euler and low_evaluations != transition_step:
        raise RuntimeError(f"selflift-Avatar: expected {transition_step} low-resolution callbacks, received {low_evaluations}; check sampler wrappers")
    if not reuse_euler:
        def capture_endpoint(step, x0, x, total):
            transition["x0"] = x0

        endpoint_streams, _ = _streams(low_sampled)
        endpoint_noise = _pack([torch.zeros_like(s) for s in endpoint_streams], nested)
        endpoint_sigmas = torch.stack((sigmas[transition_step], sigmas.new_zeros(())))
        endpoint_sampler = comfy.samplers.sampler_object("euler")
        if noise_masks is not None:
            endpoint_inpaint_noise = (comfy.utils.pack_latents(noise_low.unbind())[0]
                                      if nested else noise_low)

            def sample_endpoint(model_k, x, sigmas, **kwargs):
                model_k.noise = endpoint_inpaint_noise.to(x)
                return comfy.k_diffusion.sampling.sample_euler(model_k, x, sigmas, **kwargs)

            endpoint_sampler = comfy.samplers.KSAMPLER(sample_endpoint)
        comfy.samplers.sample(low_model, endpoint_noise, positive_low, negative_low, cfg, low_model.load_device,
                              endpoint_sampler, endpoint_sigmas, low_model.model_options,
                              latent_image=low_sampled, callback=capture_endpoint,
                              disable_pbar=True, seed=seed)
        if "x0" not in transition:
            raise RuntimeError("selflift-Avatar: missing low-resolution transition prediction; check sampler wrappers")
        del endpoint_streams, endpoint_noise, endpoint_sampler
        if noise_masks is not None:
            del endpoint_inpaint_noise
        low_timer.mark("boundary prediction (one extra low-resolution evaluation)")
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)
    transition_timer = _StageTimer("transition", model.load_device, (H, W), resolution_scale)
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_k = sigmas[transition_step - 1 if reuse_euler else transition_step]
    sigma_next = sigmas[transition_step]
    if reuse_euler:
        low_streams, _ = _streams(transition.pop("state"))
        auxiliary_resume = [model_sampling.inverse_noise_scaling(
            sigma_next, _euler_step(state.to(device), denoised.to(device), sigma_k, sigma_next))
            for state, denoised in zip(low_streams[1:], x0_streams[1:])]
    else:
        low_streams, _ = _streams(model.model.process_latent_in(low_sampled))
        auxiliary_resume = [state.to(device) for state in low_streams[1:]]
    del low_sampled
    latent_format = model.get_model_object("latent_format")
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    # Keep the aligned low-resolution endpoint as an optional global H3
    # reference. This is only consumed by the experimental spatial-anchor
    # node, so the original spatial path remains unchanged.
    lowres_anchor = z0_low_vae.detach().clone() if use_lowres_anchor else None
    if content_state is not None and lowres_anchor is not None:
        content_state.set_from_video(lowres_anchor)
        logging.info(
            "[selflift-Avatar content-aware split] scored low-resolution context "
            "%s; cuts will avoid high-gradient rows/columns",
            tuple(lowres_anchor.shape),
        )
    del low_latent, noise_low, low_streams, x0_streams, positive_low, negative_low
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    z0_high = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=m_full)
    if m_full is not None:
        # restore the keep-region with the true original instead of any lifted estimate
        m_cast = m_full.to(device=z0_high.device, dtype=z0_high.dtype)
        clean_anchor = latent_format.process_in(video_anchor.to(z0_high))
        z0_high = z0_high * m_cast + clean_anchor * (1.0 - m_cast)
    if os.environ.get("SELFLIFT_AVATAR_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    # Reuse the Euler boundary interval, or re-noise directly at the completed
    # low-resolution schedule's ending sigma for other solvers.
    video_noise = comfy.sample.prepare_noise(z0_high, (seed + 1) % (1 << 64),
                                             latent_image.get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    if reuse_euler:
        video_state = _euler_step(video_state, z0_high, sigma_k, sigma_next)
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, video_state)] + auxiliary_resume
    del z0_high, video_noise, video_state, auxiliary_resume
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
    del resume_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)

    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        high_evaluations += 1
        step = high_evaluations - 1 if reuse_euler else min(max(int(step), 0), total_steps - transition_step - 1)
        if reuse_euler and high_evaluations > total_steps - transition_step:
            raise RuntimeError("selflift-Avatar: too many high-resolution callbacks for the Euler schedule")
        result = callback(step + transition_step, x0, x, total_steps)
        high_timer.mark(f"step {step + 1}/{total_steps - transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    high_timer = _StageTimer("high_resolution", high_model.load_device, (H, W), resolution_scale)
    positive_high = positive
    if lowres_anchor is not None:
        positive_high = spatial_anchor_utils.append_global_video_reference(positive, lowres_anchor)
        logging.info(
            "[selflift-Avatar spatial anchor] appended aligned low-resolution video "
            "latent %s as an uncropped global reference for every spatial tile",
            tuple(lowres_anchor.shape),
        )
    if highres_tiling:
        if highres_tiling_mode == "spatial":
            logging.info("[selflift-Avatar plan] high-resolution spatial tiling enabled; mode=%s axis=%s", tiling_mode, tiling_axis)
        elif highres_tiling_mode == "temporal":
            logging.info("[selflift-Avatar plan] high-resolution temporal tiling enabled; chunk_frames=%d overlap_frames=%d seam_lock=%s",
                         temporal_chunk_frames, temporal_overlap_frames, temporal_seam_lock)
        else:
            logging.info("[selflift-Avatar plan] high-resolution global context enabled; mode=%s token_chunk=%d "
                         "weight_cache=%.2f GiB; masks remain full-sequence",
                         highres_tiling_mode, highres_token_chunk, highres_weight_cache_gb)
    if temporal_tiling:
        def callback_temporal(step, x0, x, total):
            callback(step + transition_step, x0, x, transition_step + total)
            high_timer.mark(f"时间窗口采样 {step + 1}/{total}")

        out = h3_temporal.sample(
            high_model, resume_noise, positive_high, negative, cfg, sampler,
            sigmas[transition_step:], resume_latent, seed,
            chunk_frames=temporal_chunk_frames, overlap_frames=temporal_overlap_frames,
            seam_lock=temporal_seam_lock, disable_pbar=disable_pbar, on_plan=on_tiling_plan,
            callback=callback_temporal, anchors=[video_anchor] + audio_streams,
            masks=noise_masks, require_callbacks=reuse_euler,
        )
    else:
        out = comfy.samplers.sample(high_model, resume_noise, positive_high, negative, cfg, high_model.load_device,
                                    sampler, sigmas[transition_step:], high_model.model_options,
                                    latent_image=resume_latent, callback=callback_high,
                                    disable_pbar=disable_pbar, seed=seed)
    del resume_latent, resume_noise, lowres_anchor, positive_high
    if reuse_euler and not temporal_tiling and high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"selflift-Avatar: expected {total_steps - transition_step} high-resolution callbacks, received {high_evaluations}; check sampler wrappers")

    result = latent_image.copy()
    if noise_masks is not None:
        # Only exact keep-regions are restored here; do not blend soft masks twice.
        out_streams, _ = _streams(out)
        out = _pack(avatar_masks.restore_kept(
            out_streams, [video_anchor] + audio_streams, noise_masks), nested)
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    high_timer.finish()
    log_memory("high_resolution end", model.load_device)
    if diagnostics_out is not None and content_state is not None:
        diagnostics_out.update({
            "axis": "H" if content_state.axis == 3 else "W",
            "length": content_state.length,
            "image_shape": content_state.image_shape,
            "regions": content_state.regions,
            "seams": list(content_state.seam_metrics),
        })
    return result


class SelfLiftAvatarH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "low_res_model": ("MODEL", {"tooltip": "高噪声阶段使用的模型。低噪声阶段模型未连接时，也会继续用于低噪声阶段。"}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "在噪声阶段切换时，用于像素空间重编码锚点的视频 VAE。"}),
            "latent_image": ("LATENT", {"tooltip": "定义目标尺寸和时长的 H3 音视频潜变量。也支持已有潜变量、Tensor 遮罩以及 H3 NestedTensor 音视频遮罩；0 表示保留内容，1 表示生成内容。空间分块要求视频遮罩全为 1、音频遮罩全为 0；全局模式会保留完整序列遮罩。"}),
            "sampler": ("SAMPLER", {"tooltip": "支持 ComfyUI 采样器。标准 Euler 且 s_churn=0 时复用切换预测；其他采样器会在切换处增加一次高噪声阶段预测，并重建采样历史。实际兼容性取决于模型和采样器。"}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "高噪声阶段包含的 sigma 区间数量。其他采样器会在切换处额外预测一次；多次评估或自适应采样器的模型调用次数不等于步数。"}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "高噪声阶段使用的空间尺寸比例，论文默认值为 0.5。"}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "向像素 VAE 锚点修正的最高风险时空位置比例。H3 默认值 0 仅使用外部潜变量放大模型，不执行 VAE 往返；当 upscaler_model=none 时，建议从 0.6 附近开始。"}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "修正强度下限。H3 的大范围最近邻升采样误差可能需要设为 1.0；0.5 是图像模型论文设置。"}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "修正强度上限。用于 SelfLift-zero 诊断时保持为 1.0。"}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "high_res_model": ("MODEL", {"tooltip": "低噪声阶段使用的模型，在分辨率提升后运行。未连接时使用高噪声阶段模型。两个阶段必须兼容架构、潜变量格式、采样参数化方式、VAE 和条件输入。"}),
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "低噪声阶段省显存：开启", "label_off": "低噪声阶段省显存：关闭", "tooltip": "仅作用于第二段低噪声高分辨率采样。空间模式按宽高分块；时间模式同步切分视频、音频和遮罩；全局模式保留完整上下文并按 token 分批。这些策略互斥。"}),
            "tiling_mode": (["auto", "manual"], {"default": "auto", "tooltip": "仅用于空间模式。auto 自动根据显存估算选择 1 至 8 块；manual 手动使用 tiling_tiles，不会自动增加分块数量。"}),
            "tiling_tiles": ([2, 4, 6, 8], {"default": 2, "tooltip": "仅用于空间模式。手动分块数量可选 2、4、6、8；自动模式会忽略此项。尺寸较小时实际分块数量可能减少。"}),
            "tiling_axis": (["auto", "width", "height"], {"default": "auto", "tooltip": "仅用于空间模式。auto 沿图像块网格较长的一边分块；width 分成左右条带；height 分成上下条带；音频和时间维度不会拆分。"}),
            "highres_tiling_mode": (["spatial", "temporal", "global_stream", "global", "global_ram"], {"default": "spatial", "tooltip": "省显存策略。spatial 进行空间分块；temporal 在低噪声阶段按 H3 时间窗口分块并融合重叠帧；global_stream、global 和 global_ram 保留完整序列上下文并按 token 分批。"}),
            "highres_token_chunk": ("INT", {"default": 4096, "min": 128, "max": 16384, "step": 128, "tooltip": "仅用于全局模式。每批投影和前馈处理的 token 数量。减小批量通常可以降低中间显存，但会增加处理开销；它不是空间分块大小，也不是时间窗口。"}),
            "highres_weight_cache_gb": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5, "tooltip": "仅用于全局流模式。DynamicVRAM 常驻权重缓存上限，单位 GiB；它不是总显存或系统内存。设为 0 会使用自动策略，不会关闭缓存。"}),
            "temporal_chunk_frames": ("INT", {"default": 124, "min": 22, "max": 3592, "step": 17, "tooltip": "仅用于时间模式。窗口长度，单位为视频像素帧，采用 17k+5 网格；124 帧在 24 帧率下约为 5.17 秒。可使用 56、73、90、107、124 等值。窗口越短，单次前向的显存占用通常越小。"}),
            "temporal_overlap_frames": ("INT", {"default": 22, "min": 5, "max": 107, "step": 17, "tooltip": "仅用于时间模式。相邻窗口重叠的像素帧数，采用 17k+5 网格，且必须小于窗口长度。重叠区线性融合，音频按整段的绝对时间边界同步切分。"}),
            "temporal_seam_lock": ("BOOLEAN", {"default": True, "label_on": "时间接缝锁定：开启", "label_off": "时间接缝锁定：关闭", "tooltip": "仅用于时间模式。固定后续窗口开头的 2 个视频 latent（局部 5 个像素帧），锚定到已生成的重叠内容；用户原本要求保留的区域优先。需要标准 ComfyUI KSAMPLER，不锁定音频。"}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift-Avatar"

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, high_res_model=None, highres_tiling=False,
               tiling_mode="auto", tiling_tiles=2, tiling_axis="auto",
               highres_tiling_mode="spatial", highres_token_chunk=4096,
               highres_weight_cache_gb=4.0, unique_id=None,
               temporal_chunk_frames=124, temporal_overlap_frames=22, temporal_seam_lock=True):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)
        if not highres_tiling:
            summary = "OFF / 低噪声阶段省显存已关闭"
        elif highres_tiling_mode == "spatial":
            summary = "准备中：空间分块｜模式={}｜轴={}｜分块={}".format(
                tiling_mode, tiling_axis, tiling_tiles if tiling_mode == "manual" else "自动")
        elif highres_tiling_mode == "temporal":
            summary = "准备中：时间分块｜窗口={} 帧｜重叠={} 帧｜接缝锁定={}".format(
                temporal_chunk_frames, temporal_overlap_frames, "开启" if temporal_seam_lock else "关闭")
        else:
            cache = (f"｜权重缓存={float(highres_weight_cache_gb):.1f} GiB"
                     if highres_tiling_mode == "global_stream" else "")
            summary = "准备中：{}｜token 分块={}{}".format(
                highres_tiling_mode, highres_token_chunk, cache)
        last_plan = None

        def send_status(text):
            server = getattr(PromptServer, "instance", None)
            if unique_id is not None and server is not None and server.client_id is not None:
                server.send_sync("selflift-avatar-tiling", {"node_id": str(unique_id), "text": text, "plan": last_plan}, server.client_id)

        def report(plan):
            nonlocal summary, last_plan
            last_plan = plan
            summary = format_plan(plan)
            send_status("RUNNING / 高清采样中\n" + summary)

        send_status(summary)
        diagnostics = {}
        spatial_anchor_mode = ("lowres_global" if highres_tiling and highres_tiling_mode == "spatial"
                               else "off")
        spatial_overlap = 6 if spatial_anchor_mode == "lowres_global" else None
        try:
            output = progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                        transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                        latent_lifter=lifter, highres_tiling=highres_tiling, high_res_model=high_res_model,
                                        tiling_mode=tiling_mode, tiling_tiles=tiling_tiles, tiling_axis=tiling_axis,
                                        on_tiling_plan=report, highres_tiling_mode=highres_tiling_mode,
                                        highres_token_chunk=highres_token_chunk,
                                        highres_weight_cache_gb=highres_weight_cache_gb,
                                        spatial_anchor_mode=spatial_anchor_mode,
                                        tiling_overlap=spatial_overlap,
                                        diagnostics_out=diagnostics,
                                        temporal_chunk_frames=temporal_chunk_frames,
                                        temporal_overlap_frames=temporal_overlap_frames,
                                        temporal_seam_lock=temporal_seam_lock)
        except Exception:
            send_status("FAILED / 运行失败：以下不是成功结果；详见错误报告\n" + summary)
            raise
        if highres_tiling and last_plan is None:
            summary = "NOT RUN / 本次未执行高清分块规划（例如空采样日程）"
        seams = diagnostics.get("seams") or []
        if seams:
            values = [float(item.get("relative_diff", 0.0)) for item in seams]
            summary += (f"\nSeam diag / 接缝差异: {len(seams)} boundaries | "
                        f"mean={sum(values) / len(values):.4f} | max={max(values):.4f}")
        summary = "DONE / 上次运行完成\n" + summary
        ui = {"selflift_tiling": [summary], "selflift_tiling_plan": [last_plan]}
        if diagnostics:
            ui["selflift_seam_diag"] = [diagnostics]
        return {"ui": ui, "result": (output,)}


class SelfLiftAvatarH3SpatialAnchoredSampler(SelfLiftAvatarH3Sampler):
    """Experimental spatial-only sampler with a full-frame low-resolution anchor.

    This node is deliberately separate from ``SelfLiftAvatarH3Sampler`` so the
    original spatial/global behavior can be kept as an A/B control. It adds the
    aligned low-resolution SelfLift endpoint as an uncropped H3 video reference
    on the high-resolution stage, and exposes a wider explicit stripe overlap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        schema = copy.deepcopy(SelfLiftAvatarH3Sampler.INPUT_TYPES())
        optional = schema["optional"]
        optional["highres_tiling"] = ("BOOLEAN", {
            "default": True,
            "label_on": "空间分块：开启",
            "label_off": "空间分块：关闭",
            "tooltip": "测试节点，仅支持空间分块。关闭后可执行完整帧对照。",
        })
        optional["tiling_mode"] = (["manual", "auto"], {
            "default": "manual",
            "tooltip": "建议手动进行 A/B 对比：先使用 2 或 3 条带；只有显存不足时再增加数量。",
        })
        optional["tiling_tiles"] = ([2, 3, 4, 6, 8], {
            "default": 2,
            "tooltip": "较少的条带可以保留更多上下文。使用 3 条带时，居中主体可能完整保留在中间条带中。",
        })
        optional["tiling_overlap"] = ("INT", {
            "default": 6, "min": 1, "max": 16, "step": 1,
            "tooltip": "H3 块单位中的显式重叠范围。数值越大，每个条带获得的邻域上下文越多，但显存占用也会增加。",
        })
        optional["spatial_anchor"] = (["lowres_global", "off"], {
            "default": "lowres_global",
            "tooltip": "lowres_global 会把对齐后的完整帧高噪声端点作为未裁剪的 H3 视频参考追加到低噪声阶段；off 表示对照路径。",
        })
        optional["highres_tiling_mode"] = (["spatial"], {
            "default": "spatial",
            "tooltip": "此测试节点只支持空间分块；需要全局模式时请使用标准采样器。",
        })
        optional.pop("highres_token_chunk", None)
        optional.pop("highres_weight_cache_gb", None)
        optional.pop("temporal_chunk_frames", None)
        optional.pop("temporal_overlap_frames", None)
        optional.pop("temporal_seam_lock", None)
        return schema

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, high_res_model=None,
               highres_tiling=True, tiling_mode="manual", tiling_tiles=2, tiling_axis="auto",
               spatial_anchor="lowres_global", tiling_overlap=6, highres_tiling_mode="spatial",
               unique_id=None):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        if spatial_anchor not in ("lowres_global", "off"):
            raise ValueError("SelfLift H3 spatial anchor must be 'lowres_global' or 'off'")

        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning(
                    "SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; "
                    "select upscaler_model=none to test the paper's SelfLift-zero direct route"
                )
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)

        if not highres_tiling:
            summary = "OFF / 空间分块已关闭（全帧对照）"
        elif spatial_anchor == "lowres_global":
            summary = ("Preparing / 准备中: spatial + full-frame low-res anchor | "
                       "{} | axis={} | tiles={} | overlap={}\n"
                       "Actual plan appears at high-resolution preparation / 高清阶段准备时显示实际方案").format(
                           tiling_mode, tiling_axis,
                           tiling_tiles if tiling_mode == "manual" else "auto",
                           tiling_overlap)
        else:
            summary = ("Preparing / 准备中: spatial control (anchor off) | "
                       "{} | axis={} | tiles={} | overlap={}").format(
                           tiling_mode, tiling_axis,
                           tiling_tiles if tiling_mode == "manual" else "auto",
                           tiling_overlap)

        last_plan = None

        def send_status(text):
            server = getattr(PromptServer, "instance", None)
            if unique_id is not None and server is not None and server.client_id is not None:
                server.send_sync(
                    "selflift-avatar-tiling",
                    {"node_id": str(unique_id), "text": text, "plan": last_plan},
                    server.client_id,
                )

        def report(plan):
            nonlocal summary, last_plan
            last_plan = plan
            summary = format_plan(plan)
            send_status("RUNNING / 高清采样中\n" + summary)

        send_status(summary)
        diagnostics = {}
        try:
            output = progressive_sample(
                low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                latent_lifter=lifter, highres_tiling=highres_tiling, high_res_model=high_res_model,
                tiling_mode=tiling_mode, tiling_tiles=tiling_tiles, tiling_axis=tiling_axis,
                on_tiling_plan=report, highres_tiling_mode="spatial",
                spatial_anchor_mode=spatial_anchor, tiling_overlap=tiling_overlap,
                diagnostics_out=diagnostics,
            )
        except Exception:
            send_status("FAILED / 运行失败：以下不是成功结果；详见错误报告\n" + summary)
            raise
        if highres_tiling and last_plan is None:
            summary = "NOT RUN / 本次未执行高清分块规划（例如空采样日程）"
        seams = diagnostics.get("seams") or []
        if seams:
            values = [float(item.get("relative_diff", 0.0)) for item in seams]
            summary += (f"\nSeam diag / 接缝差异: {len(seams)} boundaries | "
                        f"mean={sum(values) / len(values):.4f} | max={max(values):.4f}")
        summary = "DONE / 上次运行完成\n" + summary
        ui = {"selflift_tiling": [summary], "selflift_tiling_plan": [last_plan]}
        if diagnostics:
            ui["selflift_seam_diag"] = [diagnostics]
        return {"ui": ui, "result": (output,)}


class SelfLiftAvatarImageSampler:
    """SelfLift-zero for compatible rectified-flow image backbones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "low_res_model": ("MODEL", {"tooltip": "高噪声阶段使用的模型。低噪声阶段模型未连接时，也会继续用于低噪声阶段。"}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "在噪声阶段切换时，用于像素空间重编码锚点的 VAE。"}),
            "latent_image": ("LATENT", {"tooltip": "定义目标尺寸的图像潜变量。也支持已有内容以及 BHW/BCHW Tensor 遮罩；0 表示保留内容，1 表示生成内容。"}),
            "sampler": ("SAMPLER", {"tooltip": "支持 ComfyUI 采样器。标准 Euler 且 s_churn=0 时复用切换预测；其他采样器会在切换处增加一次高噪声阶段预测，并重建采样历史。实际兼容性取决于模型和采样器。"}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "高噪声阶段的 sigma 区间数。其他采样器会在切换处额外预测一次；多次评估或自适应采样器的模型调用数不等于步数。"}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "高噪声阶段使用的空间尺寸比例，论文默认值为 0.5。"}),
            "rho": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "向像素 VAE 锚点修正的位置比例。论文设置：FLUX.2-Klein 为 0.4，Z-Image-Turbo 为 0.3。"}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "latent_upsample": (["nearest", "bilinear"], {"default": "nearest", "tooltip": "直接提升潜变量时使用的插值方式，论文默认使用最近邻。"}),
        }, "optional": {
            "high_res_model": ("MODEL", {"tooltip": "低噪声阶段使用的模型，在分辨率提升后运行。未连接时使用高噪声阶段模型。两个阶段必须兼容架构、潜变量格式、采样参数化方式、VAE 和条件输入。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift-Avatar"

    def sample(self, low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, high_res_model=None):
        return (progressive_sample(low_res_model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, latent_upsample,
                                   high_res_model=high_res_model),)


class SelfLiftAvatarH3TST:
    """Training-free Temporal State Transport correction (arXiv:2609.08505) for MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tau": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "稳态校正强度，论文设置为 0.2。设为 0 会关闭校正，但仍保留诊断功能。"}),
            "log_diagnostics": ("BOOLEAN", {"default": True, "label_on": "诊断日志：开启", "label_off": "诊断日志：关闭",
                                            "tooltip": "将每次前向计算的 Spectral Tension 和校正统计信息记录到控制台。启用 SelfLift 采样器的低噪声阶段分块时，TST 会发出警告并跳过。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "selflift-Avatar"

    def patch(self, model, tau, log_diagnostics):
        return (h3_tst.patch_model(model, tau, log_diagnostics),)


NODE_CLASS_MAPPINGS = {
    "SelfLiftAvatarH3Sampler": SelfLiftAvatarH3Sampler,
    "SelfLiftAvatarH3SpatialAnchoredSampler": SelfLiftAvatarH3SpatialAnchoredSampler,
    "SelfLiftAvatarImageSampler": SelfLiftAvatarImageSampler,
    "SelfLiftAvatarH3TST": SelfLiftAvatarH3TST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftAvatarH3Sampler": "SelfLift-Avatar｜H3 采样器",
    "SelfLiftAvatarH3SpatialAnchoredSampler": "SelfLift-Avatar｜空间锚定采样器（测试）",
    "SelfLiftAvatarImageSampler": "SelfLift-Avatar｜图像采样器",
    "SelfLiftAvatarH3TST": "SelfLift-Avatar｜H3 时序状态传输校正",
}
