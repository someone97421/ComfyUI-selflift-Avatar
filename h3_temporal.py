"""仅用于 H3 第二段采样的重叠时间窗口。"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

import comfy.nested_tensor
import comfy.samplers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN

from .avatar_sampling import native_mask_model
from .avatar_masks import restore_kept


def _pixel_frames(video_steps):
    cycles, tail = divmod(video_steps, len(FRAME_PER_TOKEN))
    return cycles * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:tail])


def _video_steps(frame_count):
    if type(frame_count) is not int or frame_count < 5 or (frame_count - 5) % 17:
        raise ValueError("时间窗口和重叠帧数必须采用 H3 的 17k+5 网格，例如 5、22、39、56、124。")
    return (frame_count - 5) // 17 * 5 + 2


def validate_options(video_t, audio_t, chunk_frames, overlap_frames):
    if type(video_t) is not int or video_t < 2 or (video_t - 2) % 5:
        raise ValueError("时间分块要求 H3 视频潜变量的时间长度为 5k+2。")
    if type(audio_t) is not int or audio_t < 1:
        raise ValueError("时间分块要求非空的 H3 音频潜变量。")
    chunk_t, overlap_t = _video_steps(chunk_frames), _video_steps(overlap_frames)
    if overlap_t >= chunk_t:
        raise ValueError("时间重叠帧数必须小于时间窗口帧数。")


def plan_chunks(video_t, audio_t, chunk_frames=124, overlap_frames=22):
    validate_options(video_t, audio_t, chunk_frames, overlap_frames)
    total_frames = _pixel_frames(video_t)
    chunk_t, overlap_t = _video_steps(chunk_frames), _video_steps(overlap_frames)
    plan = []
    end = 0
    while end < video_t:
        start = 0 if not plan else end - overlap_t
        stop = min(start + chunk_t, video_t)
        frame_start, frame_end = _pixel_frames(start), _pixel_frames(stop)
        audio_start = round(frame_start * audio_t / total_frames)
        audio_end = round(frame_end * audio_t / total_frames)
        if audio_end <= audio_start:
            raise ValueError("时间窗口对应的音频为空，请增大时间窗口。")
        plan.append({
            "video_start": start, "video_end": stop,
            "audio_start": audio_start, "audio_end": audio_end,
            "context_video_t": 0 if not plan else end - start,
            "context_audio_t": 0 if not plan else plan[-1]["audio_end"] - audio_start,
            "frame_start": frame_start, "frame_end": frame_end,
        })
        end = stop
    return plan


def validate_conditioning(model, conditioning):
    for item in conditioning or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[1], dict):
            raise ValueError("时间分块收到无效的条件输入。")
        if item[1].get("control") is not None or item[1].get("gligen") is not None:
            raise ValueError("时间分块暂不支持 ControlNet 或 GLIGEN 条件。")
        guides = item[1].get("minimax_keyframes") or []
        if guides and not callable(getattr(model.model, "_window_payload", None)):
            raise ValueError("当前 ComfyUI 缺少 H3 引导的时间窗口裁剪接口，请更新 ComfyUI。")
        for guide in guides:
            if "resolved_frame_index" not in guide:
                raise ValueError("H3 时间引导缺少 resolved_frame_index。")


def _slice_conditioning(model, conditioning, chunk, total_video_t):
    if conditioning is None:
        return None
    output = []
    for item in conditioning:
        values = dict(item[1])
        guides = values.get("minimax_keyframes")
        if guides:
            audio_window = SimpleNamespace(index_list=range(chunk["audio_start"], chunk["audio_end"]))
            window = SimpleNamespace(
                total_frames=total_video_t,
                index_list=range(chunk["video_start"], chunk["video_end"]),
                get_window_for_modality=lambda index: audio_window,
            )
            payload = model.model._window_payload(
                {"keyframes": guides, "refs": values.get("minimax_refs")}, window)
            values["minimax_keyframes"] = payload["keyframes"]
        output.append([item[0], values])
    return output


def _window(length, left, right):
    result = torch.ones(length, dtype=torch.float32)
    if left:
        result[:left] *= (torch.arange(left, dtype=torch.float32) + 0.5) / left
    if right:
        result[-right:] *= 1.0 - (torch.arange(right, dtype=torch.float32) + 0.5) / right
    return result


def _streams(samples):
    if not getattr(samples, "is_nested", False):
        raise ValueError("时间分块要求 H3 的音视频 NestedTensor。")
    streams = list(samples.unbind())
    if len(streams) != 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
        raise ValueError("时间分块要求五维视频和四维音频潜变量。")
    return streams


def _pack(streams):
    return comfy.nested_tensor.NestedTensor(list(streams))


def _slice(stream, index, start, end):
    axis = 2 if index == 0 else stream.ndim - 1
    return stream.narrow(axis, start, end - start).contiguous()


def sample(model, noise, positive, negative, cfg, sampler, sigmas, latent_image,
           seed, chunk_frames=124, overlap_frames=22, seam_lock=True,
           disable_pbar=False, on_plan=None, callback=None, anchors=None, masks=None,
           require_callbacks=False):
    streams = _streams(latent_image)
    noise_streams = _streams(noise)
    video, audio = streams
    if any(tuple(n.shape) != tuple(s.shape) for n, s in zip(noise_streams, streams)):
        raise ValueError("时间分块的噪声与潜变量形状不一致。")
    validate_conditioning(model, positive)
    validate_conditioning(model, negative)
    plan = plan_chunks(video.shape[2], audio.shape[-1], chunk_frames, overlap_frames)
    if (masks is not None or (seam_lock and len(plan) > 1)) and (
            not isinstance(sampler, comfy.samplers.KSAMPLER)
            or type(sampler).sample is not comfy.samplers.KSAMPLER.sample):
        raise ValueError("时间接缝锁定和遮罩要求标准 ComfyUI KSAMPLER 接口。")
    if masks is not None and (anchors is None or len(anchors) != 2):
        raise ValueError("时间遮罩需要对应的干净音视频锚点。")
    if on_plan is not None:
        on_plan({
            "kind": "temporal", "mode": "temporal", "chunks": len(plan),
            "chunk_frames": chunk_frames, "overlap_frames": overlap_frames,
            "seam_lock": bool(seam_lock), "video_shape": tuple(video.shape),
            "audio_shape": tuple(audio.shape), "full_sequence": False,
        })

    outputs = [torch.zeros(s.shape, dtype=torch.float32, device="cpu") for s in streams]
    weights = [torch.zeros(video.shape[2]), torch.zeros(audio.shape[-1])]
    steps = len(sigmas) - 1
    total_steps = steps * len(plan)
    for index, chunk in enumerate(plan):
        ranges = [(chunk["video_start"], chunk["video_end"]),
                  (chunk["audio_start"], chunk["audio_end"])]
        chunk_streams = [_slice(s, i, *r) for i, (s, r) in enumerate(zip(streams, ranges))]
        chunk_noise = _pack([_slice(n, i, *r) for i, (n, r) in enumerate(zip(noise_streams, ranges))])
        chunk_anchors = ([_slice(a, i, *r).clone() for i, (a, r) in enumerate(zip(anchors, ranges))]
                         if masks is not None else [s.clone() for s in chunk_streams])
        chunk_masks = ([ _slice(m.expand_as(s), i, *r).clone()
                        for i, (m, s, r) in enumerate(zip(masks, streams, ranges))]
                       if masks is not None else None)
        if seam_lock and index > 0:
            if chunk_masks is None:
                chunk_masks = [torch.ones_like(s, dtype=torch.float32) for s in chunk_streams]
            # 只锁定开头的一个 H3 时间组（5 个像素帧），让其余重叠区继续去噪。
            lock_t = min(2, chunk["context_video_t"])
            start = chunk["video_start"]
            previous = outputs[0][:, :, start:start + lock_t] / weights[0][start:start + lock_t].view(1, 1, -1, 1, 1)
            lock_mask = chunk_masks[0][:, :, :lock_t]
            chunk_anchors[0][:, :, :lock_t] = torch.where(
                lock_mask.to(chunk_anchors[0].device) == 0,
                chunk_anchors[0][:, :, :lock_t], previous.to(chunk_anchors[0]))
            chunk_masks[0][:, :, :lock_t] = 0
        chunk_model = native_mask_model(model, chunk_anchors, chunk_masks, True)
        chunk_positive = _slice_conditioning(model, positive, chunk, video.shape[2])
        chunk_negative = _slice_conditioning(model, negative, chunk, video.shape[2])
        evaluations = 0

        def report(step, x0, x, total):
            nonlocal evaluations
            evaluations += 1
            local_step = evaluations - 1 if require_callbacks else min(max(int(step), 0), steps - 1)
            if callback is not None:
                callback(index * steps + local_step, x0, x, total_steps)

        logging.info("[selflift-Avatar 时间分块] %d/%d 视频=%s 音频=%s", index + 1, len(plan), ranges[0], ranges[1])
        sampled = comfy.samplers.sample(
            chunk_model, chunk_noise, chunk_positive, chunk_negative, cfg,
            chunk_model.load_device, sampler, sigmas, chunk_model.model_options,
            latent_image=_pack(chunk_streams), callback=report,
            disable_pbar=disable_pbar, seed=seed,
        )
        if require_callbacks and evaluations != steps:
            raise RuntimeError(f"时间窗口 {index + 1} 预期 {steps} 次 Euler 回调，实际 {evaluations} 次。")
        predicted = _streams(sampled)
        if any(tuple(p.shape) != tuple(s.shape) for p, s in zip(predicted, chunk_streams)):
            raise RuntimeError("时间采样器返回的音视频形状与窗口不一致。")
        if chunk_masks is not None:
            predicted = restore_kept(predicted, chunk_anchors, chunk_masks)
        lefts = [chunk["context_video_t"], chunk["context_audio_t"]]
        rights = ([ranges[0][1] - plan[index + 1]["video_start"],
                   ranges[1][1] - plan[index + 1]["audio_start"]]
                  if index + 1 < len(plan) else [0, 0])
        for i, (prediction, (start, end)) in enumerate(zip(predicted, ranges)):
            window = _window(end - start, lefts[i], rights[i])
            shape = [1] * prediction.ndim
            shape[2 if i == 0 else -1] = end - start
            target = outputs[i].narrow(2 if i == 0 else outputs[i].ndim - 1, start, end - start)
            target.add_(prediction.float().cpu() * window.view(shape))
            weights[i][start:end].add_(window)
        del sampled, predicted, prediction, chunk_streams, chunk_anchors, chunk_masks, chunk_noise, chunk_model
    for i, out in enumerate(outputs):
        shape = [1] * out.ndim
        shape[2 if i == 0 else -1] = len(weights[i])
        out.div_(weights[i].view(shape))
    return _pack([out.to(s) for out, s in zip(outputs, streams)])
