from __future__ import annotations

import argparse
import logging
import os
import random
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from rivagan import eval_common as ec
from rivagan.noise import Crop, Scale
from rivagan.rivagan import RivaGAN, get_acc
from rivagan.utils import mjpeg, psnr, ssim

logger = logging.getLogger(__name__)
_VAL_MAX_CROP = (360, 480)


def _spatial_rng(video_path: str) -> random.Random:
    seed = zlib.adler32(
        (video_path + "::rivagan_val_crop").encode("utf-8", errors="replace")
    ) & 0xFFFFFFFF
    return random.Random(seed)


def _read_val_clip_tensor(video_path: str, seq_len: int) -> torch.Tensor:
    rng = _spatial_rng(video_path)
    cap = cv2.VideoCapture(video_path)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_idx = ec.deterministic_start_index(video_path, seq_len, nb_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise ValueError("Не удалось прочитать видео: %s" % video_path)
    H, W = int(frame.shape[0]), int(frame.shape[1])
    max_h, max_w = _VAL_MAX_CROP
    dy = min(H, max_h)
    dx = min(W, max_w)
    y0 = rng.randint(0, max(0, H - dy))
    x0 = rng.randint(0, max(0, W - dx))
    frames_np: List[np.ndarray] = []
    for ti in range(seq_len):
        if ti > 0:
            ok, frame = cap.read()
        cur = frame[y0 : y0 + dy, x0 : x0 + dx].astype(np.float32)
        frames_np.append(cur)
    cap.release()
    arr = np.stack(frames_np, axis=0)
    t = torch.from_numpy(arr).permute(3, 0, 1, 2).float()
    return (t / 127.5 - 1.0).unsqueeze(0)


def _deterministic_data_tensor(
    data_dim: int, video_path: str, model_path: str
) -> torch.Tensor:
    seed = (
        (zlib.adler32(video_path.encode("utf-8", errors="replace")) & 0xFFFFFFFF)
        ^ (zlib.adler32(os.path.abspath(model_path).encode("utf-8")) & 0xFFFFFFFF)
        ^ (int(data_dim) << 16)
    ) & 0xFFFFFFFF
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    bits = torch.randint(
        0, 2, (1, data_dim), generator=g, dtype=torch.float32, device="cpu"
    )
    return bits.cuda()


def _dedupe_model_paths(model_paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for p in model_paths:
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            logger.warning("Пропуск отсутствующего файла: %s", ap)
            continue
        key = os.path.normcase(os.path.normpath(ap))
        if key in seen:
            logger.warning("Пропуск дубликата пути: %s", ap)
            continue
        seen.add(key)
        out.append(ap)
    return out


def _load_rivagan_checkpoint(path: str) -> RivaGAN:
    obj = torch.load(path, map_location="cuda", weights_only=False)
    if not isinstance(obj, RivaGAN):
        raise TypeError("expected RivaGAN checkpoint, got %s" % type(obj))
    obj.encoder = obj.encoder.cuda()
    obj.decoder = obj.decoder.cuda()
    obj.adversary = obj.adversary.cuda()
    obj.critic = obj.critic.cuda()
    return obj


def _summary_mean_metrics_by_model(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, List[float]]] = {}
    keys = (
        "test.ssim",
        "test.psnr",
        "test.crop_acc",
        "test.scale_acc",
        "test.mjpeg_acc",
    )
    for r in results:
        if r.get("status") != "ok":
            continue
        mp = str(r.get("model_path", ""))
        acc.setdefault(mp, {k: [] for k in keys})
        for k in keys:
            v = r.get(k)
            if v is not None:
                acc[mp][k].append(float(v))
    return {
        mp: {k: float(np.mean(vs)) if vs else float("nan") for k, vs in d.items()}
        for mp, d in acc.items()
    }


def run_model_metrics(
    model_paths: Sequence[str],
    test_dataset: str,
    output_json: str,
    seq_len: int = 1,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    ec.setup_logging(log_file)
    if not torch.cuda.is_available():
        raise RuntimeError("Нужна CUDA (как в ``rivagan.RivaGAN``).")

    paths_unique = _dedupe_model_paths(model_paths)
    if not paths_unique:
        raise ValueError("Нет валидных путей к моделям после дедупликации")

    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    riva_models: List[Tuple[str, RivaGAN]] = []
    for ap in paths_unique:
        logger.info("Загрузка модели: %s", ap)
        riva_models.append((ap, _load_rivagan_checkpoint(ap)))

    videos = ec.list_videos(test_dataset)
    if not videos:
        raise ValueError(
            "В каталоге не найдено видео (*.avi, *.mp4): %s" % test_dataset
        )
    logger.info("Найдено видео: %d", len(videos))

    config = {
        "test_dataset": os.path.abspath(test_dataset),
        "model_paths": [os.path.abspath(p) for p in paths_unique],
        "seq_len": seq_len,
        "val_max_crop": list(_VAL_MAX_CROP),
    }
    results, start_vi = ec.load_resume_from_output_json(
        output_json, config, videos, ec.configs_equal_model_metrics
    )
    if start_vi >= len(videos):
        logger.info(
            "Все %d видео уже в %s",
            len(videos),
            os.path.abspath(output_json),
        )
        return {
            "config": config,
            "progress": {"videos_completed": len(videos), "videos_total": len(videos)},
            "summary_mean_metrics_by_model": _summary_mean_metrics_by_model(results),
            "results": results,
        }

    crop_layer = Crop()
    scale_layer = Scale()

    def _flush_results(videos_done: int) -> None:
        ec.write_json_atomic(
            output_json,
            {
                "config": config,
                "progress": {
                    "videos_completed": videos_done,
                    "videos_total": len(videos),
                },
                "summary_mean_metrics_by_model": _summary_mean_metrics_by_model(
                    results
                ),
                "results": results,
            },
            tmp_prefix="rivagan_model_metrics_",
        )
        logger.info(
            "Промежуточный JSON записан (%d/%d видео): %s",
            videos_done,
            len(videos),
            os.path.abspath(output_json),
        )

    for vi in range(start_vi, len(videos)):
        video_path = videos[vi]
        try:
            frames_cpu = _read_val_clip_tensor(video_path, seq_len)
        except ValueError as e:
            logger.error("%s — пропуск видео: %s", video_path, e)
            for mp, rg in riva_models:
                results.append(
                    {
                        "model_path": mp,
                        "model_arch": rg.model,
                        "data_dim": int(rg.data_dim),
                        "video": video_path,
                        "seq_len": seq_len,
                        "status": "read_error",
                        "error": str(e),
                        "test.ssim": None,
                        "test.psnr": None,
                        "test.crop_acc": None,
                        "test.scale_acc": None,
                        "test.mjpeg_acc": None,
                    }
                )
            _flush_results(vi + 1)
            continue

        frames = frames_cpu.cuda(non_blocking=True)
        logger.info("Видео [%d/%d] %s", vi + 1, len(videos), video_path)

        for mp, rg in riva_models:
            data_dim = int(rg.data_dim)
            data = _deterministic_data_tensor(data_dim, video_path, mp)
            rg.encoder.eval()
            rg.decoder.eval()
            try:
                with torch.no_grad():
                    wm_frames = rg.encoder(frames, data)
                    wm_crop_data = rg.decoder(mjpeg(crop_layer(wm_frames)))
                    wm_scale_data = rg.decoder(mjpeg(scale_layer(wm_frames)))
                    wm_mjpeg_data = rg.decoder(mjpeg(wm_frames))
                    test_ssim = ssim(
                        frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]
                    ).item()
                    test_psnr = psnr(
                        frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]
                    ).item()
                    test_crop_acc = get_acc(data, wm_crop_data)
                    test_scale_acc = get_acc(data, wm_scale_data)
                    test_mjpeg_acc = get_acc(data, wm_mjpeg_data)
                logger.info(
                    "  %s | SSIM %.4f PSNR %.4f crop %.4f scale %.4f mjpeg %.4f",
                    os.path.basename(mp),
                    test_ssim,
                    test_psnr,
                    test_crop_acc,
                    test_scale_acc,
                    test_mjpeg_acc,
                )
                results.append(
                    {
                        "model_path": mp,
                        "model_arch": rg.model,
                        "data_dim": data_dim,
                        "video": video_path,
                        "seq_len": seq_len,
                        "status": "ok",
                        "test.ssim": test_ssim,
                        "test.psnr": test_psnr,
                        "test.crop_acc": test_crop_acc,
                        "test.scale_acc": test_scale_acc,
                        "test.mjpeg_acc": test_mjpeg_acc,
                    }
                )
            except Exception as e:
                logger.error(
                    "Ошибка для модели=%s видео=%s: %s",
                    mp,
                    video_path,
                    e,
                    exc_info=True,
                )
                results.append(
                    {
                        "model_path": mp,
                        "model_arch": rg.model,
                        "data_dim": data_dim,
                        "video": video_path,
                        "seq_len": seq_len,
                        "status": "error",
                        "error": str(e),
                        "test.ssim": None,
                        "test.psnr": None,
                        "test.crop_acc": None,
                        "test.scale_acc": None,
                        "test.mjpeg_acc": None,
                    }
                )
        _flush_results(vi + 1)

    out_doc = {
        "config": config,
        "progress": {
            "videos_completed": len(videos),
            "videos_total": len(videos),
        },
        "summary_mean_metrics_by_model": _summary_mean_metrics_by_model(results),
        "results": results,
    }
    logger.info("Готово: %s", os.path.abspath(output_json))
    return out_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Метрики RivaGAN на тестовом видео"
    )
    parser.add_argument("--models", type=str, nargs="+", default=[])
    parser.add_argument(
        "--dataset", type=str, default=os.path.join(".", "data", "val")
    )
    parser.add_argument(
        "--output-json", type=str, default="./model_metrics_results.json"
    )
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()
    if not args.models:
        parser.error("Укажите хотя бы один путь: --models path1.pt [path2.pt ...]")
    run_model_metrics(
        model_paths=args.models,
        test_dataset=args.dataset,
        output_json=args.output_json,
        seq_len=args.seq_len,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
