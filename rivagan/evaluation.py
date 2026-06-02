import argparse
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from rivagan import eval_common as ec
from rivagan.watermarking_with_control_bits import (
    RivaGanWithControlBits,
    _model_for_data_dim,
    _torch_load_model,
)

logger = logging.getLogger(__name__)


def _dedupe_model_paths(model_paths: Sequence[str]) -> List[str]:
    out = []
    seen_dim = set()
    for p in model_paths:
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            logger.warning("Пропуск отсутствующего файла: %s", ap)
            continue
        d = int(_torch_load_model(ap).data_dim)
        if d in seen_dim:
            logger.warning("Пропуск %s: уже есть модель с data_dim=%d", ap, d)
            continue
        seen_dim.add(d)
        out.append(ap)
    return out


def run_evaluation(
    model_paths: Sequence[str],
    test_dataset: str,
    output_json: str,
    correctable_error_counts: Sequence[int],
    message_lengths: Sequence[int],
    num_frames: int = 8,
    log_file: Optional[str] = None,
    fps: float = 20.0,
) -> Dict[str, Any]:
    ec.setup_logging(log_file)
    logger.info(
        "Старт оценки: датасет=%s, num_frames=%d, correctable_error_counts=%s, message_lengths=%s",
        test_dataset,
        num_frames,
        tuple(map(int, correctable_error_counts)),
        tuple(map(int, message_lengths)),
    )
    if any(int(x) > 0 for x in correctable_error_counts):
        import galois  # noqa: F401

    correctable_error_counts = [int(x) for x in correctable_error_counts]
    message_lengths = [int(x) for x in message_lengths]
    if any(t < 0 for t in correctable_error_counts):
        raise ValueError("correctable_error_counts must be non-negative")
    if any(L < 1 for L in message_lengths):
        raise ValueError("message_lengths must be positive")

    paths_unique = _dedupe_model_paths(model_paths)
    if not paths_unique:
        raise ValueError("Нет валидных путей к моделям после дедупликации по data_dim")

    wm = RivaGanWithControlBits(paths_unique)
    data_dims = sorted({int(m.data_dim) for m in wm.models})
    logger.info("Загружены модели с data_dim: %s", data_dims)

    videos = ec.list_videos(test_dataset)
    if not videos:
        raise ValueError(
            "В каталоге не найдено видео (*.avi, *.mp4): %s" % test_dataset
        )
    logger.info("Найдено видео для оценки: %d", len(videos))

    config = {
        "test_dataset": os.path.abspath(test_dataset),
        "model_paths": [os.path.abspath(p) for p in paths_unique],
        "num_frames": num_frames,
        "correctable_error_counts": correctable_error_counts,
        "message_lengths": message_lengths,
        "fps": fps,
    }
    results, start_vi = ec.load_resume_from_output_json(
        output_json, config, videos, ec.configs_equal_bit_eval
    )
    if start_vi >= len(videos):
        logger.info(
            "Все %d видео уже есть в %s — повторная оценка не требуется",
            len(videos),
            os.path.abspath(output_json),
        )
        return {
            "config": config,
            "progress": {"videos_completed": len(videos), "videos_total": len(videos)},
            "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
                results
            ),
            "results": results,
        }

    def _flush_results(videos_done: int) -> None:
        ec.write_json_atomic(
            output_json,
            {
                "config": config,
                "progress": {
                    "videos_completed": videos_done,
                    "videos_total": len(videos),
                },
                "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
                    results
                ),
                "results": results,
            },
        )
        logger.info(
            "Промежуточные результаты записаны (%d/%d видео): %s",
            videos_done,
            len(videos),
            os.path.abspath(output_json),
        )

    with tempfile.TemporaryDirectory(prefix="rivagan_eval_") as tmp:
        for vi in range(start_vi, len(videos)):
            video_path = videos[vi]
            clip_in, start_idx, nb_frames = ec.prepare_video_clip(
                video_path, tmp, vi, num_frames, fps
            )
            logger.info(
                "Видео [%d/%d] %s: кадры %d..%d (всего кадров в источнике %d)",
                vi + 1,
                len(videos),
                video_path,
                start_idx,
                start_idx + num_frames - 1,
                nb_frames,
            )
            for data_dim in data_dims:
                for want_t in correctable_error_counts:
                    plan = ec.resolve_coding_plan(data_dim, want_t, video_path)
                    if plan is None:
                        reason = "Нет BCH с t=%d при data_dim=%d" % (want_t, data_dim)
                        logger.warning(
                            "%s — пропуск (видео %s)",
                            reason,
                            os.path.basename(video_path),
                        )
                        for msg_len in message_lengths:
                            results.append(
                                ec.make_skip_record(
                                    video_path,
                                    data_dim,
                                    want_t,
                                    "bch",
                                    msg_len,
                                    reason,
                                    start_idx,
                                    num_frames,
                                )
                            )
                        continue

                    coding = plan["coding"]
                    use_raw = plan["use_raw"]
                    bch_n, bch_k, bch_t = plan["bch_n"], plan["bch_k"], plan["bch_t"]
                    parity_bits = plan["parity_bits"]
                    master = plan["master"]

                    for msg_len in message_lengths:
                        reason = ec.skip_length_reason(
                            msg_len, data_dim, use_raw, bch_k, bch_n
                        )
                        if reason:
                            logger.warning("%s — пропуск", reason)
                            results.append(
                                ec.make_skip_record(
                                    video_path,
                                    data_dim,
                                    want_t,
                                    coding,
                                    msg_len,
                                    reason,
                                    start_idx,
                                    num_frames,
                                    bch_n=bch_n,
                                    bch_k=bch_k,
                                    bch_t=bch_t,
                                    parity_bits=parity_bits,
                                )
                            )
                            continue

                        truth = master[:msg_len]
                        msg_list = truth.tolist()
                        clip_wm = os.path.join(
                            tmp, "v%d_d%d_t%d_L%d.mp4" % (vi, data_dim, want_t, msg_len)
                        )
                        try:
                            if use_raw:
                                wm.encode_with_control_bits(
                                    clip_in,
                                    msg_list,
                                    clip_wm,
                                    data_dim=data_dim,
                                    raw_payload=True,
                                )
                            else:
                                wm.encode_with_control_bits(
                                    clip_in,
                                    msg_list,
                                    clip_wm,
                                    data_dim=data_dim,
                                    bch_n=bch_n,
                                    bch_k=bch_k,
                                    bch_t=bch_t,
                                    raw_payload=False,
                                )
                        except Exception as e:
                            logger.error(
                                "Ошибка encode dim=%d t=%d L=%d видео=%s: %s",
                                data_dim,
                                want_t,
                                msg_len,
                                video_path,
                                e,
                                exc_info=True,
                            )
                            results.append(
                                {
                                    "video": video_path,
                                    "data_dim": data_dim,
                                    "correctable_errors": want_t,
                                    "coding": coding,
                                    "parity_bits": parity_bits,
                                    "message_length": msg_len,
                                    "bch_n": bch_n,
                                    "bch_k": bch_k,
                                    "bch_t": bch_t,
                                    "frame_start": start_idx,
                                    "num_frames": num_frames,
                                    "status": "encode_error",
                                    "error": str(e),
                                    "mean_bit_accuracy": None,
                                    "per_frame_bit_accuracy": None,
                                    "full_recovery_frame_fraction": None,
                                    "full_recovery_frame_count": None,
                                    "majority_vote_success": None,
                                }
                            )
                            continue

                        _model_for_data_dim(wm.models, data_dim).decoder.eval()
                        if use_raw:
                            dec_iter = wm.decode_with_control_bits(
                                clip_wm,
                                data_dim=data_dim,
                                message_bits_len=msg_len,
                                raw_payload=True,
                            )
                        else:
                            dec_iter = wm.decode_with_control_bits(
                                clip_wm,
                                data_dim=data_dim,
                                bch_n=bch_n,
                                bch_k=bch_k,
                                bch_t=bch_t,
                                message_bits_len=msg_len,
                                raw_payload=False,
                            )

                        dec_rows = []
                        for fi, dec in enumerate(dec_iter):
                            dec_rows.append(
                                np.asarray(dec, dtype=np.int64).ravel()[:msg_len]
                            )
                            if fi + 1 >= num_frames:
                                break

                        mean_acc, accs, full_frac, full_count, maj_ok = (
                            ec.aggregate_decode_stats(dec_rows, truth, msg_len)
                        )
                        n_used = len(accs)
                        if use_raw:
                            logger.info(
                                "  dim=%d t=0 raw parity=0 L=%d mean_bit_acc=%.4f "
                                "full_frames=%.2f maj_ok=%s (%d кадров)",
                                data_dim,
                                msg_len,
                                mean_acc,
                                full_frac,
                                maj_ok,
                                n_used,
                            )
                        else:
                            logger.info(
                                "  dim=%d t=%d BCH(%d,%d) parity=%d L=%d mean_bit_acc=%.4f "
                                "full_frames=%.2f maj_ok=%s (%d кадров)",
                                data_dim,
                                want_t,
                                bch_n,
                                bch_k,
                                bch_n - bch_k,
                                msg_len,
                                mean_acc,
                                full_frac,
                                maj_ok,
                                n_used,
                            )
                        results.append(
                            {
                                "video": video_path,
                                "data_dim": data_dim,
                                "correctable_errors": want_t,
                                "coding": coding,
                                "parity_bits": parity_bits,
                                "message_length": msg_len,
                                "bch_n": bch_n,
                                "bch_k": bch_k,
                                "bch_t": bch_t,
                                "frame_start": start_idx,
                                "num_frames": num_frames,
                                "status": "ok",
                                "mean_bit_accuracy": mean_acc,
                                "per_frame_bit_accuracy": accs,
                                "full_recovery_frame_fraction": full_frac,
                                "full_recovery_frame_count": full_count,
                                "majority_vote_success": maj_ok,
                            }
                        )
            _flush_results(vi + 1)

    out_doc = {
        "config": config,
        "progress": {
            "videos_completed": len(videos),
            "videos_total": len(videos),
        },
        "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
            results
        ),
        "results": results,
    }
    logger.info("Оценка завершена, итог в %s", os.path.abspath(output_json))
    return out_doc


def main():
    parser = argparse.ArgumentParser(
        description="Оценка декодирования RivaGanWithControlBits на тестовом наборе видео"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[],
        help="Пути к чекпоинтам .pt (по одному на каждый data_dim)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(".", "data", "val"),
        help="Каталог с тестовыми видео (*.avi, *.mp4), рекурсивно",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="./evaluation_rivagan_results.json",
        help="Путь к JSON с результатами",
    )
    parser.add_argument(
        "--num-frames", type=int, default=8, help="Число кадров на клип"
    )
    parser.add_argument(
        "--correctable-errors",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="t=0: raw payload; t>=1: BCH",
    )
    parser.add_argument(
        "--message-lengths",
        type=int,
        nargs="+",
        default=[4, 8, 16, 24],
        help="Длины префиксов сообщения",
    )
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if not args.models:
        parser.error("Укажите хотя бы один путь: --models path1.pt [path2.pt ...]")
    run_evaluation(
        model_paths=args.models,
        test_dataset=args.dataset,
        output_json=args.output_json,
        correctable_error_counts=args.correctable_errors,
        message_lengths=args.message_lengths,
        num_frames=args.num_frames,
        log_file=args.log_file,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
