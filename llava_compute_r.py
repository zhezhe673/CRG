#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Set

import torch
from PIL import Image
from tqdm import tqdm

from ace_forward_patch import apply_ace_forward_patch, reset_ace_cache
from pope_dataset import PopeDataset
from vri_utils import (
    build_model_and_processor,
    calc_r,
    check_trainable,
    compute_ace_with_attns_c2,
    prefill_llava,
    pwrite,
    resolve_single_file_path,
    set_attn_impl,
    set_env,
)

VERBOSE = False
YES_ID = 3869
NO_ID = 1939


def trim_past_kv_last(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length") and hasattr(past_key_values, "crop"):
        seq_len = int(past_key_values.get_seq_length())
        if seq_len > 1:
            past_key_values.crop(seq_len - 1)
        return past_key_values
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)}")


def _verbose_flag() -> bool:
    return bool(VERBOSE)


@torch.no_grad()
def compute_var_from_attns(attns, vis_start: int, vis_len: int):
    vars_vis = []
    vars_lift = []
    for a in attns:
        if a is None:
            continue
        a_last = a[0, :, -1, :]
        k_len = a_last.shape[-1]
        v0, v1 = int(vis_start), int(vis_start + vis_len)

        vis_share = a_last[:, v0:v1].sum(dim=-1)
        vars_vis.append(vis_share)

        r0 = float(vis_len) / float(k_len)
        lift = (vis_share - r0) / (1.0 - r0 + 1e-8)
        vars_lift.append(lift)

    var_vis = torch.stack(vars_vis, dim=0)
    var_txt = 1.0 - var_vis
    var_lift = torch.stack(vars_lift, dim=0)
    return var_vis, var_txt, var_lift


def _collect_pope_files(pope_path: str, category: str) -> List[str]:
    if os.path.isdir(pope_path):
        candidates = []
        cat = category.lower()
        for filename in os.listdir(pope_path):
            low = filename.lower()
            if cat in low and low.endswith((".json", ".jsonl")):
                candidates.append(os.path.join(pope_path, filename))
        if not candidates:
            raise FileNotFoundError(
                f"No JSON file containing category '{category}' was found in directory {pope_path}."
            )
        return [sorted(candidates)[0]]
    return [pope_path]


def _build_splits_map(ds: PopeDataset) -> Dict[str, List[str]]:
    tmp_map: Dict[str, Set[str]] = {}
    for ex in ds:
        image_path = ex.get("image_path")
        if not image_path:
            continue
        tmp_map.setdefault(image_path, set()).add(ex.get("split") or "unknown")
    return {k: sorted(list(v)) for k, v in tmp_map.items()}


def _forward_with_grad(model, last_prompt_tok, past_kv_trim, device):
    past_len = int(past_kv_trim.get_seq_length())
    attention_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
    position_ids = torch.tensor([[past_len]], dtype=torch.long, device=device)

    reset_ace_cache(model)
    if hasattr(model, "_ace_collect_mode"):
        model._ace_collect_mode = "full"

    with torch.enable_grad():
        embed = model.get_input_embeddings()
        inputs_embeds = embed(last_prompt_tok.to(device))
        inputs_embeds.requires_grad_(True)

        out = model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_kv_trim,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        logits = out.logits[:, 0, :]
        ell = (logits[:, YES_ID] - logits[:, NO_ID]).mean()

        layer_ids = sorted(getattr(model, "_ace_cache", {}).keys())
        outs = []
        lids = []
        for lid in layer_ids:
            rec = model._ace_cache.get(lid, {})
            if rec.get("oW", None) is None or rec.get("out", None) is None:
                continue
            outs.append(rec["out"])
            lids.append(lid)

        grads_post = torch.autograd.grad(
            outputs=ell,
            inputs=outs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

    grad_post_map = {lid: g for lid, g in zip(lids, grads_post) if g is not None}
    return out, grad_post_map


def run(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    pwrite(f"[Info] Device: {device}")
    model, processor = build_model_and_processor(
        args.model,
        device,
        args.fourbit,
        args.single_gpu,
        verbose=_verbose_flag(),
    )

    apply_ace_forward_patch(model)
    set_attn_impl(model, "eager", verbose=_verbose_flag())

    model.eval()
    model.requires_grad_(False)
    check_trainable(model)

    pope_files = _collect_pope_files(args.pope_path, args.category)
    ds = PopeDataset(root_or_files=pope_files, image_root=args.image_root, include_splits=None)

    total_all = len(ds)
    limit = min(args.max_samples, total_all)
    pwrite(f"[Info] Category={args.category} Total pairs={total_all} Processing this run={limit}")
    samples = [ds[i] for i in range(limit)]

    output_path = resolve_single_file_path(args.output_path, args.category)
    if output_path is not None:
        pwrite(f"[Info] Single-file aggregation will be saved to: {output_path}")

    splits_map = _build_splits_map(ds)
    aggregated: Optional[List[Dict]] = [] if output_path else None

    done, failed = 0, 0
    t0 = time.time()
    total = len(samples)
    for i, sample in enumerate(tqdm(samples, total=total, desc="POPE[qa]"), 1):
        img_path = sample["image_path"]
        base = os.path.basename(img_path)
        question = sample.get("question")
        answer = sample.get("answer")
        split = sample.get("split")

        with Image.open(img_path) as opened:
            image = opened.convert("RGB")
            prompt_text = args.qa_template.format(question=question) if question is not None else args.prompt
            past_kv, _, (_, vis_start, vis_len), last_prompt_tok = prefill_llava(
                model, processor, image, prompt_text, device
            )
            image.close()

        past_kv_trim = trim_past_kv_last(past_kv)
        out, grad_post_map = _forward_with_grad(model, last_prompt_tok, past_kv_trim, device)

        attns = list(out.attentions) if out.attentions is not None else []
        past_kv_new = out.past_key_values
        A_vis, A_txt = compute_ace_with_attns_c2(model, attns, past_kv_new, vis_start, vis_len, grad_post_map)

        r, eps_val = calc_r(A_vis, A_txt)
        var_vis = var_lift = None
        if attns:
            var_vis, _, var_lift = compute_var_from_attns(attns, vis_start, vis_len)

        r_save = r.detach().cpu()
        var_vis_save = var_vis.detach().cpu() if var_vis is not None else None
        var_lift_save = var_lift.detach().cpu() if var_lift is not None else None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        splits = splits_map.get(img_path, [])
        meta = {
            "model": args.model,
            "vis_start": int(vis_start),
            "vis_len": int(vis_len),
            "L": int(r.shape[0]),
            "H": int(r.shape[1]),
            "eps": eps_val,
            "splits": splits,
            "gt_answer": answer,
        }

        try:
            if output_path:
                assert aggregated is not None
                aggregated.append(
                    {
                        "image": base,
                        "image_path": img_path,
                        "question": question,
                        "answer": answer,
                        "split": split,
                        "r": r_save,
                        "A_vis": A_vis.detach().cpu(),
                        "A_txt": A_txt.detach().cpu(),
                        "var_vis": var_vis_save,
                        "var_lift": var_lift_save,
                        "meta": meta,
                    }
                )
                done += 1
                if (done % args.log_every) == 0 or VERBOSE or i <= 3:
                    pwrite(f"[{i}/{total}] Logged r: {base} ; LxH={meta['L']}x{meta['H']}")
            else:
                os.makedirs(args.out_dir, exist_ok=True)
                fname = f"{os.path.splitext(base)[0]}_{i:05d}.r.pt"
                torch.save(
                    {
                        "image": base,
                        "image_path": img_path,
                        "question": question,
                        "answer": answer,
                        "split": split,
                        "r": r_save,
                        "meta": meta,
                    },
                    os.path.join(args.out_dir, fname),
                )
                done += 1
                if (done % args.log_every) == 0 or VERBOSE or i <= 3:
                    pwrite(f"[{i}/{total}] Saved r(QA): {fname} ; LxH={meta['L']}x{meta['H']}")
        except Exception as exc:
            pwrite(f"[{i}/{total}] Save failed: {base} ; {exc}")
            failed += 1

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if output_path:
        try:
            assert aggregated is not None
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torch.save(
                {
                    "model": args.model,
                    "num_items": len(aggregated),
                    "items": aggregated,
                    "params": {
                        "prompt": args.prompt,
                        "fourbit": bool(args.fourbit),
                        "single_gpu": bool(args.single_gpu),
                        "category": args.category,
                        "max_samples": int(args.max_samples),
                    },
                    "note": "items[i] contains image/image_path/r/meta, where r is an [L, H] tensor",
                },
                output_path,
            )
            pwrite(f"\n[Info] Saved to single file: {output_path}")
        except Exception as exc:
            pwrite(f"\n[Warn] Failed to save single file: {exc}")

    dt = time.time() - t0
    pwrite(f"\n[Summary] Done: {done}  Failed: {failed}  Time: {dt / 60:.1f} min  Mode=qa")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.environ.get("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf"))
    parser.add_argument(
        "--pope-path",
        type=str,
        default="./data/POPE/coco",
        help="POPE dataset directory or a single file path",
    )
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument(
        "--category",
        type=str,
        default="popular",
        choices=["random", "popular", "adversarial"],
        help="Process only the specified category (up to --max-samples items)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Aggregated output path (can be a directory or a file; if empty, use category-based default)",
    )
    parser.add_argument("--out-dir", type=str, default="./outputs", help="Used only when --output-path is disabled")
    parser.add_argument("--prompt", type=str, default="<image>\nPlease briefly describe this image in Chinese:")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fourbit", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print more verbose logs")
    parser.add_argument("--log-every", type=int, default=50, help="Print routine logs every N images")
    parser.add_argument("--max-samples", type=int, default=3000, help="Maximum number of POPE items to process")
    parser.add_argument(
        "--qa-template",
        type=str,
        default="<image>\nQuestion: {question}\nAnswer (must be exactly one token): Yes or No\nAnswer:",
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_env()
    args = parse_args()
    VERBOSE = bool(args.verbose)
    run(args)
