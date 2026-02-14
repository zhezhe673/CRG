import os
import json
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F


def build_model_and_processor(
    model_name: str,
    device: str,
    fourbit: bool = False,
    single_gpu: bool = False,
    verbose: bool = False,
):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs: Dict = {"low_cpu_mem_usage": True}
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import transformers: {e}")

    if device.startswith("cuda"):
        load_kwargs["dtype"] = dtype
        if not single_gpu:
            load_kwargs["device_map"] = "auto"

    if fourbit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            load_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                    "device_map": "auto",
                }
            )
            if verbose:
                pwrite("[Info] 4-bit quantization enabled")
        except Exception as e:
            pwrite(f"[Warn] Failed to enable 4-bit quantization: {e}")

    if verbose:
        pwrite("[Info] Loading Llava model ...")

    model = LlavaForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

    if single_gpu and device.startswith("cuda"):
        try:
            model.to(device)  # type: ignore
            if verbose:
                pwrite(f"[Info] Single-GPU mode: model moved to {device}.")
        except Exception as e:
            pwrite(f"[Warn] Failed to move model to a single GPU: {e}")

    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def build_qwen_model_and_processor(
    model_name: str,
    device: str,
    fourbit: bool = False,
    single_gpu: bool = False,
    verbose: bool = False,
):
    load_kwargs: Dict = {"low_cpu_mem_usage": True}
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor  # noqa: F401

    if device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
        if not single_gpu:
            load_kwargs["device_map"] = "auto"

    load_kwargs["trust_remote_code"] = True

    if fourbit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            load_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                    "device_map": "auto",
                }
            )
            if verbose:
                pwrite("[Info] Qwen 4-bit quantization enabled")
        except Exception as e:
            pwrite(f"[Warn] Failed to enable Qwen 4-bit quantization: {e}")

    if verbose:
        pwrite("[Info] Loading Qwen model ...")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if single_gpu and device.startswith("cuda"):
        try:
            model.to(device)  # type: ignore
            if verbose:
                pwrite(f"[Info] Qwen model moved to {device}.")
        except Exception as e:
            pwrite(f"[Warn] Failed to move Qwen model to a single GPU: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def build_qwen25_model_and_processor(
    model_name: str,
    device: str,
    fourbit: bool = False,
    single_gpu: bool = False,
    verbose: bool = False,
):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    load_kwargs: Dict = {"low_cpu_mem_usage": True}

    if device.startswith("cuda"):
        load_kwargs["torch_dtype"] = "auto"
        if not single_gpu:
            load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    if fourbit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            if verbose:
                pwrite("[Info] Qwen2.5-VL 4-bit quantization enabled")
        except Exception as e:
            pwrite(f"[Warn] Failed to enable Qwen2.5-VL 4-bit quantization: {e}")

    if verbose:
        pwrite(f"[Info] Loading Qwen2.5-VL: {model_name}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    if single_gpu and device.startswith("cuda"):
        try:
            model.to(device)  # type: ignore
            if verbose:
                pwrite(f"[Info] Model moved to {device}")
        except Exception as e:
            pwrite(f"[Warn] Failed to move model to a single GPU: {e}")

    model.eval()
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


@torch.no_grad()
def prefill_llava(model, processor, image, prompt, device):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    past_kv = out.past_key_values

    input_ids = inputs["input_ids"]
    last_prompt_tok = inputs["input_ids"][:, -1:]

    tok = processor.tokenizer
    try:
        image_token_id = tok.convert_tokens_to_ids("<image>")
    except Exception:
        image_token_id = -1

    try:
        any_k = past_kv[0][0]
        if any_k.dim() >= 3:
            cand1 = int(any_k.shape[-2])
            cand2 = int(any_k.shape[-3]) if any_k.dim() >= 4 else cand1
            L_total = max(cand1, cand2)
        else:
            L_total = int(input_ids.shape[1])
    except Exception:
        L_total = int(input_ids.shape[1])

    if image_token_id >= 0:
        pos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).squeeze(-1)
        first_pos = int(pos[0].item()) if pos.numel() > 0 else 0
        num_placeholders = pos.numel()
    else:
        first_pos = 0
        num_placeholders = 1

    vis_len = L_total - (input_ids.shape[1] - num_placeholders)
    vis_start = first_pos
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past_kv, next_token, (L_total, vis_start, vis_len), last_prompt_tok


@torch.no_grad()
def prefill_qwen_vl_chat(model, tokenizer, image_path: str, prompt: str, device: str):
    query = tokenizer.from_list_format(
        [
            {"image": image_path},
            {"text": prompt},
        ]
    )

    inputs = tokenizer(query, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    past_kv = out.past_key_values
    input_ids = inputs["input_ids"]
    last_prompt_tok = input_ids[:, -1:]

    img_st = tokenizer.img_start_id
    img_ed = tokenizer.img_end_id
    st_pos = (input_ids[0] == img_st).nonzero(as_tuple=False).squeeze(-1)
    ed_pos = (input_ids[0] == img_ed).nonzero(as_tuple=False).squeeze(-1)
    if st_pos.numel() == 0 or ed_pos.numel() == 0:
        raise RuntimeError("Could not find the <img>...</img> span in input_ids; the image may not be in the sequence.")

    vis_start = int(st_pos[0].item())
    vis_end = int(ed_pos[0].item())
    vis_len = vis_end - vis_start + 1

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return past_kv, next_token, (int(input_ids.shape[1]), vis_start, vis_len), last_prompt_tok


@torch.no_grad()
def prefill_qwen25_vl_instruct(model, processor, image_path: str, prompt: str, device: str):
    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Missing qwen-vl-utils. Recommended install: pip install qwen-vl-utils[decord]==0.0.8"
        ) from e

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _to_file_uri(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    out = model(**inputs, use_cache=True, output_attentions=False, return_dict=True)
    past_kv = out.past_key_values

    input_ids = inputs["input_ids"]
    last_prompt_tok = input_ids[:, -1:]

    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    vis_start, vis_len = _locate_vision_span(input_ids[0], processor.tokenizer)
    L_total = int(input_ids.shape[1])

    return past_kv, next_token, (L_total, vis_start, vis_len), last_prompt_tok, inputs


def _pick_single_token_id(tok, cands):
    for s in cands:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s
    return None, None


def _to_file_uri(path: str) -> str:
    ap = os.path.abspath(path)
    return ap if ap.startswith(("http://", "https://", "file://")) else f"file://{ap}"


def _locate_vision_span(input_ids_1d: torch.Tensor, tokenizer):
    ids = input_ids_1d.tolist()

    def _tok_id(tok: str):
        try:
            return tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            return None

    vs = _tok_id("<|vision_start|>")
    ve = _tok_id("<|vision_end|>")

    if vs is None or vs < 0:
        vs = getattr(tokenizer, "vision_start_id", None)
    if ve is None or ve < 0:
        ve = getattr(tokenizer, "vision_end_id", None)

    if vs is None:
        vs = getattr(tokenizer, "img_start_id", None)
    if ve is None:
        ve = getattr(tokenizer, "img_end_id", None)

    if vs is None or ve is None:
        raise RuntimeError("Could not find token ids for vision_start/vision_end (tokenizer lacks these special tokens).")

    vs = int(vs)
    ve = int(ve)

    try:
        i_vs = ids.index(vs)
        i_ve = ids.index(ve)
    except ValueError as e:
        raise RuntimeError(f"vision_start/vision_end not found in input_ids: {e}")

    if i_ve < i_vs:
        raise RuntimeError(f"Invalid vision span: start@{i_vs}, end@{i_ve}")

    vis_start = i_vs
    vis_len = i_ve - i_vs + 1
    return int(vis_start), int(vis_len)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def split_heads_by_sign(
    A_vis_row: torch.Tensor,
    A_txt_row: torch.Tensor,
    *,
    include_zero: bool = False,
) -> Dict[str, List[int]]:
    assert A_vis_row.shape == A_txt_row.shape, "A_vis_row and A_txt_row must have the same shape"
    assert A_vis_row.dim() == 1, "expect 1D tensor of shape [H]"

    if include_zero:
        vis_pos = A_vis_row >= 0
        vis_neg = A_vis_row <= 0
        txt_pos = A_txt_row >= 0
        txt_neg = A_txt_row <= 0
    else:
        vis_pos = A_vis_row > 0
        vis_neg = A_vis_row < 0
        txt_pos = A_txt_row > 0
        txt_neg = A_txt_row < 0

    pp = torch.nonzero(vis_pos & txt_pos, as_tuple=False).squeeze(-1)
    nn = torch.nonzero(vis_neg & txt_neg, as_tuple=False).squeeze(-1)
    pn = torch.nonzero(vis_pos & txt_neg, as_tuple=False).squeeze(-1)
    np_ = torch.nonzero(vis_neg & txt_pos, as_tuple=False).squeeze(-1)

    return {
        "pp": pp.tolist(),
        "nn": nn.tolist(),
        "pn": pn.tolist(),
        "np": np_.tolist(),
    }


def compute_g_txt_rank_range(
    r_row: torch.Tensor,
    heads: List[int],
    *,
    g_min: float = 0.5,
    g_max: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-3,
    topk: Optional[int] = None,
) -> Dict[int, float]:
    if len(heads) == 0:
        return {}

    heads_tensor = torch.tensor(heads, dtype=torch.long, device=r_row.device)
    v_all = r_row[heads_tensor].to(torch.float32).flatten()

    N = v_all.numel()
    use_topk = topk is not None and topk > 0 and topk < N

    if use_topk:
        sel_order = torch.argsort(v_all)[:topk]
        v = v_all[sel_order]
        sel_heads_tensor = heads_tensor[sel_order]
    else:
        v = v_all
        sel_heads_tensor = heads_tensor

    n_sel = v.numel()
    if n_sel == 0:
        return {}
    if n_sel == 1:
        return {int(sel_heads_tensor.item()): float(g_min)}

    order = torch.argsort(v)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n_sel, device=v.device)

    s = ranks.to(torch.float32) / (n_sel - 1)
    if gamma != 1.0:
        s = s.pow(gamma)

    s = s.clamp(min=eps, max=1.0 - eps)
    g = g_min + (g_max - g_min) * s

    out: Dict[int, float] = {}
    for head_idx, gv in zip(sel_heads_tensor.tolist(), g.tolist()):
        out[int(head_idx)] = float(gv)
    return out


def compute_g_txt_by_sign(
    r_row: torch.Tensor,
    heads_by_sign: Dict[str, List[int]],
    *,
    mode: str = "both",
    A_gamma: float = 1.0,
    B_gamma: float = 1.0,
    eps: float = 1e-3,
    A_g_min: float = 0.5,
    A_g_max: float = 1.0,
    A_topk: Optional[int] = None,
    B_g_min: float = 0.0,
    B_g_max: float = 0.5,
    B_topk: Optional[int] = None,
) -> Dict[int, float]:
    out: Dict[int, float] = {}

    if mode in ("A", "both"):
        pn_heads = heads_by_sign.get("pn", [])
        out.update(
            compute_g_txt_rank_range(
                r_row,
                pn_heads,
                g_min=A_g_min,
                g_max=A_g_max,
                gamma=A_gamma,
                eps=eps,
                topk=A_topk,
            )
        )

    if mode in ("B", "both"):
        np_heads = heads_by_sign.get("np", [])
        out.update(
            compute_g_txt_rank_range(
                r_row,
                np_heads,
                g_min=B_g_min,
                g_max=B_g_max,
                gamma=B_gamma,
                eps=eps,
                topk=B_topk,
            )
        )

    return out


def _load_pope_index(root_dir: str, include_splits: Optional[List[str]] = None) -> Dict[Tuple[str, str, str], int]:
    idx: Dict[Tuple[str, str, str], int] = {}
    if not root_dir:
        return idx
    try:
        for fn in os.listdir(root_dir):
            if not fn.lower().endswith(".json"):
                continue
            low = fn.lower()
            split = "unknown"
            if "random" in low:
                split = "random"
            elif "popular" in low:
                split = "popular"
            elif "adversarial" in low:
                split = "adversarial"
            if include_splits and split not in include_splits:
                continue
            path = os.path.join(root_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    qid = int(obj.get("question_id"))
                    img = str(obj.get("image"))
                    q = str(obj.get("text"))
                    idx[(split, os.path.basename(img), q)] = qid
    except Exception:
        pass
    return idx


def _tqdm(x, total=None, desc=None):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(x, total=total, desc=desc)
    except Exception:
        return x


def pwrite(msg: str) -> None:
    try:
        from tqdm import tqdm  # type: ignore

        if hasattr(tqdm, "write"):
            tqdm.write(str(msg))  # type: ignore[attr-defined]
            return
    except Exception:
        pass
    print(str(msg))


def set_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_attn_impl(model, impl: str, verbose: bool = False) -> None:
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation(impl)
            if verbose:
                pwrite(f"[Info] attention implementation -> {impl}")
        except Exception as e:
            pwrite(f"[Warn] Failed to set attention implementation to {impl}: {e}")

    def _try_set_cfg(obj) -> bool:
        ok = False
        cfg = getattr(obj, "config", None)
        if cfg is None:
            return False
        for key in ("attn_implementation", "_attn_implementation"):
            if hasattr(cfg, key):
                try:
                    setattr(cfg, key, impl)
                    ok = True
                except Exception:
                    pass
        return ok

    changed = False
    changed = _try_set_cfg(model) or changed
    for sub in (getattr(model, "model", None), getattr(model, "language_model", None)):
        if sub is not None:
            changed = _try_set_cfg(sub) or changed
            sub2 = getattr(sub, "model", None)
            if sub2 is not None:
                changed = _try_set_cfg(sub2) or changed

    if changed and verbose:
        pwrite(f"[Info] config attention implementation -> {impl}")


def f(x: "MyClass") -> "MyClass":
    return x


class MyClass:
    pass


from typing import Dict as _Dict, Tuple as _Tuple, Any as _Any, Optional as _Optional  # noqa: E402
import torch as _torch  # noqa: E402


def _parse_cuda_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":")[1])
        except Exception:
            return 0
    return 0


def calc_r(A_vis: torch.Tensor, A_txt: torch.Tensor) -> Tuple[torch.Tensor, float]:
    abs_vis = A_vis.abs()
    abs_txt = A_txt.abs()
    try:
        eps_val = (abs_txt.median().item()) * 1e-3
    except Exception:
        eps_val = 0.0
    if not (eps_val > 0.0):
        eps_val = 1e-12
    denom = abs_vis + abs_txt + eps_val
    r = abs_vis / denom
    return r.to(torch.float32), float(eps_val)


def _default_single_filename(include_splits: str) -> str:
    name = (include_splits or "").strip()
    low = name.lower()
    if name and ("," not in name) and low in {"random", "popular", "adversarial"}:
        return f"{low}.pt"
    name = name.replace(",", "_") or "all"
    return f"pope_r_{name}.pt"


def resolve_single_file_path(single_file: Optional[str], include_splits: str) -> str:
    if not single_file:
        return _default_single_filename(include_splits)
    sf = os.path.expanduser(single_file)
    if os.path.isdir(sf):
        return os.path.join(sf, _default_single_filename(include_splits))
    if os.path.isabs(sf) or os.path.dirname(sf):
        return sf
    return sf


def compute_ace_with_attns_c2(
    model,
    attn_list: List[torch.Tensor],
    past_kv_new,
    vis_start: int,
    vis_len: int,
    grad_post_map: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_ids = sorted(getattr(model, "_ace_cache", {}).keys())  # type: ignore
    if not layer_ids:
        raise RuntimeError("No layer data captured (_ace_cache is empty).")

    A_vis_list: List[torch.Tensor] = []
    A_txt_list: List[torch.Tensor] = []
    used = 0

    for logical_idx, lid in enumerate(layer_ids):
        rec = model._ace_cache[lid]  # type: ignore

        if logical_idx >= len(attn_list) or logical_idx >= len(past_kv_new):
            continue

        attn_full = attn_list[logical_idx]
        if attn_full is None:
            continue

        attn = attn_full
        if attn.dim() == 4 and attn.shape[2] == 1:
            attn = attn[:, :, 0, :]
        elif attn.dim() != 3:
            continue

        V = past_kv_new[logical_idx][1]
        oW = rec.get("oW", None)

        grad_post = grad_post_map.get(lid, None)
        if oW is None or grad_post is None:
            continue

        B, H = attn.shape[0], attn.shape[1]
        hd = V.shape[-1]
        D = grad_post.shape[-1]

        use_matmul = (oW.dim() == 2 and oW.shape[0] == D and oW.shape[1] == H * hd)

        if use_matmul:
            try:
                grad_pre = torch.matmul(grad_post.to(oW.dtype), oW.T).view(B, 1, H, hd)[:, 0]
            except Exception:
                use_matmul = False

        if not use_matmul:
            if D != H * hd:
                hd2 = D // H
                grad_pre = grad_post.view(B, 1, H, hd2)[:, 0]
                if hd2 != hd:
                    if hd2 > hd:
                        grad_pre = grad_pre[..., :hd]
                    else:
                        pad = torch.zeros(B, H, hd - hd2, device=grad_pre.device, dtype=grad_pre.dtype)
                        grad_pre = torch.cat([grad_pre, pad], dim=-1)
            else:
                grad_pre = grad_post.view(B, 1, H, hd)[:, 0]

        device_t = V.device
        attn = attn.to(device_t)
        grad_pre = grad_pre.to(device_t)

        k0, k1 = int(vis_start), int(vis_start + vis_len)
        attn_vis = attn[..., k0:k1]
        V_vis = V[:, :, k0:k1, :]
        O_vis = torch.einsum("bhk,bhkd->bhd", attn_vis, V_vis)
        O_all = torch.einsum("bhk,bhkd->bhd", attn, V)
        O_txt = O_all - O_vis

        gv = (grad_pre.to(torch.float32) * O_vis.to(torch.float32)).sum(dim=-1)
        gt = (grad_pre.to(torch.float32) * O_txt.to(torch.float32)).sum(dim=-1)

        A_vis_list.append(gv.mean(dim=0).to("cpu"))
        A_txt_list.append(gt.mean(dim=0).to("cpu"))
        used += 1

    if used == 0:
        raise RuntimeError("No valid ACE components obtained (missing attn/V or gradients).")

    return torch.stack(A_vis_list, dim=0), torch.stack(A_txt_list, dim=0)


def compute_ace_from_cache_c2(
    model,
    vis_start: int,
    vis_len: int,
    grad_post_map: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_ids = sorted(getattr(model, "_ace_cache", {}).keys())  # type: ignore

    if not layer_ids:
        for mod_name in [
            "attention_patches",
            "ace_forward_patch",
            "llava.ace_forward_patch",
            "Qwen_VL.ace_forward_patch",
            "Qwen25.ace_forward_patch",
        ]:
            try:
                mod = __import__(mod_name, fromlist=["ACE_GLOBAL_CACHE"])
                ACE_GLOBAL_CACHE = getattr(mod, "ACE_GLOBAL_CACHE")
                layer_ids = sorted(ACE_GLOBAL_CACHE.keys())
                if hasattr(model, "_ace_cache") and isinstance(model._ace_cache, dict):  # type: ignore
                    for lid in layer_ids:
                        model._ace_cache[lid] = dict(ACE_GLOBAL_CACHE[lid])  # type: ignore
                if layer_ids:
                    break
            except Exception:
                continue

    if not layer_ids:
        raise RuntimeError("No layer data captured (_ace_cache is empty).")

    A_vis_list: List[torch.Tensor] = []
    A_txt_list: List[torch.Tensor] = []
    used = 0

    for lid in layer_ids:
        rec = model._ace_cache[lid]  # type: ignore
        attn = rec.get("attn", None)
        V = rec.get("V", None)
        oW = rec.get("oW", None)
        grad_post = grad_post_map.get(lid, None)

        if attn is None or V is None or oW is None or grad_post is None:
            continue

        if attn.dim() == 4 and attn.shape[2] == 1:
            attn = attn[:, :, 0, :]
        if attn.dim() != 3:
            continue
        B, Hq, Lk = attn.shape

        if not (hasattr(V, "dim") and V.dim() == 4):
            continue

        if V.shape[1] == Lk:
            V = V.permute(0, 2, 1, 3).contiguous()
        elif V.shape[2] == Lk:
            V = V.contiguous()
        else:
            continue

        Hkv = V.shape[1]
        hd = V.shape[-1]

        if Hkv != Hq:
            if Hq % Hkv != 0:
                continue
            rep = Hq // Hkv
            V = V.repeat_interleave(rep, dim=1)

        D = grad_post.shape[-1]
        Din = Hq * hd

        use_matmul = (oW.dim() == 2 and oW.shape[0] == D and oW.shape[1] == Din)
        if use_matmul:
            try:
                grad_pre = torch.matmul(grad_post.to(oW.dtype), oW.T).view(B, 1, Hq, hd)[:, 0]
            except Exception:
                use_matmul = False

        if not use_matmul:
            if D == Din:
                grad_pre = grad_post.view(B, 1, Hq, hd)[:, 0]
            else:
                hd2 = D // Hq
                if hd2 <= 0:
                    continue
                grad_pre = grad_post.view(B, 1, Hq, hd2)[:, 0]
                if hd2 != hd:
                    if hd2 > hd:
                        grad_pre = grad_pre[..., :hd]
                    else:
                        pad = torch.zeros(B, Hq, hd - hd2, device=grad_pre.device, dtype=grad_pre.dtype)
                        grad_pre = torch.cat([grad_pre, pad], dim=-1)

        device_t = V.device
        attn = attn.to(device_t)
        grad_pre = grad_pre.to(device_t)

        k0, k1 = int(vis_start), int(vis_start + vis_len)
        attn_vis = attn[..., k0:k1]
        V_vis = V[:, :, k0:k1, :]

        O_vis = torch.einsum("bhk,bhkd->bhd", attn_vis, V_vis)
        O_all = torch.einsum("bhk,bhkd->bhd", attn, V)
        O_txt = O_all - O_vis

        gv = (grad_pre.to(torch.float32) * O_vis.to(torch.float32)).sum(dim=-1)
        gt = (grad_pre.to(torch.float32) * O_txt.to(torch.float32)).sum(dim=-1)

        A_vis_list.append(gv.mean(dim=0).to("cpu"))
        A_txt_list.append(gt.mean(dim=0).to("cpu"))
        used += 1

    if used == 0:
        raise RuntimeError("All layers are missing required data (attn/V/oW or grad_post).")

    return torch.stack(A_vis_list, dim=0), torch.stack(A_txt_list, dim=0)


def compute_ace_from_cache_qwen25_c2(
    model,
    vis_start: int,
    vis_len: int,
    grad_post_map: Dict[int, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_ids = sorted(getattr(model, "_ace_cache", {}).keys())  # type: ignore
    if not layer_ids:
        for mod_name in [
            "attention_patches",
            "ace_forward_patch",
            "llava.ace_forward_patch",
            "Qwen_VL.ace_forward_patch",
            "Qwen25.ace_forward_patch",
        ]:
            try:
                mod = __import__(mod_name, fromlist=["ACE_GLOBAL_CACHE"])
                ACE_GLOBAL_CACHE = getattr(mod, "ACE_GLOBAL_CACHE")
                layer_ids = sorted(ACE_GLOBAL_CACHE.keys())
                if hasattr(model, "_ace_cache") and isinstance(model._ace_cache, dict):  # type: ignore
                    for lid in layer_ids:
                        model._ace_cache[lid] = dict(ACE_GLOBAL_CACHE[lid])  # type: ignore
                if layer_ids:
                    break
            except Exception:
                continue
    if not layer_ids:
        raise RuntimeError("No layer data captured (_ace_cache is empty).")

    def _normalize_V(V: torch.Tensor, Hq: int, Lk: int) -> torch.Tensor:
        if V.dim() != 4:
            raise RuntimeError(f"V dim != 4, got {tuple(V.shape)}")

        if V.shape[1] == Lk:
            V = V.permute(0, 2, 1, 3).contiguous()
        elif V.shape[2] == Lk:
            V = V.contiguous()
        else:
            raise RuntimeError(f"V Lk mismatch: V.shape={tuple(V.shape)}, expected Lk={Lk}")

        Hkv = V.shape[1]
        if Hkv == Hq:
            return V

        if Hq % Hkv == 0:
            rep = Hq // Hkv
            return V.repeat_interleave(rep, dim=1)

        raise RuntimeError(f"GQA head mismatch: Hq={Hq}, Hkv={Hkv}")

    A_vis_list: List[torch.Tensor] = []
    A_txt_list: List[torch.Tensor] = []
    used = 0

    for lid in layer_ids:
        rec = model._ace_cache[lid]  # type: ignore
        attn = rec.get("attn", None)
        V = rec.get("V", None)
        oW = rec.get("oW", None)
        grad_post = grad_post_map.get(int(lid), None)

        if attn is None or V is None or oW is None or grad_post is None:
            continue

        if attn.dim() == 4:
            attn = attn[:, :, -1, :]
        if attn.dim() != 3:
            continue

        B, Hq, Lk = attn.shape

        if grad_post.dim() == 3:
            grad_post = grad_post[:, -1, :]
        if grad_post.dim() != 2:
            continue
        D = grad_post.shape[-1]

        V = _normalize_V(V, Hq=Hq, Lk=Lk)
        hd = V.shape[-1]

        Din = Hq * hd
        use_matmul = (oW.dim() == 2 and oW.shape[0] == D and oW.shape[1] == Din)
        if use_matmul:
            try:
                grad_pre = torch.matmul(grad_post.to(oW.dtype), oW).view(B, Hq, hd)
            except Exception:
                use_matmul = False

        if not use_matmul:
            if D == Din:
                grad_pre = grad_post.view(B, Hq, hd)
            else:
                hd2 = D // Hq
                if hd2 <= 0:
                    continue
                grad_pre = grad_post.view(B, Hq, hd2)
                if hd2 != hd:
                    if hd2 > hd:
                        grad_pre = grad_pre[..., :hd]
                    else:
                        pad = torch.zeros(B, Hq, hd - hd2, device=grad_pre.device, dtype=grad_pre.dtype)
                        grad_pre = torch.cat([grad_pre, pad], dim=-1)

        device_t = V.device
        attn = attn.to(device_t)
        grad_pre = grad_pre.to(device_t)

        k0, k1 = int(vis_start), int(vis_start + vis_len)
        attn_vis = attn[..., k0:k1]
        V_vis = V[:, :, k0:k1, :]

        O_vis = torch.einsum("bhk,bhkd->bhd", attn_vis, V_vis)
        O_all = torch.einsum("bhk,bhkd->bhd", attn, V)
        O_txt = O_all - O_vis

        gv = (grad_pre.to(torch.float32) * O_vis.to(torch.float32)).sum(dim=-1)
        gt = (grad_pre.to(torch.float32) * O_txt.to(torch.float32)).sum(dim=-1)

        A_vis_list.append(gv.mean(dim=0))
        A_txt_list.append(gt.mean(dim=0))
        used += 1

    A_vis = torch.stack(A_vis_list, dim=0).detach().cpu()
    A_txt = torch.stack(A_txt_list, dim=0).detach().cpu()

    if used == 0:
        raise RuntimeError(
            "C2: No layers produced ACE. Possible causes: grad_post_map is empty/too small, "
            "patch did not cache out, or attn/V/oW is missing. "
            f"layers={len(layer_ids)} grad_layers={len(grad_post_map)}"
        )

    return A_vis, A_txt


def _safe_div(numer: float, denom: float) -> float:
    try:
        if denom == 0:
            return 0.0
        return float(numer) / float(denom)
    except Exception:
        return 0.0


def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def evaluate_pope(gt_file: str, pred_file: str) -> Dict[str, float]:
    gt_items = [json.loads(q) for q in open(os.path.expanduser(gt_file), "r", encoding="utf-8")]
    pred_items = [json.loads(q) for q in open(os.path.expanduser(pred_file), "r", encoding="utf-8")]

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    yes_answers = 0
    total_questions = len(gt_items)

    if len(pred_items) != total_questions:
        pwrite(
            f"[Warn] GT count ({total_questions}) != prediction count ({len(pred_items)}); "
            "will evaluate using the shorter length."
        )
        total_questions = min(total_questions, len(pred_items))

    for i in range(total_questions):
        g = gt_items[i]
        p = pred_items[i]
        try:
            assert int(g["question_id"]) == int(p["question_id"])
        except Exception:
            pass
        gt_answer = str(g.get("label", "")).strip().lower()
        gen_answer = str(p.get("text", "")).strip().lower()

        if gt_answer == "yes":
            if "yes" in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == "no":
            if "no" in gen_answer:
                true_neg += 1
            else:
                yes_answers += 1
                false_pos += 1
        else:
            unknown += 1

    precision = _safe_div(true_pos, (true_pos + false_pos))
    recall = _safe_div(true_pos, (true_pos + false_neg))
    f1 = _safe_div(2 * precision * recall, (precision + recall))
    accuracy = _safe_div((true_pos + true_neg), total_questions)
    yes_proportion = _safe_div(yes_answers, total_questions)
    unknown_prop = _safe_div(unknown, total_questions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes": yes_proportion,
        "unknown": unknown_prop,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "unknown_cnt": unknown,
        "total": total_questions,
    }


def evaluate_and_log_pope(
    gt_file: str,
    pred_file: str,
    log_file: str,
    run_params: Optional[Dict] = None,
) -> Dict[str, float]:
    metrics = evaluate_pope(gt_file, pred_file)
    from datetime import datetime

    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gt_file": os.path.expanduser(gt_file),
        "pred_file": os.path.expanduser(pred_file),
        "metrics": metrics,
        "run": run_params or {},
    }
    try:
        _append_jsonl(log_file, rec)
        pwrite(f"[Eval] Appended evaluation results to: {log_file}")
    except Exception as e:
        pwrite(f"[Warn] Failed to write evaluation log: {e}")
    return metrics


def infer_past_len_from_kv(past_kv) -> int:
    k = past_kv[0][0]
    if not isinstance(k, torch.Tensor):
        raise TypeError("past_kv[0][0] is not a Tensor; cannot infer past_len")

    if k.dim() == 4:
        cand1 = int(k.shape[1])
        cand2 = int(k.shape[2])
        return max(cand1, cand2)
    if k.dim() == 3:
        return int(k.shape[1])
    raise ValueError(f"Unsupported KV tensor rank: {k.shape}")


def check_trainable(model):
    n_total = 0
    n_train = 0
    for p in model.parameters():
        n_total += p.numel()
        if p.requires_grad:
            n_train += p.numel()
    print(f"[GradCheck] trainable params: {n_train}/{n_total} ({100*n_train/n_total:.6f}%)")
