from typing import Dict, List

import torch
import torch.nn.functional as F

ACE_GLOBAL_CACHE: Dict[int, Dict[str, torch.Tensor]] = {}
ACE_LAYER_COUNTER: int = 0


def _get_layer_modules(model) -> List:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [getattr(layer, "self_attn", None) for layer in model.model.layers]
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "layers")
    ):
        return [getattr(layer, "self_attn", None) for layer in model.language_model.model.layers]

    modules = []
    for mod in model.modules():
        if all(hasattr(mod, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
            modules.append(mod)
    return modules


def _attach_layer_indices(model) -> List:
    modules = _get_layer_modules(model)
    tagged = []
    for idx, attn in enumerate(modules):
        if attn is None:
            continue
        setattr(attn, "layer_idx", idx)
        tagged.append(attn)
    if not tagged:
        raise AttributeError("No attention modules found to tag.")
    return tagged


def apply_ace_forward_patch(model) -> None:
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except Exception as exc:
        raise RuntimeError(f"Failed to import transformers' LlamaAttention: {exc}")

    _attach_layer_indices(model)

    model._ace_cache = {}
    if not hasattr(model, "_ace_layer_counter"):
        model._ace_layer_counter = 0
    if not hasattr(model, "_ace_collect_mode"):
        model._ace_collect_mode = "full"

    orig_forward = LlamaAttention.forward

    def patched_forward(self, *args, **kwargs):
        out = orig_forward(self, *args, **kwargs)

        attn_output, attn_weights, past_kv = None, None, None
        out_is_tuple = isinstance(out, tuple)
        if out_is_tuple:
            if len(out) >= 3:
                attn_output, attn_weights, past_kv = out[0], out[1], out[2]
            elif len(out) == 2:
                attn_output, past_kv = out[0], out[1]
            elif len(out) == 1:
                attn_output = out[0]
        else:
            attn_output = out

        if isinstance(attn_output, torch.Tensor):
            attn_output_leaf = attn_output.detach()
            attn_output_leaf.requires_grad_(True)
        else:
            attn_output_leaf = attn_output

        layer_idx = getattr(self, "layer_idx", None)
        if layer_idx is None:
            layer_idx = getattr(self, "_ace_runtime_idx", None)

        if layer_idx is None:
            try:
                layer_idx = int(getattr(model, "_ace_layer_counter"))
                setattr(self, "_ace_runtime_idx", layer_idx)
                setattr(model, "_ace_layer_counter", layer_idx + 1)
            except Exception:
                global ACE_LAYER_COUNTER
                layer_idx = ACE_LAYER_COUNTER
                try:
                    setattr(self, "_ace_runtime_idx", layer_idx)
                except Exception:
                    pass
                ACE_LAYER_COUNTER = layer_idx + 1

        if layer_idx is not None:
            record: Dict[str, torch.Tensor] = {}

            if attn_weights is not None:
                try:
                    record["attn"] = attn_weights.detach()
                except Exception:
                    pass

            if past_kv is not None and isinstance(past_kv, tuple) and len(past_kv) >= 2:
                try:
                    record["V"] = past_kv[1].detach()
                except Exception:
                    pass

            if getattr(model, "_ace_collect_mode", "full") == "full":
                try:
                    record["oW"] = self.o_proj.weight.detach()
                except Exception:
                    pass
                if isinstance(attn_output_leaf, torch.Tensor):
                    record["out"] = attn_output_leaf

            if record:
                existing = getattr(model, "_ace_cache", {}).get(layer_idx, {})
                if getattr(model, "_ace_collect_mode", "full") != "full" and existing:
                    merged = dict(existing)
                    merged.update(record)
                    model._ace_cache[layer_idx] = merged
                else:
                    model._ace_cache[layer_idx] = record

                global_cache = ACE_GLOBAL_CACHE.get(layer_idx, {})
                merged_global = dict(global_cache)
                merged_global.update(record)
                ACE_GLOBAL_CACHE[layer_idx] = merged_global

        if out_is_tuple:
            if len(out) >= 3:
                return (attn_output_leaf, attn_weights, past_kv) + tuple(out[3:])
            if len(out) == 2:
                return (attn_output_leaf, past_kv)
            if len(out) == 1:
                return (attn_output_leaf,)
        return attn_output_leaf

    LlamaAttention.forward = patched_forward


def reset_ace_cache(model) -> None:
    if hasattr(model, "_ace_cache"):
        try:
            model._ace_cache.clear()
        except Exception:
            model._ace_cache = {}
    ACE_GLOBAL_CACHE.clear()


def _unpack_attn_forward_output(out, pkv_ref):
    attn_output, attn_weights, past_kv = None, None, None
    if isinstance(out, tuple):
        if len(out) >= 3:
            attn_output = out[0]
            if isinstance(out[1], torch.Tensor):
                attn_weights = out[1]
                past_kv = out[2]
            elif isinstance(out[2], torch.Tensor):
                past_kv = out[1]
                attn_weights = out[2]
            else:
                return out, None, None, None
        elif len(out) == 2:
            attn_output = out[0]
            if isinstance(out[1], torch.Tensor):
                attn_weights = out[1]
                past_kv = pkv_ref
            else:
                past_kv = out[1]
        elif len(out) == 1:
            attn_output = out[0]
    else:
        attn_output = out
    return out, attn_output, attn_weights, past_kv


def _apply_gate_to_last_token(self, out, attn_output, attn_weights, past_kv, g_txt_vec, g_vis_vec=None):
    if attn_weights is None:
        return out

    cfg = getattr(self, "_gate_runtime_cfg", None)
    if cfg is None:
        return out

    vis_start = int(cfg["vis_start"])
    vis_len = int(cfg["vis_len"])

    if attn_weights.dim() == 4:
        attn_last = attn_weights[:, :, -1, :]
    elif attn_weights.dim() == 3:
        attn_last = attn_weights
    else:
        return out

    B, H = attn_last.shape[0], attn_last.shape[1]
    if g_txt_vec.numel() != H:
        return out
    if g_vis_vec is not None and g_vis_vec.numel() != H:
        return out

    V_kv = None
    layer_idx = getattr(self, "layer_idx", None)
    if hasattr(past_kv, "layers") and layer_idx is not None:
        layer = past_kv.layers[layer_idx]
        if hasattr(layer, "values"):
            V_kv = layer.values
        elif hasattr(layer, "value"):
            V_kv = layer.value

    if V_kv is None or not torch.is_tensor(V_kv):
        return out

    try:
        from transformers.models.llama.modeling_llama import repeat_kv
    except Exception:
        return out

    V = repeat_kv(V_kv, self.num_key_value_groups)

    kv_heads = V.shape[1]
    if kv_heads != H:
        if H % kv_heads != 0:
            return out
        V = V.repeat_interleave(H // kv_heads, dim=1)

    if V.shape[2] != attn_last.shape[-1]:
        return out

    k0, k1 = vis_start, vis_start + vis_len
    if k0 < 0 or k1 > attn_last.shape[-1] or k0 >= k1:
        return out

    device_t = attn_last.device
    g_txt = g_txt_vec.to(device_t, dtype=attn_last.dtype).view(1, H, 1)
    g_vis = None
    if g_vis_vec is not None:
        g_vis = g_vis_vec.to(device_t, dtype=attn_last.dtype).view(1, H, 1)

    attn_vis = attn_last[..., k0:k1]
    V_vis = V[:, :, k0:k1, :]

    O_vis = torch.einsum("bhk,bhkd->bhd", attn_vis, V_vis)
    O_all = torch.einsum("bhk,bhkd->bhd", attn_last, V)
    O_txt = O_all - O_vis

    if g_vis is None:
        O_new = O_vis + g_txt * O_txt
    else:
        O_new = g_vis * O_vis + g_txt * O_txt

    head_dim = O_new.shape[-1]
    merged = O_new.reshape(B, 1, H * head_dim)
    projected = F.linear(merged, self.o_proj.weight, getattr(self.o_proj, "bias", None))

    if attn_output is not None and attn_output.dim() == 3 and attn_output.shape[-2] >= 1:
        new_out = attn_output.clone()
        new_out[:, -1:, :] = projected
    else:
        new_out = projected

    if isinstance(out, tuple):
        if len(out) == 2:
            return (new_out, out[1])
        if len(out) >= 3:
            return (new_out,) + out[1:]
    return new_out


def apply_gating_patch(model) -> None:
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except Exception as exc:
        raise RuntimeError(f"Failed to import transformers' LlamaAttention: {exc}")

    _attach_layer_indices(model)
    orig_forward = LlamaAttention.forward

    def patched_forward(self, *args, **kwargs):
        cfg = getattr(model, "_gate_runtime", None)
        layer_idx = getattr(self, "layer_idx", None)
        g_vec = None
        if cfg is not None and layer_idx is not None:
            g_vec = cfg.get("per_layer", {}).get(layer_idx)

        need_gate = cfg is not None and layer_idx is not None and g_vec is not None
        pkv_ref = kwargs.get("past_key_value", kwargs.get("past_key_values"))

        if need_gate:
            kwargs["output_attentions"] = True
            kwargs["use_cache"] = True
            self._gate_runtime_cfg = cfg

        out = orig_forward(self, *args, **kwargs)
        out, attn_output, attn_weights, past_kv = _unpack_attn_forward_output(out, pkv_ref)
        if not need_gate:
            return out
        return _apply_gate_to_last_token(self, out, attn_output, attn_weights, past_kv, g_vec)

    LlamaAttention.forward = patched_forward


def apply_gating_patch_all(model) -> None:
    try:
        from transformers.models.llama.modeling_llama import LlamaAttention
    except Exception as exc:
        raise RuntimeError(f"Failed to import transformers' LlamaAttention: {exc}")

    _attach_layer_indices(model)
    orig_forward = LlamaAttention.forward

    def patched_forward(self, *args, **kwargs):
        cfg = getattr(model, "_gate_runtime", None)
        layer_idx = getattr(self, "layer_idx", None)
        g_txt_vec = None
        g_vis_vec = None
        if cfg is not None and layer_idx is not None:
            g_txt_vec = cfg.get("per_layer", {}).get(layer_idx)
            g_vis_vec = cfg.get("per_layer_vis", {}).get(layer_idx)

        need_gate = (
            cfg is not None
            and layer_idx is not None
            and g_txt_vec is not None
            and g_vis_vec is not None
        )
        pkv_ref = kwargs.get("past_key_value", kwargs.get("past_key_values"))

        if need_gate:
            kwargs["output_attentions"] = True
            kwargs["use_cache"] = True
            self._gate_runtime_cfg = cfg

        out = orig_forward(self, *args, **kwargs)
        out, attn_output, attn_weights, past_kv = _unpack_attn_forward_output(out, pkv_ref)
        if not need_gate:
            return out
        return _apply_gate_to_last_token(
            self,
            out,
            attn_output,
            attn_weights,
            past_kv,
            g_txt_vec,
            g_vis_vec=g_vis_vec,
        )

    LlamaAttention.forward = patched_forward
