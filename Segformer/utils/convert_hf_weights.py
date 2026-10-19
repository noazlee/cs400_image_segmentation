"""
File created by: Noah Lee
Converts the HuggingFace SegFormer-B0 checkpoint (nvidia/segformer-b0-finetuned-ade-512-512)
into our encoder/decoder key names. The ADE20K classifier (150 classes) is dropped.
Download pytorch_model.bin from the HF repo into pretrained/segformer_b0_ade.bin first.
Run from the Segformer folder: python utils/convert_hf_weights.py
"""

import torch

SRC = "pretrained/segformer_b0_ade.bin"
DST = "pretrained/segformer_b0_ade.pth"

# Names inside a transformer block: HF name -> our name
BLOCK_RENAMES = {
    "layer_norm_1": "norm1",
    "layer_norm_2": "norm2",
    "attention.self.query": "attn.q",
    "attention.self.key": "attn.k", # k and v get merged into attn.kv below
    "attention.self.value": "attn.v",
    "attention.self.sr": "attn.sr",
    "attention.self.layer_norm": "attn.norm",
    "attention.output.dense": "attn.proj",
    "mlp.dense1": "mlp.fc1",
    "mlp.dense2": "mlp.fc2",
    "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
}

def convert_key(hf_key):
    """
    One HF key -> our key. HF counts stages from 0, we count from 1 (patch_embed1, block1, ...).
    """
    parts = hf_key.split(".")
    param = parts[-1]  # weight, bias, running_mean, ...

    # segformer.encoder.patch_embeddings.0.proj.weight -> encoder.patch_embed1.proj.weight
    if hf_key.startswith("segformer.encoder.patch_embeddings."):
        stage = int(parts[3]) + 1
        layer = "norm" if parts[4] == "layer_norm" else "proj"
        return f"encoder.patch_embed{stage}.{layer}.{param}"

    # segformer.encoder.block.0.1.mlp.dense1.weight -> encoder.block1.1.mlp.fc1.weight
    if hf_key.startswith("segformer.encoder.block."):
        stage = int(parts[3]) + 1
        block = parts[4]
        layer = ".".join(parts[5:-1])  # e.g. "mlp.dense1"
        return f"encoder.block{stage}.{block}.{BLOCK_RENAMES[layer]}.{param}"

    # segformer.encoder.layer_norm.0.weight -> encoder.norm1.weight
    if hf_key.startswith("segformer.encoder.layer_norm."):
        stage = int(parts[3]) + 1
        return f"encoder.norm{stage}.{param}"

    # decode_head.linear_c.0.proj.weight -> decoder.linear_c1.proj.weight
    if hf_key.startswith("decode_head.linear_c."):
        stage = int(parts[2]) + 1
        return f"decoder.linear_c{stage}.proj.{param}"

    # decode_head.linear_fuse.weight -> decoder.linear_fuse.conv.weight
    if hf_key.startswith("decode_head.linear_fuse."):
        return f"decoder.linear_fuse.conv.{param}"

    # decode_head.batch_norm.weight -> decoder.linear_fuse.bn.weight
    if hf_key.startswith("decode_head.batch_norm."):
        return f"decoder.linear_fuse.bn.{param}"

    raise KeyError(f"unknown key: {hf_key}")


hf = torch.load(SRC, map_location="cpu")

out = {}
for hf_key, tensor in hf.items():
    if hf_key.startswith("decode_head.classifier"):  # ADE20K has 150 classes, we have 4
        continue
    out[convert_key(hf_key)] = tensor

# HF splits our kv linear into separate key and value layers - stack them back
# (k first, then v, to match kv.chunk(2) in EffAttention)
for k_name in [name for name in out if ".attn.k." in name]:
    v_name = k_name.replace(".attn.k.", ".attn.v.")
    kv_name = k_name.replace(".attn.k.", ".attn.kv.")
    out[kv_name] = torch.cat([out.pop(k_name), out.pop(v_name)], dim=0)

torch.save(out, DST)
print(f"saved {len(out)} tensors to {DST}")