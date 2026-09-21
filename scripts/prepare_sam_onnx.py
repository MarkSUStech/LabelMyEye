"""Export MobileSAM to encoder/decoder ONNX for the in-app click segmentation.

Usage:  python scripts/prepare_sam_onnx.py [--checkpoint assets_mobile_sam.pt]

Downloads the MobileSAM checkpoint if missing, then writes:
    assets/models/mobile_sam_encoder.onnx
    assets/models/mobile_sam_decoder.onnx

The SamOnnxModel below reproduces segment_anything/utils/onnx.py
(Apache-2.0, Copyright (c) Meta Platforms, Inc. semantics) with mobile_sam
imports. Pure torch — deliberately avoids numpy so it can run in
environments where torch was built against numpy 1.x.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT_URL = ("https://github.com/ChaoningZhang/MobileSAM/raw/"
                  "master/weights/mobile_sam.pt")
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "assets_mobile_sam.pt")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets", "models")


class SamOnnxModel(nn.Module):
    """Decoder-side model for ONNX export (single best mask output)."""

    def __init__(self, model, return_single_mask: bool = True):
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.prompt_encoder = model.prompt_encoder
        self.img_size = int(model.image_encoder.img_size)
        self.return_single_mask = return_single_mask

    @torch.no_grad()
    def forward(self, image_embeddings, point_coords, point_labels,
                mask_input, has_mask_input):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )
        # Keep ALL mask tokens (0 = no-mask token, 1..3 = multimask scales);
        # token selection with an area prior happens in Python
        # (labelmyeye/sam_seg.SamSegmenter.segment). Single positive point,
        # static shapes; upsampled to image scale in Python after inference —
        # dynamic-size interpolate is not exportable with the dynamo exporter.
        return masks, scores, torch.zeros(1)

    def _embed_points(self, point_coords, point_labels):
        """Labels: -1 pad, 0 negative point, 1 positive point,
        2 box top-left corner, 3 box bottom-right corner — mirroring
        mobile_sam PromptEncoder._embed_points/_embed_boxes."""
        point_coords = point_coords + 0.5
        point_coords = point_coords / float(self.img_size)
        pe = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        pe = torch.where((point_labels == -1.0).unsqueeze(-1), 0.0, pe)
        pen = self.prompt_encoder
        pe = pe + torch.where((point_labels == -1.0).unsqueeze(-1),
                              pen.not_a_point_embed.weight.view(1, 1, -1), 0.0)
        pe = pe + torch.where((point_labels == 0.0).unsqueeze(-1),
                              pen.point_embeddings[0].weight.view(1, 1, -1), 0.0)
        pe = pe + torch.where((point_labels == 1.0).unsqueeze(-1),
                              pen.point_embeddings[1].weight.view(1, 1, -1), 0.0)
        pe = pe + torch.where((point_labels == 2.0).unsqueeze(-1),
                              pen.point_embeddings[2].weight.view(1, 1, -1), 0.0)
        pe = pe + torch.where((point_labels == 3.0).unsqueeze(-1),
                              pen.point_embeddings[3].weight.view(1, 1, -1), 0.0)
        return pe

    def _embed_masks(self, input_mask, has_mask_input):
        mask_embedding = self.prompt_encoder.mask_downscaling(input_mask)
        has_mask = has_mask_input.unsqueeze(0) > 0.5        # bool predicate
        no_mask = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        mask_embedding = mask_embedding + torch.where(has_mask, 0.0, no_mask)
        return mask_embedding


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not os.path.isfile(ckpt):
        print(f"downloading checkpoint -> {ckpt}")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt)

    from mobile_sam import sam_model_registry

    print("building model…")
    sam = sam_model_registry["vit_t"](checkpoint=ckpt)
    sam.eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    enc_path = os.path.join(OUT_DIR, "mobile_sam_encoder.onnx")
    print("exporting encoder…")
    dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.no_grad():
            try:
                torch.onnx.export(
                    sam.image_encoder, (dummy,), enc_path,
                    input_names=["x"], output_names=["image_embeddings"],
                    opset_version=args.opset, do_constant_folding=True,
                    use_external_data_format=False)
            except TypeError:  # older/newer torch without the kwarg
                torch.onnx.export(
                    sam.image_encoder, (dummy,), enc_path,
                    input_names=["x"], output_names=["image_embeddings"],
                    opset_version=args.opset, do_constant_folding=True)

    dec_path = os.path.join(OUT_DIR, "mobile_sam_decoder.onnx")
    print("exporting decoder…")
    onnx_model = SamOnnxModel(sam, return_single_mask=False)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_in = [4 * x for x in embed_size]
    # static 6-slot prompt graph: box mode = [TL(2), BR(3), pad(-1)x4]
    dummy = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.tensor([[[5.0, 5.0], [500.0, 500.0],
                                       [0.0, 0.0], [0.0, 0.0],
                                       [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float),
        "point_labels": torch.tensor([[2.0, 3.0, -1.0, -1.0, -1.0, -1.0]],
                                     dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_in, dtype=torch.float),
        "has_mask_input": torch.tensor([0], dtype=torch.float),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.no_grad():
            torch.onnx.export(
                onnx_model, tuple(dummy.values()), dec_path,
                input_names=list(dummy.keys()),
                output_names=["masks", "iou_predictions", "low_res_masks"],
                opset_version=args.opset, do_constant_folding=True)

    print("done:", enc_path, dec_path)


if __name__ == "__main__":
    main()
