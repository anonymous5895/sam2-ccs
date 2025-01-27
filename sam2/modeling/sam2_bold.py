# model with sam2_img_encoder and mask_decoder just for img seg
import logging

import matplotlib.pyplot as plt
from multisstd import SSTD
from PIL.Image import Image
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.utils.transforms import SAM2Transforms


class SAM2B(torch.nn.Module):
    def __init__(
            self,
            image_encoder,
            image_size=1024,
            backbone_stride=16,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=False,
            if_sstd=False
    ):
        super().__init__()
        self._features = None
        self._orig_hw = None
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.multimask_output_in_sam = multimask_output_in_sam
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        # Part 0: the preprocess transformer
        self._transforms = SAM2Transforms(
            resolution=self.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.hidden_dim = image_encoder.neck.d_model
        # Part 2:SAM-style mask decoder for the final mask output

        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.hidden_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=False,
            pred_obj_scores_mlp=False,
            use_multimask_token_for_obj_ptr=False,
        )
        self.if_sstd = if_sstd
        self.sstd = SSTD()
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_image(self, input_image: torch.Tensor):
        backbone_out = self.image_encoder(input_image)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def forward(self, image, point_inputs):
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
                len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"

        # ---------image encoder------------------
        backbone_out = self.forward_image(input_image)
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]  # list[(1,32,256,256),(1,64,128,128),(1,256,64,64)]
        # self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        high_res_features = feats[:-1]
        image_embed = feats[-1]

        # ------------prompt encoder---------------------
        if point_inputs is not None:
            point_coords = torch.as_tensor(
                point_inputs["point_coords"], dtype=torch.float, device=self.device
            )  # point_inputs["point_coords"] is np with shape (1,2)
            sam_point_coords = self._transforms.transform_coords(
                point_coords, normalize=True, orig_hw=self._orig_hw[-1]
            )
            sam_point_labels = torch.as_tensor(point_inputs["point_labels"], dtype=torch.int, device=self.device)
            if len(sam_point_coords.shape) == 2:
                sam_point_coords, sam_point_labels = sam_point_coords[None, ...], sam_point_labels[None, ...]

        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(1, 1, 2, device=self.device)
            sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=self.device)

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )
        # -----------mask decoder-----------------------
        low_res_masks, iou_predictions, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output_in_sam,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[-1]
        ) #(1,1,H,W)
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        return masks, iou_predictions, low_res_masks


if __name__ == '__main__':
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from PIL import Image

    image_encoder = ImageEncoder(scalp=1,
                                 trunk=Hiera(embed_dim=112, num_heads=2),
                                 neck=FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                                              d_model=256,
                                              backbone_channel_list=[896, 448, 224, 112],
                                              fpn_top_down_levels=[2, 3],
                                              fpn_interp_model='nearest'))
    sam2model = SAM2B(image_encoder)
    checkpoint = r'D:\postgraduate\SAM2(copy)\checkpoints\sam2.1_hiera_base_plus.pt'
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
    sam2model.load_state_dict(state_dict, strict=False)

    image = Image.open('../../images/dog.jpg')
    image = np.array(image.convert('RGB'))
    point_inputs = {'point_coords': np.array([[200, 155]]), "point_labels": np.array([1])}
    masks, iou_predictions, low_res_masks = sam2model(image, point_inputs)
    print(masks.shape)
    print(iou_predictions)
    masks = masks > 0
    masks_np = masks.squeeze().float().detach().cpu().numpy()

