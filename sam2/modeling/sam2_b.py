# model with sam2_img_encoder and mask_decoder just for img seg
from typing import List, Dict, Any

from sam2.modeling.css_module import CCS
from PIL.Image import Image
import numpy as np
import torch
import torch.distributed


from sam2.modeling.sam.mask_decoder_v import MaskDecoder
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
            if_ccs=True,
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
        
        # sstd config
        self.if_ccs = if_ccs
        
        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False
        
        else:
            self.sstd = CCS()

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

    def forward(
            self,
            image
    ):
        """
        'images':numpy HxWx3
        """
        
        
        if isinstance(image, np.ndarray):
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        
        input_image = input_image[None, ...]
        
        assert (
                len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        
        input_image = input_image.to(self.device)
        # ---------image encoder------------------
        backbone_out = self.forward_image(input_image)
        
        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]  # list[(1,32,256,256),(1,64,128,128),(1,256,64,64)]
        high_res_features = feats[:-1]
        image_embed = feats[-1]
        # ------------prompt encoder---------------------

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        # -----------mask decoder-----------------------
        low_res_masks, low_res_vector_field = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output_in_sam,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # Upscale the masks to the original image resolution
        o = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[-1]
        )  # (1,1,H,W)
        vector_field = self._transforms.postprocess_masks(
            low_res_vector_field, self._orig_hw[-1]
        )
       
        
        if self.if_ccs:
            
            # (B,2,H,W)-->(B,H,W,2)
            vector_field = vector_field.permute(0, 2, 3, 1)
            norm = torch.norm(vector_field, dim=-1, keepdim=True)
            vector_field_n = vector_field / (norm + 1e-8)

            
            
            masks = self.sstd(o, vector_field_n.squeeze(0))
            output = {
            'mask': masks,
            'vector_field': vector_field_n.squeeze()#(tensor:HXWx2)
        }
        else:
            masks = o.squeeze()
            output = {
            'mask': masks,
        }
            
        return output
