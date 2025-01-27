import torch
import numpy as np
from sam2.modeling.sam2_b import SAM2B
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.position_encoding import PositionEmbeddingSine


def build_sam2base(
        checkpoint=None,
        if_ccs=False,
):

    sam2model = SAM2B(
        image_encoder=ImageEncoder(
            scalp=1,
            trunk=Hiera(embed_dim=112, num_heads=2),
            neck=FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                         d_model=256,
                         backbone_channel_list=[896, 448, 224, 112],
                         fpn_top_down_levels=[2, 3],
                         fpn_interp_model='nearest')),
        if_ccs=if_ccs
    )

    sam2model.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        state_dict = state_dict['model']
        # get state_dict for this model
        model_dict = sam2model.state_dict()

        # fliter parameter weight in new model
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        # update state_dict
        model_dict.update(filtered_state_dict)

        # load model
        sam2model.load_state_dict(model_dict)
    return sam2model
