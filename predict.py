import os
import torch
import torch.amp

from torchvision.utils import save_image
from build_sam2b import build_sam2base
import cv2
from tools.visual_field import Vis_Field

device = 'cuda' if torch.cuda.is_available() else 'cpu'


checkpoint = 'checkpoint/SAMccs_model.pth'

model = build_sam2base(checkpoint=None, if_ccs=True)

model.load_state_dict(torch.load(checkpoint))
model.to(device)
image_path = 'ISIC_0002107.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

os.makedirs("predict", exist_ok=True)
os.makedirs("predict_field", exist_ok=True)
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        output = model(image)
        pred = output["mask"]
        field = output["vector_field"]
        mask_path = os.path.join('predict', os.path.basename(image_path))
        save_image((pred>0).cpu().float(), mask_path, normalize=True, nrow=1)
        Vis_Field(field.squeeze().cpu().numpy(), os.path.join('predict_field', os.path.basename(image_path)))



