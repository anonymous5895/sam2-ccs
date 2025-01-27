import os

import torch.amp
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from build_sam2b import build_sam2base
from torch.utils.data import DataLoader
import logging
import argparse
import torch
import numpy as np
import random
import math
import datetime


join = os.path.join
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from tools.calculate_metric import calculate_dice_batch, shape_loss_batch
from Mydataset import MyDataset
from modified_uent import ModifiedUNetPlusPlus


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SAM2 model.")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/sam2.1_hiera_base_plus.pt"
    )
    parser.add_argument("--model_name", type=str, default="Unet", help="model (Unet, Unetsl or Unetsstd)")
    parser.add_argument("--dataname", type=str, default="WHU")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--field", type=str, default="edge",
                        help="field type (edge or open)")
    # Optimizer parameters
    parser.add_argument(
        "--lam", type=float, default=0.05, help="weight of TD term"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="weight decay (default: 0.01)"
    )
    return parser.parse_args()


def to_device(data, device):
    """Move tensors in a dictionary to device."""
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_device(element, device) for element in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


# set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, optimizer, criterion, device, type='SAMsstd', epoch=None):
    model.train()
    total_loss = 0
    bceloss = criterion(mask, gt2D)
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    cosine_loss = nn.CosineSimilarity(dim=-1)
    for step, batched_input in loop:
        optimizer.zero_grad(set_to_none=True)
        gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
        gt2D = gt2D.to(device=device, dtype=torch.float32)
        gt_star_field = torch.stack([x["field"] for x in batched_input], dim=0)
        gt_star_field = gt_star_field.to(device=device, dtype=torch.float32)
        inputs = to_device(batched_input, device)
        image = torch.stack([x["image"] for x in batched_input], dim=0)
        
        scaler = torch.amp.GradScaler()

        with torch.amp.autocast('cuda'):
            output = model(image)
            mask = torch.stack([y["mask"] for y in output], dim=0)
            star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)
            
            if type == 'SAMsstd':
                reg_loss = 1 - cosine_loss(star_field_pre, gt_star_field).mean()
                loss = 0.9*bceloss + 0.1*reg_loss
            elif type == 'SAM':
                reg_loss = torch.tensor(0)
                loss = bceloss
            elif type == 'SAMSL':
                loss = bceloss
                reg_loss = torch.tensor(0)
                if epoch>=10:
                    reg_loss = shape_loss_batch(mask, gt_star_field)
                    loss +=  0.01*reg_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loop.set_postfix(BCE=bceloss.item(), Reg=reg_loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def val(model, val_loader, criterion, calculatedice, device, type='SAMsstd', epoch=None):
    model.eval()
    total_loss = 0
    total_dice = 0
    cosine_loss = nn.CosineSimilarity(dim=-1)

    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, batched_input in loop:
            gt2D = torch.stack([x["gt"] for x in batched_input], dim=0)
            gt2D = gt2D.to(device=device, dtype=torch.float32)
            gt_star_field = torch.stack([x["field"] for x in batched_input], dim=0)
            gt_star_field = gt_star_field.to(device=device, dtype=torch.float32)
            inputs = to_device(batched_input, device)
            with torch.amp.autocast('cuda'):
                output = model(inputs)
                mask = torch.stack([y["mask"] for y in output], dim=0)
                star_field_pre = torch.stack([y["vector_field"] for y in output], dim=0)

                bceloss = criterion(mask, gt2D)
                if type == 'SAMsstd':
                    star_field_loss = 1 - cosine_loss(star_field_pre, gt_star_field).mean()
                    loss = 0.9*bceloss + 0.1*star_field_loss
                elif type == 'SAM':
                    loss = bceloss
                elif type == 'SAMSL':
                    if epoch>=10:
                        shape_loss = shape_loss_batch(mask, gt_star_field)
                        loss = 0.9*bceloss + 0.01*shape_loss
                    else:
                        loss = bceloss
                dice = calculatedice(mask, gt2D)
                total_loss += loss.item()
                total_dice += dice
    return total_loss / len(val_loader), total_dice / len(val_loader)


def setup_logging(log_file='training.log'):
    logging.basicConfig(level=logging.INFO)

    # create ogger
    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def custom_collate_fn(batch):
    return batch  # return batch list

def Lr_lambda(epoch, warm_up_steps=4, period=4):
    if epoch < warm_up_steps:
        return 0.8 ** (warm_up_steps - epoch)
    else:
        if (epoch - warm_up_steps) < period:
            return (1 + math.cos((epoch - warm_up_steps) * math.pi / period)) / 2
        else:
            return Lr_lambda(epoch - warm_up_steps - period, warm_up_steps=0, period=period * 2)


def main():
    seed = 0
    set_seed(seed)
    CONFIG = {'WHU': (1, 50, 1e-4)}
    args = parse_args()
    batch_size, epochs, lr = CONFIG[args.dataname]

     # 创建记录文件夹
    time_str = datetime.datetime.now().strftime(
        '%Y-%m-%d_%H.%M.%S') # 执行脚本时的时间
    model_save_path = os.path.join(args.model_name + "_runs", args.dataname + time_str)
    os.makedirs(model_save_path, exist_ok=True)

    logger = setup_logging(join(model_save_path, 'training.log'))
    logger.info("Arguments received: %s", args)
    logger.info(f'Using device {args.device}')

    # ----------- TensorBoard --------------------
    writer = SummaryWriter(log_dir=join(model_save_path, 'logs'))
    # ------------dataset-------------------
    train_image_path = join('dataset', args.dataname, 'train', 'images')
    train_field_path = join('dataset', args.dataname, 'train', f'{args.field}_center_field')
    train_mask_path = join('dataset', args.dataname, 'train', 'masks')
    val_image_path = join('dataset', args.dataname, 'val', 'images')
    val_mask_path = join('dataset', args.dataname, 'val', 'masks')
    val_field_path = join('dataset', args.dataname, 'val', f'{args.field}_center_field')
    train_dataset = MyDataset(train_image_path, train_mask_path, train_field_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=custom_collate_fn)

    val_dataset = MyDataset(val_image_path, val_mask_path, val_field_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                            collate_fn=custom_collate_fn)

    # -------------load SAM----------------------
    if args.model_name == 'SAMsstd': if_sstd = True
    else: if_sstd = False
    model = build_sam2base(checkpoint=args.checkpoint, if_sstd=if_sstd, lam=args.lam)

    model.to(args.device)
    # ---------------total number of trainable parameters-------------
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'total number of trainable parameters {trainable_num}')

    # -------------loss function and optimizer---------------
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=args.weight_decay
                                 )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=Lr_lambda)
    # --------------train----------------------------
    BEST_SCORE = 0
    logger.info(
        f"begin training {args.model_name} on {args.dataname} with learning_rate: {lr} for {epochs} epochs, batch_size:{batch_size}")
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, args.device, args.model_name, epoch)
        scheduler.step()
        val_loss, val_score = val(model, val_loader, criterion, calculate_dice_batch, args.device, args.model_name, epoch)

        writer.add_scalar('Training Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        writer.add_scalar('Validation Score', val_score, epoch + 1)
        logger.info(
            f"Epoch [{epoch + 1}/{epochs}]: Train Loss - {train_loss:.4f}, Validation Loss - {val_loss:.4f}, "
            f"Val_score-{val_score:.4f}")
        # save the latest model
        os.makedirs(join(model_save_path, args.dataname), exist_ok=True)

        # sam_lora.save_lora_parameters(join(model_save_path, args.dataname, "lora_rank.pth"))
        torch.save(model.state_dict(), join(model_save_path, args.dataname, args.model_name + "_latest.pth"))
        if val_score > BEST_SCORE:
            BEST_SCORE = val_score
            torch.save(model.state_dict(), join(model_save_path, args.dataname, args.model_name + "_model_best.pth"))
            logger.info(f'BEST {epoch + 1} saved!')
    # close TensorBoard
    writer.close()


if __name__ == "__main__":
    main()
