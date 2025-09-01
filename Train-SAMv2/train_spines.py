import numpy as np
import torch
import os
import wandb
from torch.onnx.symbolic_opset11 import hstack
import torch.nn.functional as F
from utils import SpineGeneratorStream
import torch.nn as nn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--logger", type=str, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_DATA_PATH = r"./data/DeepD3_Training.d3set"
VALIDATION_DATA_PATH = r"./data/DeepD3_Validation.d3set"

batch_size = args.batch_size
logger = eval(args.logger)

dg_training = SpineGeneratorStream(TRAINING_DATA_PATH,
                                  batch_size=batch_size, # Data processed at once, depends on your GPU
                                  target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                  min_content=100,
                                  resizing_size = 128) # images need to have at least 50 segmented px

dg_validation = SpineGeneratorStream(VALIDATION_DATA_PATH,
                                    batch_size=batch_size, 
                                    target_resolution=0.094,
                                    min_content=100, 
                                    augment=False,
                                    shuffle=False,
                                    resizing_size = 128)

# Load model
model_name = args.model_name
if model_name == 'large':
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" # path to model weight
    model_cfg = "sam2.1_hiera_l.yaml" #  model config
elif model_name == 'base_plus':
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt" # path to model weight
    model_cfg = "sam2.1_hiera_b+.yaml" #  model config
elif model_name == 'small':
    sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt" # path to model weight
    model_cfg = "sam2.1_hiera_s.yaml" #  model config
elif model_name == 'tiny':
    sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt" # path to model weight
    model_cfg = "sam2.1_hiera_t.yaml" #  model config

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") # load model

predictor = SAM2ImagePredictor(sam2_model)


# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.amp.GradScaler('cuda') # mixed precision


time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
ckpt_path = f'results/samv2_spines_{model_name}_{time_str}'
if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)

if logger:
    wandb.init(
        # set the wandb project where this run will be logged
        project="N-SAMv2 Spines",

        # track hyperparameters and run metadata
        config={
        "architecture": "SAMv2",
        "dataset": "DeepD3",
        "model": model_name,
        "epochs": 100000,
        "ckpt_path": ckpt_path,
        "image_size": (1,128,128),
        "min_content": 100,
        "batch_size": batch_size,
        "prompt_seed":42
        }
    )

# add val code here

def perform_validation(predictor):
    print('Performing Validation')
    mean_iou = []
    mean_dice = []
    mean_loss = []
    with torch.no_grad():
        for i in range(20):
            # n = np.random.randint(len(dg_validation))
            try:
                image,mask,input_point, input_label = dg_validation[n]
            except:
                print('Error')
                continue
            
            # image,mask,input_point, input_label = read_single(val_data,n=i) # load data batch
            # image, mask, input_point, input_label = [image], np.array([mask]), np.array([input_point]), np.array([input_label])
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image_batch(image) # apply SAM image encoder to the image
            # predictor.get_image_embedding()
            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
            prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            total_sum = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1)
            dicee = ((2 * inter) / total_sum).mean()
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

            mean_iou.append(iou.cpu().detach().numpy())
            mean_dice.append(dicee.cpu().detach().numpy())
            mean_loss.append(loss.cpu().detach().numpy())

    print(f'Validation: mean_iou: {np.array(mean_iou).mean()}, mean_dice: {np.array(mean_dice).mean()}, mean_loss: {np.array(mean_loss).mean()}')

    return np.array(mean_iou).mean(), np.array(mean_dice).mean(), np.array(mean_loss).mean()

best_dice = 0

for itr in range(100000):
    epoch_dice = epoch_iou = epoch_loss = []
    with torch.amp.autocast('cuda'): # cast to mix precision
        n = np.random.randint(len(dg_training))

        try:
            image,mask,input_point, input_label = dg_training[n]

        except Exception as e:
            print(f'Error: {e}')
            continue
        if mask.shape[0]==0: continue # ignore empty batches
        predictor.set_image_batch(image) # apply SAM image encoder to the image
        # prompt encoding

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

        # mask decoder

        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

        gt_mask = torch.tensor(mask.astype(np.float32)).to(device)
        prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        total_sum = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1)
        dicee = ((2 * inter) / total_sum).mean()
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss=seg_loss+score_loss*0.05  # mix losses

        predictor.model.zero_grad() # empty gradient 
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update() # Mix precision


        # Display results
        if itr==0: mean_iou=0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        if logger:
            wandb.log({'step': itr, 'loss':loss, 'iou':mean_iou, 'dice_score':dicee})
        print(f'step: {itr}, loss: {loss}, iou: {mean_iou}, dice_score: {dicee}')

        if itr%500 == 0: 
            val_iou, val_dice, val_loss = perform_validation(predictor)
            if logger:
                wandb.log({'step': itr, 'val_loss':val_loss, 'val_iou':val_iou, 'val_dice_score':val_dice})
            if val_dice > best_dice:
                best_dice = val_dice
                print(f'New best dice score: {best_dice} at step {itr}')
                torch.save(predictor.model.state_dict(), f"{ckpt_path}/model_best_dice_{best_dice:.4f}.torch")
                print(f'step: {itr}, val_loss: {val_loss}, val_iou: {val_iou}, val_dice_score: {val_dice}')

if logger:
    wandb.finish()
    
    