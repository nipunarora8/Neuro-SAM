import numpy as np
import torch
import os
import wandb
from torch.onnx.symbolic_opset11 import hstack
import torch.nn.functional as F
from .utils.stream_dendrites import DataGeneratorStream
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import time
import argparse
from neuro_sam.utils import get_weights_path

def main():
    parser = argparse.ArgumentParser(description="Train Neuro-SAM Dendrite Segmenter")
    parser.add_argument("--ppn", type=int, required=True, help="Positive Points Number")
    parser.add_argument("--pnn", type=int, required=True, help="Negative Points Number")
    parser.add_argument("--model_name", type=str, required=True, choices=['small', 'base_plus', 'large', 'tiny'], help="SAM2 Model Size")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch Size")
    parser.add_argument("--logger", type=str, default="False", help="Use WandB Logger (True/False)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing .d3set data files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TRAINING_DATA_PATH = os.path.join(args.data_dir, "DeepD3_Training.d3set")
    VALIDATION_DATA_PATH = os.path.join(args.data_dir, "DeepD3_Validation.d3set")

    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")

    positive_points = args.ppn
    negative_points = args.pnn
    batch_size = args.batch_size
    logger = (args.logger.lower() == "true")

    print("Initializing Data Generator...")
    dg_training = DataGeneratorStream(TRAINING_DATA_PATH,
                                    batch_size=batch_size, # Data processed at once, depends on your GPU
                                    target_resolution=0.094, # fixed to 94 nm, can be None for mixed resolution training
                                    min_content=100,
                                    resizing_size = 128,
                                    positive_points = positive_points,
                                    negative_points = negative_points) # images need to have at least 50 segmented px

    dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH,
                                        batch_size=batch_size, 
                                        target_resolution=0.094,
                                        min_content=100, 
                                        augment=False,
                                        shuffle=False,
                                        resizing_size = 128,
                                        positive_points = positive_points,
                                    negative_points = negative_points)

    # Load model
    model_name_map = {
        'large': ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml"),
        'base_plus': ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
        'small': ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"),
        'tiny': ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"),
    }
    
    ckpt_name, model_cfg = model_name_map[args.model_name]
    
    # Try to find weights via util or local defaults
    try:
        sam2_checkpoint = get_weights_path(ckpt_name)
    except:
        # Fallback if not downloadable or found
        sam2_checkpoint = ckpt_name
        print(f"Warning: Could not resolve weight path for {ckpt_name}, assuming local file.")

    print(f"Loading SAM2 model: {args.model_name} from {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device) # load model

    predictor = SAM2ImagePredictor(sam2_model)

    # Set training parameters

    predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
    predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler() # mixed precision


    time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
    ckpt_path = f'results/samv2_{args.model_name}_{time_str}'
    if not os.path.exists(ckpt_path): os.makedirs(ckpt_path)

    if logger:
        wandb.init(
            # set the wandb project where this run will be logged
            project="N-SAMv2",

            # track hyperparameters and run metadata
            config={
            "architecture": "SAMv2",
            "dataset": "DeepD3",
            "model": args.model_name,
            "epochs": 100000,
            "ckpt_path": ckpt_path,
            "image_size": (1,128,128),
            "min_content": 100,
            "positive_points": positive_points,
            "negative_points": negative_points,
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
                try:
                    n = np.random.randint(len(dg_validation))
                    image,mask,input_point, input_label = dg_validation[n]
                    
                except:
                    print('Error in validation batch generation')
                    continue
                if mask.shape[0]==0: continue # ignore empty batches
                predictor.set_image_batch(image) # apply SAM image encoder to the image

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

        return np.array(mean_iou).mean(), np.array(mean_dice).mean(), np.array(mean_loss).mean()
    

    for itr in range(100000):
        epoch_dice = epoch_iou = epoch_loss = []
        with torch.cuda.amp.autocast(): # cast to mix precision
            n = np.random.randint(len(dg_training))
            try:
                image,mask,input_point, input_label = dg_training[n]
            except:
                print('Error in training batch')
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
                print(f'step: {itr}, val_loss: {val_loss}, val_iou: {val_iou}, val_dice_score: {val_dice}')

                torch.save(predictor.model.state_dict(), f"{ckpt_path}/model_{itr}.torch") # save model
    if logger:
        wandb.finish()

if __name__ == "__main__":
    main()
