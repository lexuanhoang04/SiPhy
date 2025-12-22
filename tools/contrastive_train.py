import os, sys
sys.path.insert(0, os.getcwd())

import torch
import open_clip
import numpy as np
from tqdm import tqdm
import io, contextlib
import pickle
import json
import datetime
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from arguments import get_args
from predict_property import predict_physical_property_integral
from utils import *
# Make sure to import the new loss function and model from your updated file
from models.mlp_contrastive import MLPContrastive, supervised_contrastive_loss, MaskAwareMLPContrastive, calculate_combined_loss, calculate_combined_loss_entropy
from external.sam2.utils_sam import masks_related_names, get_sam_clip_sim_map, apply_mask_to_image

import torch.nn.functional as F

from external.Pytorch_PCGrad.pcgrad import PCGrad


def get_paths(args, timestamp, frequent=False, epoch=None):
    """Generates paths for model checkpoint and loss curve plot."""
    names_dict = name_creation(args)

    # Add a suffix if using combined losses
    mode_suffix = ""
    if args.combine_losses != "None":
        if args.combine_losses == "l2":
            other_loss_name = "L2"
        elif args.combine_losses == "cross_entropy":
            other_loss_name = "CrossEntropy"
        mode_suffix = f"_combined_{other_loss_name}_lamb_{args.l2_lambda}"

    # Specific suffix for alternate L2 ground truth
    if args.alternate_l2_gt == 'our22':
        mode_suffix += "_our22"
    elif args.alternate_l2_gt == 'blip2':
        mode_suffix += "_blip2"

    # Include start and end index if they exist
    idx_suffix = ""
    if hasattr(args, 'start_idx') and hasattr(args, 'end_idx'):
        idx_suffix = f"_s{args.start_idx}_e{args.end_idx}"

    # Add PCGrad suffix if using it
    pcgrad_suffix = "_pcgrad" if args.use_pcgrad else ""

    # Construct base name
    base_name_common = f"{mode_suffix}{idx_suffix}{pcgrad_suffix}_{args.split}_bs{args.mlp_batch_size}_{args.train_mode}_{timestamp}"
    checkpoint_base_name = f"{args.model}_contrastive{names_dict['source_point_name']}{names_dict['one_mask_name']}{base_name_common}"
    loss_curve_base_name = f"{args.model}{names_dict['source_point_name']}{names_dict['one_mask_name']}{base_name_common}"

    if frequent:
        checkpoint_path = os.path.join("checkpoints", f"{checkpoint_base_name}_epoch_frequent_{epoch}.pth")
        loss_curve_path = f"viz/train_loss_curve/{loss_curve_base_name}_epoch_{epoch}.png"
    else:
        checkpoint_path = os.path.join("checkpoints", f"{checkpoint_base_name}_{epoch}.pth")
        loss_curve_path = f"viz/train_loss_curve/{loss_curve_base_name}_final.png"

    return {
        "checkpoint_path": checkpoint_path,
        "loss_curve_path": loss_curve_path,
    }


def train_contrastive(args):
    #WARMUP_EPOCHS = 30

    names_dict = name_creation(args)

    scenes = get_scenes_list(args)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    if args.model == 'mlp':
        model = MLPContrastive(clip_feat_dim=512).to(args.device)
    elif args.model == 'mlp_attention':
        model = MaskAwareMLPContrastive(clip_feat_dim=512).to(args.device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if args.use_pcgrad:
        print("Using PCGrad optimizer")
        optimizer = PCGrad(optimizer)
    else:
        print("Using standard Adam optimizer")
    # Add a learning rate scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    if args.mlp_checkpoint != "None":
        print(f"🔄 Loading checkpoint from {args.mlp_checkpoint}")
        model.load_state_dict(torch.load(args.mlp_checkpoint, map_location=args.device))
    else:
        print("🚀 Training from scratch")

    model.train()

    # Initialize loss lists based on training mode
    if args.combine_losses:
        total_losses, contrastive_losses_log, other_losses_log = [], [], []
    else:
        losses = []
        
    epochs = args.epochs if hasattr(args, "epochs") else 1
    # Generate one timestamp for the entire training run for consistent filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    f = io.StringIO()
    good_scenes_json = {'good_train': []}
    bad_scenes = []
    bad_count = 0

    # assume args.mlp_batch_size has been added in arguments.py

    if args.verbose:
        print('lambda', args.l2_lambda)
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Reset per‐epoch accumulators
        epoch_total_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_other_loss = 0.0

        # Zero gradients before accumulating
        optimizer.zero_grad()

        for scene_i, scene_name in enumerate(tqdm(scenes, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            #log_paths(args.log_file, [f"[{epoch+1}/{epochs}] Processing scene: {scene_name}"])
            scene_dir = os.path.join(args.data_dir, "scenes", scene_name)
            mask_result = masks_related_names(scene_dir, args)
            
            mask_dir = mask_result["masks_dir"]
            jsons_dir = mask_result["jsons_dir"]
            
            orig_img = Image.open(os.path.join(scene_dir, "images", f"{scene_name}_00.png")).convert("RGB")
            img_np = np.array(orig_img)

            # Fetch features silently
            with contextlib.redirect_stdout(f):
                _, result, _, _ = predict_physical_property_integral(
                    args, scene_dir, clip_model, clip_tokenizer, preprocess
                )
            image_feat = torch.tensor(result["image_features"], device=args.device)
            text_feat  = torch.tensor(result["text_features"],  device=args.device)

            # log_paths(args.log_file, [
            #     f"[feat norms] image: {image_feat.norm(dim=1).mean().item():.2f}, "
            #     f"text: {text_feat.norm(dim=1).mean().item():.2f}"
            # ])
            source_coords = np.array(result["source_pts"]).astype(np.int32)
            mask_ids = build_mask_ids(source_coords, mask_dir, one_mask=args.one_mask)
            
            if args.train_mode == "contrastive":
                # --- Loss Calculation ---
                if args.combine_losses == "None":
                    # Standard contrastive loss
                    logits = (
                        model(image_feat, text_feat, mask_ids)
                        if args.model == "mlp_attention"
                        else model(image_feat, text_feat)
                    )
                    loss = supervised_contrastive_loss(logits, mask_ids, args=args)
                    epoch_total_loss += loss.item()
                else:
                    # Combined contrastive + L2
                    sim_mlp = (
                        model(image_feat, text_feat, mask_ids)
                        if args.model == "mlp_attention"
                        else model(image_feat, text_feat)
                    )
                    if args.alternate_l2_gt == 'None':
                        with torch.no_grad():
                            sim_dot = image_feat @ text_feat.T
                    elif args.alternate_l2_gt == 'our22':
                        #print('hahahahahah')
                        _, sim_clip = get_sam_clip_sim_map(
                            mask_dir=mask_dir,
                            json_dir=jsons_dir,
                            img_np=img_np,
                            clip_model=clip_model,
                            preprocess=preprocess,
                            text_features=text_feat.cpu(),
                            mat_names=result["mat_names"],
                            device=args.device
                        )

                        sim_dot = expand_sim_map(
                            sim_clip, mask_ids, args.one_mask
                        )
                    
                    elif args.alternate_l2_gt == 'blip2':
                        sim_blip2 = load_sim_blip2(
                            scene_dir, args)
                        
                        sim_blip2 = sim_blip2.to(args.device)

                        sim_dot = expand_sim_map(
                            sim_blip2, mask_ids, args.one_mask
                        )
                        
                        sim_dot = sim_dot.to(args.device)

                    if args.combine_losses == "l2":
                        total_loss, c_loss, other_loss = calculate_combined_loss(
                            sim_mlp, sim_dot, mask_ids, l2_lambda=args.l2_lambda, args=args
                        )
                    elif args.combine_losses == "cross_entropy":
                        total_loss, c_loss, other_loss = calculate_combined_loss_entropy(
                            sim_mlp, sim_dot, mask_ids, ce_lambda=args.l2_lambda, args=args
                        )

                    loss = total_loss
                    epoch_total_loss       += total_loss.item()
                    epoch_contrastive_loss += c_loss.item()
                    epoch_other_loss          += other_loss.item()

                if args.use_two_step:
                    # ─── Two‐step manual update ───
                    # 1) big step on contrastive
                    grads_c = torch.autograd.grad(
                        c_loss, model.parameters(), retain_graph=True
                    )
                    
                    norm_c  = torch.sqrt(sum((g.detach()**2).sum() for g in grads_c))
                    print(f"[DEBUG] contrastive grad norm = {norm_c:.4f}")

                    with torch.no_grad():
                        for p, g in zip(model.parameters(), grads_c):
                            p.data -= args.lr * g

                    # 2) tiny step on L2
                    grads_l2 = torch.autograd.grad(other_loss, model.parameters())

                    norm_l2  = torch.sqrt(sum((g.detach()**2).sum() for g in grads_l2))
                    print(f"[DEBUG] L2 grad norm = {norm_l2:.4f}")
                    with torch.no_grad():
                        for p, g in zip(model.parameters(), grads_l2):
                            p.data -= args.lr2 * g

                    # skip the normal backward/step for this scene
                    continue
                
            elif args.train_mode == "l2_alignment":
                # Only L2 loss; freeze all but final MLP layer
                if epoch == 0:
                    #print("🔒 Freezing all model weights except final MLP layer...")
                    for p in model.parameters():
                        p.requires_grad = False
                    for p in model.mlp[-1].parameters():
                        p.requires_grad = True

                sim_mlp = (
                    model(image_feat, text_feat, mask_ids)
                    if args.model == "mlp_attention"
                    else model(image_feat, text_feat)
                )
                with torch.no_grad():
                    sim_dot = image_feat @ text_feat.T

                total_loss = F.mse_loss(sim_mlp, sim_dot)
                c_loss = torch.tensor(0.0, device=args.device)
                other_loss = total_loss
                loss = total_loss
                epoch_total_loss += total_loss.item()
                
            # Skip if loss is invalid
            if not torch.isfinite(loss):
                print(f"[❌] NaN loss in scene {scene_name}, skipping.")
                continue

            # --- Gradient accumulation over mlp_batch_size scenes ---
            (loss / args.mlp_batch_size).backward()

            # Every mlp_batch_size scenes (or at the end), do optimizer step
            if (scene_i + 1) % args.mlp_batch_size == 0 or (scene_i == len(scenes) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # --- End of Epoch Logging ---
        if args.combine_losses != "None":
            avg_total = epoch_total_loss       / len(scenes)
            avg_contrast = epoch_contrastive_loss / len(scenes)
            avg_other    = epoch_other_loss          / len(scenes)
            total_losses.append(avg_total)
            contrastive_losses_log.append(avg_contrast)
            other_losses_log.append(avg_other)
            print(
                f"Epoch {epoch+1} Avg Losses → "
                f"Total: {avg_total:.4f}, "
                f"Contrastive: {avg_contrast:.4f}, "
                f"Scaled other loss: {avg_other:.4f}"
            )
        else:
            avg_loss = epoch_total_loss / len(scenes)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # --- Checkpoint (include mlp_batch_size in filename) ---
        if (epoch + 1) % 1 == 0:
            paths = get_paths(args, timestamp, frequent=False, epoch=epoch+1)
            ckpt = paths["checkpoint_path"]
            #ckpt = ckpt.replace(".pth", f"_{args.model}_bs{args.mlp_batch_size}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint to {ckpt}")
            log_paths(args.log_file, [f"Saved checkpoint to {ckpt}"])

        # --- Loss Curve Plot (every 10 epochs) ---
        if (epoch + 1) % 10 == 0:
            os.makedirs("viz/train_loss_curve", exist_ok=True)
            plt.figure(figsize=(10, 6))
            if args.combine_losses:
                plt.plot(range(1, epoch + 2), total_losses,      marker='o', label='Total Loss')
                plt.plot(range(1, epoch + 2), contrastive_losses_log, marker='o', label='Contrastive Loss')
                plt.plot(range(1, epoch + 2), other_losses_log,     marker='o', label='Scaled Other Loss')
                plt.legend()
            else:
                plt.plot(range(1, epoch + 2), losses, marker='o')
            plt.title("Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            paths = get_paths(args, timestamp, frequent=True, epoch=epoch+1)
            plot_path = paths["loss_curve_path"]
            #plot_path = get_loss_curve_path(args, timestamp, frequent=True, epoch=epoch+1)
            #plot_path = plot_path.replace(".png", f"_mlpbs{args.mlp_batch_size}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"📉 Saved intermediate loss curve to {plot_path}")
            log_paths(args.log_file, [f"Saved intermediate loss curve to {plot_path}"])

    # --- Final Model and Plot Saving ---
    total_epochs_trained = epochs # Simple calculation for this run
    paths = get_paths(args, timestamp, frequent=False, epoch=total_epochs_trained)
    final_model_path = paths["checkpoint_path"]
    
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Saved final trained model to {final_model_path}")
    log_paths(args.log_file, [f"Saved final trained model to {final_model_path}"])

    # Save final loss curve
    os.makedirs("viz/train_loss_curve", exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    if args.combine_losses:
        plt.plot(range(1, epochs + 1), total_losses, marker='o', label='Total Loss')
        plt.plot(range(1, epochs + 1), contrastive_losses_log, marker='o', label='Contrastive Loss')
        plt.plot(range(1, epochs + 1), other_losses_log, marker='o', label=f'Scaled L2 Loss')
        plt.title(f"Final Training Loss Curve (L2 λ = {args.l2_lambda})")
        plt.ylabel("Loss")
        plt.legend()
    else:
        plt.plot(range(1, epochs + 1), losses, marker='o')
        plt.title("Final Training Loss Curve")
        plt.ylabel("Contrastive Loss")

    plt.xlabel("Epoch")
    plt.grid(True)
    paths = get_paths(args, timestamp, frequent=False, epoch=total_epochs_trained)
    final_plot_path = paths["loss_curve_path"]

    plt.savefig(final_plot_path)
    plt.close()
    print(f"📉 Saved final loss curve to {final_plot_path}")
    log_paths(args.log_file, [f"Saved final loss curve to {final_plot_path}"])

if __name__ == "__main__":
    args = get_args()
    # Add a safety check for the new argument
    if not hasattr(args, 'combine_losses'):
        args.combine_losses = False
    if not hasattr(args, 'l2_lambda'):
        args.l2_lambda = 100.0 # Default value if not specified
        
    train_contrastive(args)
