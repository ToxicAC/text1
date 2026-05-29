import argparse
from datetime import datetime
import json
import numpy as np
import os
import random
import torch
import torch.nn as nn

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import PromptGazeLLE
from gazelle.backbone import DinoV2Backbone
from gazelle.utils import gazefollow_auc, gazefollow_l2, compute_geo_loss, compute_mcr_loss

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default="dinov2_vitb14",
                    choices=["dinov2_vitb14", "dinov2_vitl14"])
parser.add_argument('--clip_model', type=str, default="ViT-B/32")
parser.add_argument('--data_path', type=str, default='./data/gazefollow')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_prompt_gazefollow')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lambda_geo', type=float, default=0.1,
                    help='weight for geometric constraint loss')
parser.add_argument('--lambda_mcr', type=float, default=0.1,
                    help='weight for cross-modal contrastive regression loss')
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()


def main():
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir)

    # Save config and init local log
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(exp_dir, "train_log.csv")
    log_file = open(log_path, "w")
    log_file.write("epoch,iter,loss_total,loss_heatmap,loss_geo,loss_mcr\n")
    eval_log_path = os.path.join(exp_dir, "eval_log.csv")
    eval_file = open(eval_log_path, "w")
    eval_file.write("epoch,auc,min_l2,avg_l2\n")

    # Build PromptGazeLLE model
    backbone = DinoV2Backbone(args.backbone)
    transform = backbone.get_transform((448, 448))
    model = PromptGazeLLE(backbone, clip_model_name=args.clip_model)
    model.cuda()

    # Freeze backbone and CLIP text encoder
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.clip_text_encoder.parameters():
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, \
                head_indices, head_weights, target_indices, target_weights = batch

            optimizer.zero_grad()

            # Forward: vision branch always runs, text branch runs only in training
            heatmap_preds, inout_preds, vision_tokens, f_text = model(
                images=imgs.cuda(),
                bboxes=[[bbox] for bbox in bboxes],
                target_info={
                    "head_indices": head_indices.cuda(),
                    "head_weights": head_weights.cuda(),
                    "target_indices": target_indices.cuda(),
                    "target_weights": target_weights.cuda(),
                }
            )
            heatmap_preds = torch.stack(heatmap_preds).squeeze(dim=1)

            # Heatmap loss
            loss_heatmap = loss_fn(heatmap_preds, heatmaps.cuda())

            # Geometric constraint loss
            loss_geo = compute_geo_loss(model.anchors.weight)

            # Token-level InfoNCE loss (text teacher with stop-gradient)
            loss_mcr = compute_mcr_loss(vision_tokens, f_text)

            # Total loss
            loss = loss_heatmap + args.lambda_geo * loss_geo + args.lambda_mcr * loss_mcr
            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                log_file.write("{},{},{},{},{},{}\n".format(
                    epoch, cur_iter,
                    round(loss.item(), 6), round(loss_heatmap.item(), 6),
                    round(loss_geo.item(), 6), round(loss_mcr.item(), 6)
                ))
                log_file.flush()
                print("TRAIN EPOCH {}, iter {}/{}, loss_total={}, loss_heatmap={}, loss_geo={}, loss_mcr={}".format(
                    epoch, cur_iter, len(train_dl),
                    round(loss.item(), 4), round(loss_heatmap.item(), 6),
                    round(loss_geo.item(), 6), round(loss_mcr.item(), 6)
                ))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH — pure vision branch only (no text, no ground truth target)
        print("Running evaluation")
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, \
                head_indices, head_weights, target_indices, target_weights = batch

            with torch.no_grad():
                heatmap_preds, inout_preds, _, _ = model(
                    images=imgs.cuda(),
                    bboxes=[[bbox] for bbox in bboxes],
                    target_info=None,  # no ground truth at eval time
                )

            heatmap_preds = torch.stack(heatmap_preds).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        eval_file.write("{},{},{},{}\n".format(
            epoch, round(epoch_auc, 6), round(epoch_min_l2, 6), round(epoch_avg_l2, 6)
        ))
        eval_file.flush()
        print("EVAL EPOCH {}: AUC={}, Min L2={}, Avg L2={}".format(epoch, round(epoch_auc, 4), round(epoch_min_l2, 4), round(epoch_avg_l2, 4)))

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

    log_file.close()
    eval_file.close()
    print("Completed training. Best Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch))
    print("Logs saved to: {} and {}".format(log_path, eval_log_path))

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
