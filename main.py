import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))

import argparse
import torch
import time
from datetime import datetime

from model.FGVONet import FGVONet
from model.utils_FGVONet import FGVONetLoss
from data.kitti_dataset import get_data_loaders
from tqdm import tqdm


def parse_args():

    parser = argparse.ArgumentParser(description='FGDVONet PyTorch Training/Evaluation Script')

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--data_root', type=str, required=True, help='Path to KITTI dataset root directory')
    parser.add_argument('--checkpoint_path', default='', help='Path to saved checkpoint for initialization')
    parser.add_argument('--log_dir', default='log', help='Directory to save results')

    parser.add_argument('--train_list', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('--val_list', nargs='+', type=int, default=[7, 8, 9, 10])


    parser.add_argument('--img_height', type=int, default=192)
    parser.add_argument('--img_width',  type=int, default=640)
    parser.add_argument('--img_backbone', type=str, default='hrnet_w32')
    

    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', choices=['adam', 'momentum'], default='adam')
    parser.add_argument('--decay_step', type=int, default=200000)
    parser.add_argument('--decay_rate', type=float, default=0.7)
    
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("Using device:", device, torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU")
    best_train_loss = float('inf')
    best_val_loss   = float('inf')

    log_dir = args.log_dir + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
    os.makedirs(log_dir, exist_ok=True)


    model = FGVONet(
    img_backbone='hrnet_w32',
    img_backbone_pretrained=True).to(device)

    criterion = FGVONetLoss().to(device)

    opt_params = list(model.parameters()) + list(criterion.parameters())

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=args.learning_rate)
    else:
        optimizer = torch.optim.SGD(opt_params, lr=args.learning_rate,
                                    momentum=args.momentum)
        
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step//args.batch_size, gamma=args.decay_rate)
    
    start_epoch = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        criterion.load_state_dict(ckpt['criterion_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Model restored from {args.checkpoint_path}, starting at epoch {start_epoch}")
    else:
        print("Initialize model")

    train_indices, val_indices, train_loader, val_loader = get_data_loaders(
        data_root   = args.data_root,
        img_height  = args.img_height,
        img_width   = args.img_width,
        train_list  = args.train_list,
        val_list    = args.val_list,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    print(f"Train on sequences {args.train_list}, #samples={len(train_indices)}")
    print(f"Val   on sequences {args.val_list}, #samples={len(val_indices)}")


    for epoch in range(start_epoch, args.max_epoch):
        epoch_start = time.time()

        model.train()
        total_train_loss = 0.0
        total_q_loss = 0.0
        total_t_loss = 0.0
        train_loader_pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
        for i, (img1, img2, q_gt, t_gt, K, K_resize) in enumerate(train_loader_pbar):
            img1 = img1.to(device)
            img2 = img2.to(device)
            q_gt = q_gt.to(device)
            t_gt = t_gt.to(device)

            l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t = model(img1, img2)
            loss, raw_q, raw_t = criterion(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_q_loss += raw_q.item()
            total_t_loss += raw_t.item()

            avg_train = total_train_loss / (i + 1)
            avg_q = total_q_loss / (i + 1)
            avg_t = total_t_loss / (i + 1)
            train_loader_pbar.set_postfix({
                'total': f"{avg_train:.4f}",
                'q_raw': f"{avg_q:.4f}",
                't_raw': f"{avg_t:.4f}"
            })

        scheduler.step()

        model.eval()
        total_val_loss = 0.0
        total_q_loss_val = 0.0
        total_t_loss_val = 0.0
        val_loader_pbar = tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val  ]", leave=False)
        with torch.no_grad():
            for j, (img1, img2, q_gt, t_gt, K, K_resize) in enumerate(val_loader_pbar):
                img1 = img1.to(device)
                img2 = img2.to(device)
                q_gt = q_gt.to(device)
                t_gt = t_gt.to(device)

                l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t = model(img1, img2)
                loss, raw_q, raw_t = criterion(l0_q, l0_t, l1_q, l1_t, l2_q, l2_t, l3_q, l3_t, q_gt, t_gt)
            
                total_val_loss += loss.item()
                total_q_loss_val += raw_q.item()
                total_t_loss_val += raw_t.item()

                avg_val = total_val_loss / (j + 1)
                avg_q_val = total_q_loss_val / (j + 1)
                avg_t_val = total_t_loss_val / (j + 1)

                val_loader_pbar.set_postfix({
                    'total': f"{avg_val:.4f}",
                    'q_raw': f"{avg_q_val:.4f}",
                    't_raw': f"{avg_t_val:.4f}"
                })

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:03d} | Train Loss: {total_train_loss/len(train_loader):.4f} | "
              f"Val Loss: {total_val_loss/len(val_loader):.4f} | Time: {epoch_time:.1f}s")
        
        print(f"  Train - Q: {total_q_loss/len(train_loader):.4f}, T: {total_t_loss/len(train_loader):.4f}")
        print(f"  Val   - Q: {total_q_loss_val/len(val_loader):.4f}, T: {total_t_loss_val/len(val_loader):.4f}")


        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)

        if avg_train < best_train_loss and avg_val < best_val_loss:
            best_train_loss = avg_train
            best_val_loss   = avg_val

            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
                'best_train_loss': best_train_loss,
                'best_val_loss':   best_val_loss
            }
            torch.save(ckpt, os.path.join(log_dir, 'best_ckpt.pth'))
            print(f"→ Saved new best checkpoint (train {avg_train:.4f}, val {avg_val:.4f})")


if __name__ == "__main__":
    main()
