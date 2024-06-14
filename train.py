import argparse
import datetime
import os.path as osp
import time
import numpy as np
import torch
import torch.utils.data
from utils.transforms import build_transforms
from datasets import build_test_loader, build_train_loader, build_dataset
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seqnet import SeqNet
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from models.dsbn import convert_dsbn
from torch.cuda.amp import autocast as autocast,GradScaler
from pseudo_label import generate_pseudo_label, generate_binary_part
from feature_selection import select_id_relevant

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--local_rank", default=-1)
    #FLAGS = parser.parse_args()
    local_rank = int(args.local_rank)

    # 新增：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    # 构造模型
    device = torch.device("cuda", local_rank)
    scaler = GradScaler()

    print("Creating model")
    model = SeqNet(cfg)
    convert_dsbn(model.roi_heads.box_head)
    convert_dsbn(model.roi_heads.reid_head)
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        resume_from_ckpt(args.ckpt, model)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    

    print("Loading data")
    transforms = build_transforms(is_train=True)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
    train_loader = build_train_loader(cfg, dataset)
    sur_dataset = build_dataset('CUHK-SYSU', '/dockerdata/path/data/CUHK-SYSU/', transforms, "train", append = True)
    sur_dataset = build_dataset('PRW', '/dockerdata/path/data/PRW/', transforms, "train", append = True)
    train_loader_sur = build_train_loader(cfg, sur_dataset, True)
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
            )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )
    
    #memory = model.module.roi_heads.reid_loss

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, None, None, None) + 1
    #model.to(device)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    #dist.barrier()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        # pseudo label assignment
        if epoch >= 1 and epoch%1==0:
            id_relevant_dims = select_id_relevant(model.module.roi_heads.reid_loss.lut)
            model.module.roi_heads.reid_loss.id_relevant_dims = torch.nn.Parameter(torch.tensor(id_relevant_dims).to(model.module.roi_heads.reid_loss.lut.device), requires_grad=False)
            dist.barrier()
            model.module.roi_heads.reid_loss.cq  = torch.nn.Parameter(torch.zeros(model.module.roi_heads.reid_loss.num_unlabeled, model.module.roi_heads.reid_loss.num_features).to(model.module.roi_heads.reid_loss.lut.device), requires_grad=False)
            transforms = build_transforms(is_train=False)
            #dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train")
            dataset.transforms = transforms
            train_loader = build_train_loader(cfg, dataset)
            dataset = generate_pseudo_label(model, train_loader, device, dataset)
            transforms = build_transforms(is_train=True)
            dataset.transforms = transforms
            
        #concat_dataset = torch.utils.data.ConcatDataset([dataset, sur_dataset])
        #concat_dataset.flag = np.append(dataset.flag, sur_dataset.flag)
        middle_of_frame_idx = generate_binary_part(dataset)
        print('get middle of frame idx')
        model.module.roi_heads.reid_loss.middle_of_frame_idx = middle_of_frame_idx
        train_loader = build_train_loader(cfg, dataset, sur=False, epoch=epoch)
        dist.barrier()
        train_one_epoch(cfg, model, optimizer, train_loader, train_loader_sur, device, epoch, tfboard, scaler)
        lr_scheduler.step()

        if ( (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1 ) and dist.get_rank() == 0:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if ( (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1 ):
            save_on_master(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    parser.add_argument("--local-rank", help="Path to checkpoint to resume or evaluate.")
    args = parser.parse_args()
    main(args)
