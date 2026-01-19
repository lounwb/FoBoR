import argparse

import numpy as np
import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger

import datasets.imagenet  # noqa: F401

# FG-BG baseliens
# + CoCo
from trainers import (  # noqa: F401  # noqa: F401
    SCT,
    LoCoOp,
    LoCoOpCoCo,
    Mambo,
    MamboCoCo,
    SCTCoCo,
)
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_ood_loader_ImageNet, set_val_loader


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    updates = {
        "root": "DATASET.ROOT",
        "output_dir": "OUTPUT_DIR",
        "resume": "RESUME",
        "seed": "SEED",
        "trainer": "TRAINER.NAME",
        "backbone": "MODEL.BACKBONE.NAME",
        "head": "MODEL.HEAD.NAME",
        "alpha_value": "lambda_value",
        "beta_value": "beta_value",
        "top_k": "top_k",
        "num_classes": "num_classes",
        "num_patches": "num_patches",
        "eta": "eta",
    }

    for arg_name, cfg_path in updates.items():
        val = getattr(args, arg_name, None)
        if val:
            parts = cfg_path.split(".")
            node = cfg
            for part in parts[:-1]:
                node = getattr(node, part)
            setattr(node, parts[-1], val)


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    if args.trainer in ["LoCoOp", "SCT", "Mambo", "LoCoOpCoCo", "SCTCoCo", "MamboCoCo"]:
        C = CN()
        C.N_CTX = 16  # number of context vectors
        C.CSC = False  # class-specific context
        C.CTX_INIT = ""  # initialization words
        C.PREC = "fp16"  # fp16, fp32, amp
        C.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

        cfg.TRAINER[args.trainer.upper()] = C

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    import clip_w_local

    cfg = setup_cfg(args)
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.in_dataset in ["imagenet"]:
        out_datasets = ["iNaturalist", "SUN", "places365", "Texture"]
        # Optional: OpenOOD benchmark
        out_datasets += ["SSB_hard", "NINCO", "OpenImage-O", "ImageNet-O"]
        # Optional: ImageNet-1k-OOD
        out_datasets += ["ImageNet-1k-OOD"]

    elif args.in_dataset in ["imagenet10"]:
        out_datasets = ["imagenet20"]
    elif args.in_dataset in ["imagenet20"]:
        out_datasets = ["imagenet10"]
    else:
        raise NotImplementedError("Unsupported in-distribution dataset.")

    trainer = build_trainer(cfg)

    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    id_data_loader = set_val_loader(args, preprocess)

    in_score_gl, in_score_r = trainer.test_ood(id_data_loader, args.T)

    auroc_list_gl, aupr_list_gl, fpr_list_gl = [], [], []
    auroc_list_r, aupr_list_r, fpr_list_r = [], [], []
    for out_dataset in out_datasets:
        print(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
        out_score_gl, out_score_r = trainer.test_ood(ood_loader, args.T)

        print("GL-MCM score")
        get_and_print_results(
            args, in_score_gl, out_score_gl, auroc_list_gl, aupr_list_gl, fpr_list_gl
        )

        print("R-MCM score")
        get_and_print_results(
            args, in_score_r, out_score_r, auroc_list_r, aupr_list_r, fpr_list_r
        )

        plot_distribution(args, in_score_gl, out_score_gl, out_dataset, score="GLMCM")
        plot_distribution(args, in_score_r, out_score_r, out_dataset, score="RMCM")

    print(
        "GL-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(
            np.mean(fpr_list_gl), np.mean(auroc_list_gl), np.mean(aupr_list_gl)
        )
    )
    print(
        "R-MCM avg. FPR:{}, AUROC:{}, AUPR:{}".format(
            np.mean(fpr_list_r), np.mean(auroc_list_r), np.mean(aupr_list_r)
        )
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument(
        "--in_dataset",
        default="imagenet",
        type=str,
        choices=["imagenet"],
        help="in-distribution dataset",
    )
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    # augment for GL-MCM and R-MCM
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="mini-batch size"
    )

    parser.add_argument("--T", type=float, default=1, help="temperature parameter")
    # params for CoCo
    parser.add_argument(
        "--alpha_value", type=float, default=0.2, help="weight for abs loss"
    )
    parser.add_argument(
        "--beta_value", type=float, default=3.0, help="weight for cfr loss"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="top k strategy for decomposing foreground and background patches in LoCoOp",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="number for selecting confusable classes",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=1,
        help="number of confuable foreground patches",
    )
    parser.add_argument(
        "--eta", type=float, default=5.0, help="scalling factor in abs module"
    )
    args = parser.parse_args()
    main(args)
