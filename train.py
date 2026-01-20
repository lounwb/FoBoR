import argparse

import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger

import datasets.imagenet  # noqa: F401

# FG-BG baseliens and + CoCo
from trainers import (  # noqa: F401  # noqa: F401
    SCT,
    LoCoOp,
    LoCoOpCoCo,
    Mambo,
    MamboCoCo,
    SCTCoCo,
)


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
        "alpha_value": "alpha_value",
        "beta_value": "beta_value",
        "lambda_value": "lambda_value",
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
    else:
        raise ValueError("Trainer {} not recognized, support List: ['LoCoOp', 'SCT', 'Mambo', 'LoCoOpCoCo', 'SCTCoCo', 'MamboCoCo']".format(args.trainer))


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
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
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
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
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
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # params for CoCo
    parser.add_argument(
        "--alpha_value", type=float, default=0.2, help="default weight for abs loss"
    )
    parser.add_argument(
        "--beta_value", type=float, default=3.0, help="default weight for cfr loss"
    )
    parser.add_argument(
        "--lambda_value", type=float, default=0.17, help="default weight for balancing textual modality similarity & visual modality similarity"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="top k strategy for decomposing foreground and background patches in LoCoOp",
    )
    parser.add_argument(
        "--num_confuse_classes",
        type=int,
        default=1,
        help="number for selecting confusable classes",
    )
    parser.add_argument(
        "--num_confuse_patches",
        type=int,
        default=1,
        help="number of confuable foreground patches",
    )
    parser.add_argument(
        "--eta", type=float, default=5.0, help="scalling factor in abs module"
    )

    args = parser.parse_args()
    main(args)
