import os
import sys

import yaml

sys.path.insert(0, "../../")

import pc_processor
import datetime


class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))
        # --------------------------------
        # common config --------------------
        # --------------------------------
        self.save_path = self.config["save_path"]  # log path
        self.seed = self.config["seed"]
        self.gpu = self.config["gpu"]  # GPU id to use, e.g. "0,1,2,3"
        self.master_addr = self.config["master_addr"]
        self.master_port = self.config["master_port"]
        self.rank = 0  # rank of distributed thread
        self.world_size = 1  # this is a place holder. will change in ddp mode
        self.distributed = self.config["distributed"]  #
        self.n_gpus = len(self.gpu.split(","))
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        self.print_frequency = self.config["print_frequency"]
        self.n_threads = self.config[
            "n_threads"
        ]  # number of threads used for data loading
        self.is_debug = self.config["is_debug"]
        self.pycharm = self.config["pycharm"]
        self.weak_label = self.config["weak_label"]
        self.experiment_id = self.config["experiment_id"]

        # --------------------------------
        # contrastive config ---------------
        # --------------------------------
        self.contrast_warmup = self.config["contrast_warmup"]
        self.loss_w_contrast = self.config["loss_w_contrast"]
        self.temperature = self.config["temperature"]
        self.num_anchor = self.config["num_anchor"]
        self.entropy_selection = self.config["entropy_selection"]
        self.sub_proto_size = self.config["sub_proto_size"]
        self.proto_momentum = self.config["proto_momentum"]

        # --------------------------------
        # training config ------------------
        # --------------------------------
        self.val_only = self.config["val_only"]
        self.n_epochs = self.config["n_epochs"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.warmup_epochs = self.config["warmup_epochs"]
        self.momentum = self.config["momentum"]
        self.val_frequency = self.config["val_frequency"]
        self.weight_decay = self.config["weight_decay"]
        self.optimizer = self.config["optimizer"]
        self.loss_w_lov_2d = self.config["loss_w_lov_2d"]
        self.loss_w_ce_2d = self.config["loss_w_ce_2d"]

        # checkpoint model ---------------------
        self.checkpoint = self.config["checkpoint"]
        if self.checkpoint is not None:
            self.epoch_start = self.config["epoch_start"]
        self.pretrained_model = self.config["pretrained_model"]
        self.only_encoder = self.config["only_encoder"]

        # --------------------------------
        # dataset config -------------------
        # --------------------------------
        self.dataset = self.config["dataset"]
        self.data_len = self.config["data_len"]
        self.n_classes = self.config["n_classes"]
        self.ignore_cls = self.config["ignore_cls"]
        self.data_config_path = self.config["data_config_path"]
        self.pcd_root = self.config["pcd_root"]
        self.weak_root = self.config["weak_root"]
        self.weak_label_name = self.config["weak_label_name"]
        self.train_seq = self.config["train_seq"]
        self.val_seq = self.config["val_seq"]

        # --------------------------------
        # model config ---------------------
        # --------------------------------
        # backbone config ------------------
        self.net_type = self.config["net_type"]
        self.input_channels = self.config["input_channels"]
        self.encoder_modules = yaml.safe_load(
            open(self.config["encoder_modules_path"], "r")
        ).values()

        self._prepare()

    def _prepare(self):

        self.save_path = os.path.join(
            self.save_path,
            "debug-{}_{:02d}{:02d}_id-{}".format(
                self.is_debug,
                datetime.date.today().month,
                datetime.date.today().day,
                self.experiment_id,
            ),
        )

    def check_path(self):
        if pc_processor.utils.is_main_process():
            assert os.path.exists(self.save_path) == False, "This exp exists already!"

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
