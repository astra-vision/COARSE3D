import argparse
import datetime
import os
import time

import numpy as np
import torch
import trainer
from option import Option

import pc_processor


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        if self.settings.pycharm:
            os.environ["MASTER_ADDR"] = self.settings.master_addr
            os.environ["MASTER_PORT"] = self.settings.master_port
            os.environ["NPROC_PER_NODE"] = "1"

        if self.settings.distributed:
            pc_processor.utils.init_distributed_mode(self.settings)
            torch.distributed.barrier()

        os.environ["PYTHONHASHSEED"] = str(self.settings.seed)
        np.random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)  # cpu
        torch.cuda.manual_seed(self.settings.seed)  # gpu
        torch.cuda.manual_seed_all(self.settings.seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True

        # set device
        if self.settings.pycharm:
            assert len(self.settings.gpu)
            device = torch.device(
                "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
            )
            torch.cuda.set_device(device=device)
        else:
            torch.cuda.set_device(device=self.settings.gpu)

        # init checkpoint
        if not self.settings.distributed or (self.settings.rank == 0):
            # only init recorer on the first GPU
            self.recorder = pc_processor.checkpoint.Recorder(self.settings)
        else:
            self.recorder = None

        self.model = self._initModel()

        # init trainer
        self.trainer = trainer.Trainer(
            settings=self.settings, model=self.model, recorder=self.recorder
        )
        if self.settings.checkpoint is not None:
            self.epoch_start = self.settings.epoch_start
        else:
            self.epoch_start = 0
        return

    def _initModel(self):
        if self.settings.net_type == "SalsaNextProto":
            model = pc_processor.models.SalsaNextProto(
                in_channel=self.settings.input_channels,
                nclasses=self.settings.n_classes,
                sub_proto_size=self.settings.sub_proto_size,
                ignore_label=self.settings.ignore_cls,
                proto_mom=self.settings.proto_momentum,
                dataset=self.settings.dataset,
            )
        elif self.settings.net_type in ["SqueezeSegV3Proto21", "SqueezeSegV3Proto53"]:
            if self.settings.net_type == "SqueezeSegV3Proto21":
                layers = 21
            elif self.settings.net_type == "SqueezeSegV3Proto53":
                layers = 53
            else:
                raise ValueError
            model = pc_processor.models.SqueezeSegV3Proto(
                nclasses=self.settings.n_classes,
                sub_proto_size=self.settings.sub_proto_size,
                ignore_label=self.settings.ignore_cls,
                proto_mom=self.settings.proto_momentum,
                dataset=self.settings.dataset,
                layers=layers,
            )
        elif self.settings.net_type in ["RangeNetProto21", "RangeNetProto53"]:
            if self.settings.net_type == "RangeNetProto21":
                layers = 21
            elif self.settings.net_type == "RangeNetProto53":
                layers = 53
            else:
                raise ValueError
            model = pc_processor.models.RangeNetProto(
                nclasses=self.settings.n_classes,
                sub_proto_size=self.settings.sub_proto_size,
                ignore_label=self.settings.ignore_cls,
                proto_mom=self.settings.proto_momentum,
                dataset=self.settings.dataset,
                layers=layers,
            )
        else:
            raise ValueError("invalid model: {}".format(self.settings.net_type))
        return model

    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return

        best_val_result = None

        for epoch in range(self.epoch_start, self.settings.n_epochs):
            self.trainer.run(epoch, mode="Train")
            if (
                epoch % self.settings.val_frequency == 0
                or epoch == self.settings.n_epochs - 1
            ):
                val_result = self.trainer.run(epoch, mode="Validation")

                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                "get better {} model: {}".format(k, v)
                            )
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path,
                                "best_{}_model.pth".format(k),
                            )
                            best_val_result[k] = v

                            state = {
                                "epoch": epoch,
                                "model_state": self.model.state_dict(),
                                "optimizer": self.trainer.optimizer.state_dict(),
                                "scheduler": self.trainer.scheduler.state_dict(),
                                "best_value": best_val_result[k],
                            }
                            torch.save(state, saved_path)

            # save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(
                    self.recorder.checkpoint_path, "checkpoint.pth"
                )
                checkpoint_data = {
                    "model_state": self.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "epoch": epoch,
                }

                torch.save(checkpoint_data, saved_path)
                # log
                if best_val_result is not None:
                    log_str = ">>> Best Result: "
                    for k, v in best_val_result.items():
                        log_str += "{}: {} ".format(k, v)
                    self.recorder.logger.info(log_str)

            if self.settings.is_debug and epoch > 5:
                break

        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info(
                "==== total cost time: {}".format(datetime.timedelta(seconds=cost_time))
            )
            print(self.recorder.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument(
        "config_path",
        type=str,
        metavar="config_path",
        help="path of config file, type: string",
    )
    parser.add_argument(
        "--id",
        type=int,
        metavar="experiment_id",
        required=False,
        help="id of experiment",
        default=0,
    )
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()
    print("===end train success===")
