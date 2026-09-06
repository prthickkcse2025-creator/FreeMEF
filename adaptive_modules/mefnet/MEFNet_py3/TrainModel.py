import os
import time

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchvision import transforms, utils

from e2emef import E2EMEF
from mefssim import MEF_MSSSIM

from ImageDataset import ImageSeqDataset

from batch_transformers import (
    BatchRandomResolution,
    BatchToTensor,
    BatchRGBToYCbCr,
    YCbCrToRGB,
    BatchTestResolution
)


EPS = 1e-8


# ============================================================
# TRAINER
# ============================================================

class Trainer(object):

    def __init__(
        self,
        config
    ):

        torch.manual_seed(
            config.seed
        )

        # ----------------------------------------------------
        # HIGH-RESOLUTION TRAIN TRANSFORM
        # ----------------------------------------------------

        self.train_hr_transform = (
            transforms.Compose([
                BatchRandomResolution(
                    config.high_size,
                    interpolation=2
                ),
                BatchToTensor(),
                BatchRGBToYCbCr()
            ])
        )

        # ----------------------------------------------------
        # LOW-RESOLUTION TRAIN TRANSFORM
        # ----------------------------------------------------

        self.train_lr_transform = (
            transforms.Compose([
                BatchRandomResolution(
                    config.low_size,
                    interpolation=2
                ),
                BatchToTensor(),
                BatchRGBToYCbCr()
            ])
        )

        # ----------------------------------------------------
        # HIGH-RESOLUTION TEST TRANSFORM
        # ----------------------------------------------------

        self.test_hr_transform = (
            transforms.Compose([
                BatchTestResolution(
                    2048,
                    interpolation=2
                ),
                BatchToTensor(),
                BatchRGBToYCbCr()
            ])
        )

        # ----------------------------------------------------
        # LOW-RESOLUTION TEST TRANSFORM
        # ----------------------------------------------------

        self.test_lr_transform = (
            self.train_lr_transform
        )

        # ----------------------------------------------------
        # BATCH SIZE
        # ----------------------------------------------------

        self.train_batch_size = 1
        self.test_batch_size = 1

        # ----------------------------------------------------
        # TRAIN DATA
        # ----------------------------------------------------

        self.train_data = ImageSeqDataset(
            csv_file=os.path.join(
                config.trainset,
                "train.txt"
            ),
            hr_img_seq_dir=config.trainset,
            hr_transform=self.train_hr_transform,
            lr_transform=self.train_lr_transform
        )

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1
        )

        # ----------------------------------------------------
        # TEST DATA
        # ----------------------------------------------------

        self.test_data = ImageSeqDataset(
            csv_file=os.path.join(
                config.testset,
                "test.txt"
            ),
            hr_img_seq_dir=config.testset,
            hr_transform=self.test_hr_transform,
            lr_transform=self.test_lr_transform
        )

        self.test_loader = DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = E2EMEF(
            is_guided=True
        )

        self.model_name = type(
            self.model
        ).__name__

        print(
            self.model
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        self.loss_fn = MEF_MSSSIM(
            is_lum=True
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.initial_lr = config.lr

        if self.initial_lr is None:

            lr = 0.0005

        else:

            lr = self.initial_lr

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # ----------------------------------------------------
        # CUDA
        # ----------------------------------------------------

        if (
            torch.cuda.is_available()
            and
            config.use_cuda
        ):

            self.model.cuda()

            self.loss_fn = (
                self.loss_fn.cuda()
            )

        # ----------------------------------------------------
        # STATES
        # ----------------------------------------------------

        self.start_epoch = 0
        self.start_step = 0

        self.train_loss = []

        self.test_results = []

        self.ckpt_path = (
            config.ckpt_path
        )

        self.use_cuda = (
            config.use_cuda
        )

        self.max_epochs = (
            config.max_epochs
        )

        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        if config.ckpt is not None:

            ckpt = os.path.join(
                self.ckpt_path,
                config.ckpt
            )

            self._load_checkpoint(
                ckpt
            )


    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self):

        for epoch in range(
            self.start_epoch,
            self.max_epochs
        ):

            loss = self.train(
                epoch
            )

            print(
                "Epoch {} Training: "
                "Loss = {:.6f}".format(
                    epoch,
                    loss
                )
            )


    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train(
        self,
        epoch
    ):

        self.model.train()

        for step, sample_batched in enumerate(
            self.train_loader,
            0
        ):

            i_hr = (
                sample_batched["I_hr"]
            )

            i_lr = (
                sample_batched["I_lr"]
            )

            i_hr = torch.squeeze(
                i_hr,
                dim=0
            )

            i_lr = torch.squeeze(
                i_lr,
                dim=0
            )

            Y_hr = (
                i_hr[:, 0, :, :]
                .unsqueeze(1)
            )

            Y_lr = (
                i_lr[:, 0, :, :]
                .unsqueeze(1)
            )

            I_hr = Variable(
                Y_hr
            )

            I_lr = Variable(
                Y_lr
            )

            if self.use_cuda:

                I_hr = I_hr.cuda()

                I_lr = I_lr.cuda()

            self.optimizer.zero_grad()

            O_hr, W_hr = (
                self.model(
                    I_lr,
                    I_hr
                )
            )

            loss = self.loss_fn(
                O_hr,
                I_hr
            )

            loss.backward()

            self.optimizer.step()

            self.train_loss.append(
                loss.data.item()
            )

            self.start_step = (
                step
            )

        return loss.data.item()


    # ========================================================
    # EVALUATION
    # ========================================================

    def eval(
        self,
        epoch
    ):

        self.model.eval()

        scores = []

        with torch.no_grad():

            for step, sample_batched in enumerate(
                self.test_loader,
                0
            ):

                # ------------------------------------------------
                # LOAD DATA
                # ------------------------------------------------

                i_hr = (
                    sample_batched["I_hr"]
                )

                i_lr = (
                    sample_batched["I_lr"]
                )

                i_hr = torch.squeeze(
                    i_hr,
                    dim=0
                )

                i_lr = torch.squeeze(
                    i_lr,
                    dim=0
                )

                # ------------------------------------------------
                # Y / Cb / Cr
                # ------------------------------------------------

                Y_hr = (
                    i_hr[:, 0, :, :]
                    .unsqueeze(1)
                )

                Cb_hr = (
                    i_hr[:, 1, :, :]
                    .unsqueeze(1)
                )

                Cr_hr = (
                    i_hr[:, 2, :, :]
                    .unsqueeze(1)
                )

                # ------------------------------------------------
                # Chroma fusion
                # ------------------------------------------------

                Wb = (
                    torch.abs(
                        Cb_hr - 0.5
                    )
                    +
                    EPS
                )

                Wb = (
                    Wb /
                    torch.sum(
                        Wb,
                        dim=0,
                        keepdim=False
                    )
                )

                Wr = (
                    torch.abs(
                        Cr_hr - 0.5
                    )
                    +
                    EPS
                )

                Wr = (
                    Wr /
                    torch.sum(
                        Wr,
                        dim=0,
                        keepdim=False
                    )
                )

                Cb_f = torch.sum(
                    Wb * Cb_hr,
                    dim=0,
                    keepdim=True
                ).clamp(
                    0,
                    1
                )

                Cr_f = torch.sum(
                    Wr * Cr_hr,
                    dim=0,
                    keepdim=True
                ).clamp(
                    0,
                    1
                )

                # ------------------------------------------------
                # Low-resolution luminance
                # ------------------------------------------------

                Y_lr = (
                    i_lr[:, 0, :, :]
                    .unsqueeze(1)
                )

                I_hr = Variable(
                    Y_hr
                )

                I_lr = Variable(
                    Y_lr
                )

                if self.use_cuda:

                    I_hr = I_hr.cuda()

                    I_lr = I_lr.cuda()

                # ------------------------------------------------
                # MEF-Net
                # ------------------------------------------------

                O_hr, W_hr = (
                    self.model(
                        I_lr,
                        I_hr
                    )
                )

                # ------------------------------------------------
                # Quality
                # ------------------------------------------------

                q = self.loss_fn(
                    O_hr,
                    I_hr
                ).cpu()

                scores.append(
                    q.data.numpy()
                )

                # ------------------------------------------------
                # RGB reconstruction
                # ------------------------------------------------

                O_hr_RGB = (
                    YCbCrToRGB()(
                        torch.cat(
                            (
                                O_hr.cpu(),
                                Cb_f,
                                Cr_f
                            ),
                            dim=1
                        )
                    )
                )

                # ------------------------------------------------
                # Save fused image
                # ------------------------------------------------

                self._save_image(
                    O_hr_RGB,
                    self.fused_img_path,
                    str(epoch)
                    + "_"
                    + str(step)
                )

                # ------------------------------------------------
                # Save weight map
                # ------------------------------------------------

                self._save_image(
                    W_hr,
                    self.weight_map_path,
                    str(epoch)
                    + "_"
                    + str(step)
                )

        if len(scores) == 0:

            raise RuntimeError(
                "MEF-Net evaluation produced "
                "no samples."
            )

        avg_quality = (
            sum(scores)
            /
            len(scores)
        )

        return avg_quality


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def _load_checkpoint(
        self,
        ckpt
    ):

        if os.path.isfile(
            ckpt
        ):

            print(
                "[*] loading checkpoint "
                "'{}'".format(
                    ckpt
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # PyTorch 2.6 changed torch.load default
            # weights_only from False to True.
            #
            # This old MEF-Net checkpoint is a trusted local
            # checkpoint from the cloned repository, so the
            # compatibility copy explicitly uses
            # weights_only=False.
            # ------------------------------------------------

            checkpoint = torch.load(
                ckpt,
                map_location="cpu",
                weights_only=False
            )

            # ------------------------------------------------
            # Restore states
            # ------------------------------------------------

            self.start_epoch = (
                checkpoint["epoch"]
                + 1
            )

            self.train_loss = (
                checkpoint[
                    "train_loss"
                ]
            )

            self.test_results = (
                checkpoint[
                    "test_results"
                ]
            )

            # ------------------------------------------------
            # Model weights
            # ------------------------------------------------

            self.model.load_state_dict(
                checkpoint[
                    "state_dict"
                ]
            )

            # ------------------------------------------------
            # Optimizer state
            # ------------------------------------------------

            try:

                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer"
                    ]
                )

            except Exception as exc:

                print(
                    "[!] Optimizer state could not "
                    "be restored during inference:"
                )

                print(
                    exc
                )

            # ------------------------------------------------
            # Initial learning rate
            # ------------------------------------------------

            if (
                self.initial_lr
                is not None
            ):

                for param_group in (
                    self.optimizer.param_groups
                ):

                    param_group[
                        "initial_lr"
                    ] = (
                        self.initial_lr
                    )

            print(
                "[*] loaded checkpoint '{}' "
                "(epoch {})".format(
                    ckpt,
                    checkpoint[
                        "epoch"
                    ]
                )
            )

        else:

            print(
                "[!] no checkpoint found "
                "at '{}'".format(
                    ckpt
                )
            )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    @staticmethod
    def _save_checkpoint(
        state,
        filename="checkpoint.pth.tar"
    ):

        torch.save(
            state,
            filename
        )


    # ========================================================
    # FIND LATEST CHECKPOINT
    # ========================================================

    @staticmethod
    def _get_latest_checkpoint(
        path
    ):

        ckpts = os.listdir(
            path
        )

        ckpts = [
            ckpt
            for ckpt in ckpts
            if not os.path.isdir(
                os.path.join(
                    path,
                    ckpt
                )
            )
        ]

        if not ckpts:

            return None

        all_times = sorted(
            ckpts,
            reverse=True
        )

        return os.path.join(
            path,
            all_times[0]
        )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    @staticmethod
    def _save_image(
        image,
        path,
        name
    ):

        os.makedirs(
            path,
            exist_ok=True
        )

        batch_size = (
            image.size()[0]
        )

        for i in range(
            batch_size
        ):

            tensor = (
                image.data[i].clone()
            )

            tensor[
                tensor > 1
            ] = 1

            tensor[
                tensor < 0
            ] = 0

            utils.save_image(
                tensor,
                "%s/%s_%d.png"
                % (
                    path,
                    name,
                    i
                )
            )
