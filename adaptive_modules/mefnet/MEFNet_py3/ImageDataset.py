
import os
import functools
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg',
    '.jpeg',
    '.png',
    '.ppm',
    '.bmp',
    '.pgm',
    '.tif'
]


def has_file_allowed_extension(
    filename,
    extensions
):

    filename_lower = (
        filename.lower()
    )

    return any(
        filename_lower.endswith(ext)
        for ext in extensions
    )


def image_seq_loader(
    img_seq_dir
):

    img_seq_dir = os.path.expanduser(
        img_seq_dir
    )

    img_seq = []

    for root, _, fnames in sorted(
        os.walk(img_seq_dir)
    ):

        for fname in sorted(
            fnames
        ):

            if has_file_allowed_extension(
                fname,
                IMG_EXTENSIONS
            ):

                image_name = os.path.join(
                    root,
                    fname
                )

                img_seq.append(
                    Image.open(
                        image_name
                    ).convert("RGB")
                )

    return img_seq


def get_default_img_seq_loader():

    return functools.partial(
        image_seq_loader
    )


class ImageSeqDataset(
    Dataset
):

    def __init__(
        self,
        csv_file,
        hr_img_seq_dir,
        hr_transform=None,
        lr_transform=None,
        get_loader=get_default_img_seq_loader
    ):

        # Modern pandas-free line reader.
        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.seqs = [
                line.strip()
                for line in f
                if line.strip()
            ]

        self.hr_root = (
            hr_img_seq_dir
        )

        self.hr_transform = (
            hr_transform
        )

        self.lr_transform = (
            lr_transform
        )

        self.loader = (
            get_loader()
        )


    def __getitem__(
        self,
        index
    ):

        hr_seq_dir = os.path.join(
            self.hr_root,
            self.seqs[index]
        )

        I = self.loader(
            hr_seq_dir
        )

        if self.hr_transform is not None:

            I_hr = self.hr_transform(
                I
            )

        else:

            I_hr = I

        if self.lr_transform is not None:

            I_lr = self.lr_transform(
                I
            )

        else:

            I_lr = I

        I_hr = torch.stack(
            I_hr,
            0
        ).contiguous()

        I_lr = torch.stack(
            I_lr,
            0
        ).contiguous()

        return {
            "I_hr": I_hr,
            "I_lr": I_lr
        }


    def __len__(self):

        return len(
            self.seqs
        )


    @staticmethod
    def _reorderBylum(
        seq
    ):

        I = torch.sum(
            torch.sum(
                torch.sum(
                    seq,
                    1
                ),
                1
            ),
            1
        )

        _, index = torch.sort(
            I
        )

        return seq[
            index,
            :
        ]
