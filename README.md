# BRnet
To solve the limitations of existing methods in fine blood vessel extraction, edge blur processing, and global semantic modeling, this paper proposes BRNet. It effectively solves the problem that accuracy and robustness are difficult to balance in XCA image segmentation.

## 0. Environments

```bash
pip install -r requirements.txt
```
## 1. Prepare the datasets

Download the data set used in this article from the following sources:

ARCADE dataset: https://doi.org/10.1038/s41597-023-02871-z

DCA1 dataset: https://doi.org/10.3390/app9245507

XCAD dataset: https://doi.org/10.1109/iccv48922.2021.00744

CHUAC dataset: https://doi.org/10.1016/j.bspc.2018.06.007

After downloading the datasets, you are supposed to put them into `./` and the file format reference is as follows.

```
 Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
```

## 2. Download the my model weights

Download the following files from Google Drive and place them in the `./pre_trained_weights` directory:

- [vmamba_small_e238_ema.pth](https://drive.google.com/file/d/1XL7JuacjoZCr8w2b0c8CaQn8b0hREblk/view?usp=drive_link)

- [best-epoch142-loss0.3230.pth](https://drive.google.com/file/d/1jsZKakA4FrYaMXNp6qkVtxXwwcJQKrW4/view?usp=drive_link)

- [best-epoch142-loss0.3488.pth](https://drive.google.com/file/d/1OKIzUM_L6FeEqyuIsAMn4x-FHptizTkG/view?usp=drive_link)

- [MedSAM_model.pth](https://drive.google.com/file/d/1O5IVkcVxd2RtOcZEKuTR3WkOBiosHBfz/view?usp=drive_link)


## 3. Usage

```bash
bash train.sh
```

First, train the Pure VM-UNet of Branch 1. The trained weights will be saved in `./result_branch1/`.

```bash
bash test.sh
```

Then, use the trained weights to predict the test set images and obtain pred_masks.

```bash
bash train.sh
```

Finally, train the SAM-VMNet of Branch 2. The trained weights will be saved in `./result_branch1/`.
