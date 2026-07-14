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
            ├── train/
            │   ├── images
            │   ├── 0a7e06.jpg
            │   │   ├── 0aab0a.jpg
            │   │   ├── 0b1761.jpg
            │   │   ├── ...
            │   |
            │   └── masks
            │       ├── 0
            │       |   ├── 0a7e06.png
            │       |   ├── 0aab0a.png
            │       |   ├── 0b1761.png
            │       |   ├── ...
            │       |
            │       ├── 1
            │       |   ├── 0a7e06.png
            │       |   ├── 0aab0a.png
            │       |   ├── 0b1761.png
            │       |   ├── ...
            ├── val/
            │   ├── images/
            │   └── masks/
            └── test/
            ├── images/
            └── masks/
```

## 2. Download the my model weights

Download the following files and place them in the `./models` directory:

通过网盘分享的文件：The weights I've trained.zip
链接: https://pan.baidu.com/s/1tCpnephidWLbFVi9jI-ieA?pwd=iirm 提取码: iirm 
--来自百度网盘超级会员v6的分享


## 3. Usage


First, train the BRnet. The trained weights will be saved in `./models`.

```bash
python train.py
```

Then, use the trained weights to predict the test set images and obtain segmentation result.

Check the 'outputs/' folder for prediction masks.

Check the 'output2/' folder for difference and overlay maps.

```bash
python val.py 
```

## Citation

If you use this project in your research, please cite the following paper:

```bibtex
@article{Zhang2026BRNet,
  author    = {Zhang, Zhan and Shao, Hong and Cui, Wencheng},
  title     = {{BRNet}: A Dual-Backbone {X-Ray} Coronary Angiography Segmentation Network Based on Multi-Scale Fusion and Dynamic Detail Reconstruction},
  journal   = {Applied Sciences},
  year      = {2026},
  volume    = {16},
  number    = {14},
  pages     = {6960},
  doi       = {10.3390/app16146960},
  url       = {https://doi.org/10.3390/app16146960},
  publisher = {MDPI}
}
```

---
