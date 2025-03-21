# VFFM-UNet

This is the official code repository for "Multi-Granularity Vision Fastformer with Fusion Mechanism for Skin Lesion Segmentation"

## Novelty and Contribution

- We propose VFFM-UNet, a hybrid architecture network. The model achieves good performance with 0.35M parameters and 0.494 GFLOPs, **effectively balancing computational costs and long-range dependency modelling.**
- We introduce **a language model, Fastformer**, into skin lesion segmentation **for the first time**. By tackling the challenge of identifying unclear lesion boundaries, we explored **the potential of Fastformer for this task.**
- We introduce **Multi-Granularity Vision Fastformer** to extract feature maps at different granularities and incorporate Fusion Mechanism, including Multi-Granularity Fusion and Channel Fusion, to **accomplish the model's generalization ability in lesions with a different degree of severity.**
- Extensive experiments on three datasets for public skin lesion segmentation, **ISIC2017, ISIC2018, and PH2 dataset**, demonstrate that VFFM-UNet is state-of-the-art in terms of number of parameters, computational complexity, and segmentation performance.

## Main Environments

- pytorch 2.0.0
- torchvision 0.15.1

## Prepare the Dataset

-   The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}.

-   After downloading the datasets, you are supposed to put them into './data/isic2017/' and './data/isic2018/', and the file format reference is as follows. (take the ISIC2017 dataset as an example.)

-   './data/isic2017/'
    -   train
        -   images
            -   .png
        -   masks
            -   .png
    -   val
        -   images
            -   .png
        -   masks
            -   .png

## Comparison Results

Results for ISIC 2018 Dataset. VFFM-UNet shows significant advantages on segmentation performance.

| Methods    | Year | Params (M)↓ | GFLOPs↓ | mIoU↑     | DSC↑      | Acc↑      | Sen↑      | Spe↑      |
| ---------- | ---- | ----------- | ------- | --------- | --------- | --------- | --------- | --------- |
| UNet       | 2015 | 31.04       | 54.74   | 77.88     | 87.56     | 94.03     | 87.23     | 96.19     |
| UNet++     | 2018 | 47.19       | 200.12  | 79.18     | 88.38     | 94.40     | 88.27     | 96.34     |
| TransFuse  | 2021 | 41.34       | 8.87    | 74.19     | 85.18     | 93.02     | 83.27     | 96.12     |
| UTNet      | 2021 | 15.29       | 22.55   | 78.08     | 87.69     | 93.98     | 89.09     | 95.53     |
| MISSFormer | 2022 | 35.45       | 7.28    | 79.09     | 88.32     | 94.39     | 87.61     | 96.56     |
| C²SDG      | 2023 | 22.01       | 7.97    | 79.58     | 88.63     | 94.43     | 90.13     | 95.79     |
| ERDUnet    | 2024 | 10.21       | 10.29   | 79.28     | 88.44     | 94.38     | 89.03     | 96.09     |
| MHorUNet   | 2024 | 3.49        | 0.57    | 79.18     | 88.38     | 94.49     | 86.84     | 96.92     |
| HSH-UNet   | 2024 | 18.04       | 9.36    | 77.39     | 87.25     | 93.80     | 87.95     | 95.66     |
| **Ours**   | -    | 0.35        | 0.494   | **80.62** | **89.27** | **94.73** | **90.74** | **97.23** |



Results for ISIC 2017 Dataset. VFFM-UNet shows significant advantages on segmentation performance.

| Methods    | Year | Params (M)↓ | GFLOPs↓ | mIoU↑     | DSC↑      | Acc↑      | Sen↑      | Spe↑      |
| ---------- | ---- | ----------- | ------- | --------- | --------- | --------- | --------- | --------- |
| UNet       | 2015 | 31.04       | 54.74   | 77.08     | 87.06     | 95.82     | 84.68     | 96.93     |
| UNet++     | 2018 | 47.19       | 200.12  | 76.80     | 86.88     | 95.66     | 86.31     | 97.53     |
| TransFuse  | 2021 | 41.34       | 8.87    | 72.53     | 83.89     | 94.92     | 80.53     | **97.99** |
| UTNet      | 2021 | 15.29       | 22.55   | 76.79     | 86.87     | 95.63     | 86.76     | 97.41     |
| MISSFormer | 2022 | 35.45       | 7.28    | 76.97     | 86.98     | 95.81     | 84.14     | 97.94     |
| C²SDG      | 2023 | 22.01       | 7.97    | 77.13     | 87.09     | 95.74     | 86.11     | 97.67     |
| ERDUnet    | 2024 | 10.21       | 10.29   | 78.19     | 87.76     | 95.96     | **86.89** | 97.77     |
| MHorUNet   | 2024 | 3.49        | 0.57    | **78.48** | **87.94** | **96.08** | 85.81     | 97.96     |
| HSH-UNet   | 2024 | 18.04       | 9.36    | 76.85     | 86.91     | 95.84     | 83.02     | 97.40     |
| **Ours**   | -    | 0.35        | 0.494   | **79.00** | **88.32** | **96.31** | **87.11** | **98.22** |

Results for PH$^{2}$ Dataset. VFFM-UNet shows significant advantages on segmentation performance.

| Methods   | Year | Params (M)↓ | GFLOPs↓ | mIoU↑     | DSC↑      | Acc↑      | Sen↑      | Spe↑      |
| --------- | ---- | ----------- | ------- | --------- | --------- | --------- | --------- | --------- |
| UNet      | 2015 | 31.04       | 54.74   | 78.82     | 88.16     | 92.71     | 85.05     | **96.30** |
| UNet++    | 2018 | 47.19       | 200.12  | 78.86     | 88.18     | 92.53     | 87.41     | 94.93     |
| TransFuse | 2021 | 41.34       | 8.87    | 76.44     | 86.65     | 91.18     | 89.72     | 91.87     |
| UTNet     | 2021 | 15.29       | 22.55   | 81.57     | 89.85     | 93.54     | 89.62     | 95.38     |
| C²SDG     | 2023 | 22.01       | 7.97    | 81.63     | 89.88     | 93.46     | 91.08     | 94.58     |
| MHorUNet  | 2024 | 3.49        | 0.57    | 78.98     | 88.25     | 92.50     | 88.40     | 94.42     |
| HSH-UNet  | 2024 | 18.04       | 9.36    | **82.25** | **90.26** | **93.73** | **91.14** | 94.14     |
| **Ours**  | -    | 0.35        | 0.494   | **83.45** | **91.02** | **94.94** | **92.89** | **95.85** |
