<div align="center">
  <img src="static/icon.png" alt="Logo" width="100" height="100">

  <h2 align="center">MRI Brain Tumor Detection - Computer Vision</h2>

</div>

The purpose of this project is to explore different kind of method of machine learning and computer vision to detect brain tumor in MRI scans.

I trained a total of 17 models, 15 models for set1 (5 types of image filtering for 3 modalities each), and 2 models for set2, (original and color quantization via kMeans Clustering)

---

### Definitions/Examples

- modality: t1, flair, segmentation
- dataset_trained_on: set1, set2
- CV_filtering: kMeans, harris, hough, canny, original
- the_models_accuracy: 0.90, 0.86, 0.65
- overfitting_potential: lofp, hofp, hufp
- tumor_true_or_false: True, False
- type_of_tumor: glioma, meningioma, notumor, pituitary

<br/>

- of = overfitting
- lufp = low underfitting potential
- hufp = high underfitting potential
- lofp = low overfitting potential
- hofp = high overfitting potential
- nofp = no overfitting potential
- True = contains tumor
- False = does not contain tumor

---

### Dataset Structure

```
.
├── set1: set1 consists of a dataset of 128 by 256 images containing 1 hemisphere of the brain
│     ├── canny: contains the same images in original except that they are filtered with the canny edge detector
│     │     ├── flairs
│     │     ├── segmentation
│     │     └── t1
│     │
│     ├── harris: contains the same images in original except that they are filtered with the harris corner detector
│     │     ├── flairs
│     │     ├── segmentation
│     │     └── t1
│     │
│     ├── hough: contains the same images in original except that they are filtered with the hough circle detection
│     │     ├── flairs
│     │     ├── segmentation
│     │     └── t1
│     │
│     ├── kMeans: contains the same images in original except that they are filtered with the kMeans clustering into 8 colors
│     │     ├── flairs
│     │     ├── segmentation
│     │     └── t1
│     │
│     └── original: contains the original images
│           ├── flairs
│           ├── segmentation
│           └── t1
│
└── set2: set2 consists of a dataset of 256 by 256 images containing 2 hemispheres of the brain and images taken from many axial planes
      ├── kMeans: contains the kMeans clustering of the images
      │     ├── glioma
      │     ├── meningioma
      │     ├── notumor
      │     └── pituitary
      │
      └── original: contains the original images
            ├── glioma
            ├── meningioma
            ├── notumor
            └── pituitary

```

**flairs**: the flair imaging modality with 2 classes (tumor and non-tumor)

**segmentation**: the segmentation imaging modality with 2 classes (tumor and non-tumor)

**t1**: the t1 imaging modality with 2 classes (tumor and non-tumor)

**glioma, meningioma, notumor, pituitary**: These are the 4 classes of the images in set2, this dataset is more diverse, it contains images from many axial planes, it contains 2 hemispheres of the brain, and more modalities.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Here is a list of prerequisites for this project

- python 3.10
- pip 22.3.1
- pipenv 2022.11.30 (optional)

### Installing

if you have pipenv installed, you can run the following command to install the dependencies

```bash
pipenv shell
pipenv install
```

if you don't have pipenv installed, you can run the following command to install the dependencies

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tensorflow
```

or you can use the shortcut command

```bash
pip install -r requirements.txt
```

if you encounter any issues, in terms of dependencies, please refer to the [requirements.txt](requirements.txt) file for a list of required dependencies.

### Usage

To activate this project's virtualenv, run pipenv shell.
When in the virtualenv, type exit to escape

How commands are formed

```bash
python predict.py [argument 1] [argument 2] [argument 3]
```

### argument 1: set1 or set2

set1 and set2 are the 2 different datasets used in this project, so feeding a image from set1 into a model trained on set2 (or vice versa) will give bad results.

- set1: binary classification of tumor and non-tumor
- set2: multi-class classification of glioma, meningioma, notumor, pituitary

keep [dataset_trained_on] the same in the model file name and the image file name for proper results

### argument 2: model file name

This can be any file in the models folder, but it must be a .h5 file

model file as named as

- [modality]-[dataset_trained_on]-[CV_filtering]-[the_models_accuracy]-[overfitting_potential].h5

or

- [dataset_trained_on]-[CV_filtering]-[the_models_accuracy]-[overfitting_potential].h5

### argument 3: image file name

This can be any image file in the tests folder, but it must be a .png or .jpg file, for more images to test on, look at the dataset links in the [Acknowledgments for Datasets](#acknowledgments-for-datasets) section and add them to the tests folder

The images in the tests folder can be named anything, but for user understanding, the image files in the tests folder as named as

- [CV_filtering]-[dataset_trained_on]-[modality]-[patient_id]-[tumor_true_or_false].png

or

- [CV_filtering]-[dataset_trained_on]-[type_of_tumor]-[patient_id].jpg

Example of commands to run the program

```bash
python predict.py set1 t1-set1-harris-0.65-hofp.h5 harris-set1-t1-HG0001-85-True.png

python predict.py set1 flair-set1-hough-0.86-hufp.h5 hough-set1-flair-HG0001-57-False.png

python predict.py set2 set2-kMeans-0.90-lofp.h5 kMeans-set2-pituitary-Te-pi_0056.jpg

python predict.py set2 set2-kMeans-0.90-lofp.h5 kMeans-set2-notumor-Te-no_0020.jpg

python predict.py set2 set2-kMeans-0.90-lofp.h5 kMeans-set2-glioma-Te-gl_0025.jpg
```

## Built With

[![](https://img.shields.io/badge/python-000000?style=for-the-badge&logo=python&logoColor=white)]()
[![](https://img.shields.io/badge/numpy-000000?style=for-the-badge&logo=numpy&logoColor=white)]()
[![](https://img.shields.io/badge/OpenCV-000000?style=for-the-badge&logo=opencv&logoColor=white)]()
[![](https://img.shields.io/badge/TensorFlowJS-000000?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![](https://img.shields.io/badge/keras-000000?style=for-the-badge&logo=keras&logoColor=white)]()
[![](https://img.shields.io/badge/Nextjs-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)]()

---

## Contributing

If you see an issue or would like to contribute, please do & open a pull request or ticket for/with new features or fixes.

---

## Authors

- **Justin Zhang** - _Initial work_ - [JustinZhang17](https://github.com/JustinZhang17)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments for Datasets

- [BRATS2013 Tumor-NoTumor Dataset (T-NT) for providing the dataset for set1](https://paperswithcode.com/dataset/brats-2013-1)

- [Masoud Nickparvar for providing the dataset for set2](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
