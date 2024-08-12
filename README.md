# Classification and Recognition of Soybean Leaf Diseases Using Deep Learning

## Overview

This project presents a deep learning-based approach for the automatic detection and classification of soybean leaf diseases. The research was conducted to assist farmers in Madhya Pradesh and Chhattisgarh, where soybean is a critical crop, in identifying diseases early to mitigate potential losses. The system leverages advanced image processing techniques and neural networks to classify common soybean diseases with high accuracy.

***TThis is the first research paper to utilize a soybean disease dataset specifically from Madhya Pradesh, marking it as a pioneering effort in applying deep learning to address the agricultural challenges in this region. Additionally, this dataset is the first of its kind to be made publicly available on the internet, created by us.***

## Key Features

* **Deep Learning Models**: Implemented and compared various deep learning models, including CNN and ResNet-V2, for disease classification.

* **Data Augmentation**: Used data augmentation techniques to enhance model performance by generating additional training samples.

* **High Accuracy**: Achieved a classification accuracy of 93.01% using the ResNet-V2 model.


## Technology Used

* Deep Learning Frameworks: TensorFlow, Keras

* Programming Language: Python

## Dataset Prepration

* Collected and labeled soybean leaf images affected by various diseases, including bacterial blight, frog eye leaf spot (FLS), and brown spot.

* Applied data augmentation techniques to increase the diversity of the training dataset.
## Image Processing

* Pre-processed images by converting them into different color spaces and normalizing pixel values.

* Extracted features from images using deep convolutional neural networks.
## Model Training

* Trained models on both raw images and feature-extracted data.

* Fine-tuned a ResNet-V2 model to achieve the highest classification accuracy.
## Installation

1. Clone the repository:

```bash
https://github.com/shri11081999/2023-IEEE-Conference--PCEMS-.git
```
2. Install the required dependencies:

* Python 3.7 or later
* TensorFlow, PyTorch, Keras
* OpenCV or scikit-image for image preprocessing
* Jupyter Notebook


3. Download the Dataset

Acquire the Soybean dataset and place it in the data/ directory.


## Contributing

Feel free to make any changes in the project.
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Evaluation

* Compared the performance of models with and without data augmentation.

* Evaluated models using standard classification metrics such as accuracy, precision, recall, and F1-score.
## Results

* **CNN Model**:  The basic CNN model achieved satisfactory results but was outperformed by more advanced architectures.

* **ResNet-V2**: The fine-tuned ResNet-V2 model achieved the highest accuracy of 93.01% in identifying soybean diseases.


* **Data Augmentation**: Significantly improved the performance of all models.

## Conference Presentation

**Conference** : 2nd International Conference on the Paradigm Shifts in Communication, Embedded Systems, Machine Learning, and Signal Processing (PCEMS 2023)

**Paper ID** : 152

**Date** : 2023

**Location** : National Institute of Technology (VNIT), Nagpur
## Future-Work

* **Expand Dataset** : Collect and label more data to improve model generalization.

* **Cross-Crop Application** : Adapt the model to detect diseases in other crops such as rice and maize.

* **Explainability**: Incorporate techniques to explain model predictions to increase trust in AI-driven diagnosis.
## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

We would like to express our gratitude to the participants of the PCEMS 2023 conference for their valuable feedback, and to the School of Agricultural Sciences for their assistance in providing the dataset used in this research.


## Contact

For any questions or issues, please contact dixitshriniket976@gmail.com.
