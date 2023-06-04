# Face-Aging
A GAN used for Face Aging

## Introduction

The ability to generate realistic images of a person's face as they age has many practical applications, from forensics to entertainment. However, face aging is a challenging problem due to the complex changes that occur in a person's facial features over time. In recent years, Generative Adversarial Networks (GANs) have emerged as a powerful tool for generating high-quality images. GANs are a type of deep neural network that consists of two components: a generator and a discriminator. The generator learns to generate images that are indistinguishable from real images, while the discriminator learns to distinguish between real and fake images. Together, these components form a game where the generator tries to fool the discriminator, and the discriminator tries to identify the fake images.

We propose a novel GAN architecture that is specifically designed for generating realistic images of a person's face as they age. Our model is trained on a large dataset of face images, and we evaluate its effectiveness in generating high-quality images. 

## Architecture

Our proposed novel GAN architecture consists of two generators, two age modulators, and a discriminator. One of the generators is responsible for aging/rejuvenating the input image to the target age while the other is responsible for converting the generated image back to the source age. The encoder-decoder architecture used for the generators is expected to encode the image such that only the identity information is present in the encoding. This encoding is then sent to an age modulator which also takes a target age group as input and generates age-specific features that are then sent to each layer of the decoder network for the generation of the aged/rejuvenated image. The same flow of data is followed by the second generator for the restoration of the original image. Once an image has been generated based on the target age group, the discriminator predicts the target age group. 

![Model Architecture](https://github.com/SauravSJK/Face-Aging/blob/main/Images/Architecture.png?raw=true)

## Dataset

We have used the UTKFace dataset for training and validation. The UTKFace dataset is a large-scale face dataset that contains over 20,000 images of faces with annotations for age, gender, and ethnicity. The dataset includes images of faces in a wide range of ages, from 0 to 116 years old. To facilitate the use of the dataset for research purposes, the images are preprocessed to ensure that they are cropped and aligned consistently.

## Results

## Replication steps

To setup the environment for the model, execute the below:

1. Clone the repository

`git clone SauravSJK/Face-Aging`

2. Change directory

`cd SauravSJK/Face-Aging`

3. Install virtualenv if not already installed

`pip install virtualenv`

4. Create your new environment (called 'venv' here)

`virtualenv venv`

5. Enter the virtual environment

`source venv/bin/activate`

6. Install the requirements in the current environment

`pip install -r requirements.txt`


To train the model from scratch, execute the below. This will:
1. Download the dataset
2. Create tfrecords
3. Train the discriminator and generator
4. Generate a prediction for the image "/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg" wth source age: 48

Note: This will take around 5 days to complete

`python3 main.py --strtopt "a"`

To predict using trained models, execute the below.
The default image is "/UTKFace/48_0_0_20170120134009260.jpg.chip.jpg" wth source age: 48. This can be changed by specifying a different image using the "prediction_file_name" argument.

`python3 main.py --strtopt "p" --prediction_file_name "/UTKFace/1_0_0_20161219200338012.jpg.chip.jpg"`
