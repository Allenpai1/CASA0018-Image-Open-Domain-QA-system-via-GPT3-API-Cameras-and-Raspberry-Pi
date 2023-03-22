# CASA0018: Image Open Domain QA system via GPT3 API and Cameras

CASA0018 Project - Image Open Domain QA system via GPT3 API and Cameras

The detailed project description can be found in my report (Link wait to be upolad).

> We suggest that students take a fork of this repository so that they can add their own work in progress as they work through the material.

## System Overview

For this project, my multilingual language (Chinese and English) image-to-text open domain QA system with the [ChatGPT-API](https://openai.com/blog/introducing-chatgpt-and-whisper-apis) and [Paddle OCR](https://github.com/PaddlePaddle/PaddleOCR) consists of 3 steps:
 1. Language and text detection: a detection model FRCNN is used to recognize the text from the image and language of that text(Chinese or English) and crop the text from the original image with the language classification results passed into downstream tasks;
 2. Convert zoomed image to text: the previous step results are passed into paddleOCR, and the corresponding language text recognition model is loaded to extract the text from zoomed text image.
 3. QA system: ChatGPT API is used to pass the detected text and return a detailed answer, saved as PDF format and JPG images.

![alt text](https://github.com/Allenpai1/CASA0018-Image-Open-Domain-QA-system-via-GPT3-API-and-Cameras/main/Images/system.png)

I tried the handwritten text recognition models on the open-source library KerasOCR; it turns out that the KeraOCR cannot recognize handwritten text, and only English language recognition models are supported. This is because KerasOCR is primarily designed to recognize printed text. 

Therefore to allow muti-language handwritten text detections, I have trained two-stage object detection FAST-RCNN models to detect Chinese or English handwriting questions text from images. The dataset connected by myself contains my handwritten text from ChatGPT history questions. The detected text image is cropped, and, with the language, classification results pass into PaddleOCR to load crossposting text extraction model. Then the detected text is then used as an input pass to the ChatGPT model to return an answer. The program will output intermediate step results and save the final question&answer to a pdf file.
## Learning Objectives

On completion, students will be able to:

Domain Knowledge
 - Understand AI / machine learning terminology
 - Understand deep learning opportunities and limitations
 - Understand different types of deep learning models

Prototyping Skills
 - Implement deep learning models in Python
 - Prepare data for model training
 - Select and train suitable models for different use cases (video & timeseries)
 - Embed AI on sensor devices, such as a mobile phone or a microcontroller.

Collaboration
 - Document and share project information to support reproducible research
 - Provide peer feedback to fellow students on project work
 - Present design decisions and prototypes to receive critical feedback


## Reading List

There is a course reading list under the ReadingLists@UCL facilty which can be accessed here: (https://ucl.rl.talis.com/modules/casa0018.html)

The core text for the module is [TinyML](https://tinymlbook.com/) by Pete Warden and Daniel Situnayake  

We also reference
- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow and Yoshua Bengio and Aaron Courville
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron


## Assessment

(2500 word equiv)
- project build (30%),
- github page - code / docs / photos / video (30%),
- crit (40%)


