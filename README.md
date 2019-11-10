# AudioClassification

My Final Capstone Project from the Galvanize Data Science Immersive

Goals
-----------------------
Having been a recording engineer prior to enrolling in Galvanize's Data Science Immersive, I really wanted to use my new found data science skills to solve a problem I often encountered in the Post Production Audio world, poorly organized and labelled sound effect libraries. I set out to solve that problem by creating an audio classification model to help organize and tag  Sound FX directories/libraries.

Data
-----------------------

Trained a convolutional and recurrent neural network on about 10,000 labelled audio files from the freesound audio tagging challenge belonging to 41 different classes.

Results
-----------------------
After extensive training, both models acheived about 67% validation accuracy. And while both models did well on data from the freesound audio tagging challenge, I worried they wouldn't generalize well to large sound FX libraries with far more than 41 different types of sound FX.

Luckily, I found the IBM Developer Model Asset Exchange: Audio Classifier https://github.com/IBM/MAX-Audio-Classifier

It is a very robust audio classifier trained on the google audio set, which consists of 5.8 thousand hours of labelled youtube audio falling into 527 classes.

With their docker image:

$ docker run -it -p 5000:5000 codait/max-audio-classifier

I was able to write a script (ClassifyAndTag.py) that takes a directory of audio files, sends them to the docker image running the IBM Audio Classifier to get it's top classifications and adds them as metadata tags.

This script has improved my own Sound FX Library's organization and metadata tags immensely. If any audio engineers out there would like help running this on their own Sound FX Libraries, please reach out.
