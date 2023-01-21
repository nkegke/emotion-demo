# Real-Time Emotion Recognition Demo

This is a demo which takes input from webcam and predicts the following emotions:

* **Face**: Neutral, Happiness, Sadness, Surprise, Frustration, Fear, Disgust

* **Body**: Positive, Neutral, Negative

Face     |  Body
:-------:|:----------:
<img src="https://github.com/nkegke/files/blob/main/demo/face.gif " style="width: 30vw;"/> | <img src="https://github.com/nkegke/files/blob/main/demo/body.gif" style="width: 30vw;"/>

## Prerequisites
Install the required packages below:

```
pip3 install torch torchvision torchaudio
pip install opencv-python
pip install mediapipe
pip install gdown
```

## How to run

Webcam Input:
```
python demo.py INPUT
```
where ```INPUT``` = { ```face``` , ```body``` }.

Optionally, apply [medical face mask](https://github.com/nkegke/medical-face-mask-applier), by running:
```
python demo.py INPUT mask
```

## How it works

<img src="https://github.com/nkegke/files/blob/main/demo/demo.png"/>

The Face Mesh detection step is optional for body input.

The face MobileNetV2 model is trained on the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset, whereas the body MobileNetV2 model is trained on the [BoLD](https://cydar.ist.psu.edu/emotionchallenge/dataset.php) dataset.
