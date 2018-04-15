## Real-time image saliency


_See what your classifier is looking at!_ [[PAPER]](https://arxiv.org/abs/1705.07857)

![UI](minsaliency/border_collie.jpg)


#### Real-time saliency view
Run `python real_time_saliency.py` to perform the saliency detection on the video feed from your webcam. 
You can choose the class to visualise (1000 ImageNet classes) as well as the confidence level - low
confidence will highlight anything that resembles or is related to the target class, while higher confidence
will only show the most salient parts. 
  
 The model runs on a CPU by default and achieves about 5 frames per 
 second on my MacBook Pro (and over 150 frames per second on a GPU).

#### Training

Run `python saliency_train.py` to start the training. By default it will train the model to perform the saliency detection on the ImageNet dataset for the resnet50 classifier, but you can choose your own dataset/classifier combination. 
You will need PyTorch wich cuda support, the training will be performed on all your GPUs in parallel. I also advide to run the script from iTerm 2 terminal so that you can see the images during traning.  

![UI](minsaliency/training.jpg)

#### Using pretrained model

```python
from saliency_eval import get_pretrained_saliency_fn

sal_fn = get_pretrained_saliency_fn()

# get the saliency map (see get_pretrained_saliency_fn doc for details)
sal_map = sal_fn(images, selectors)

```

#### Requirements

`pip install -r requirements.txt`

Also, in case you don't have OpenCV3 installed run:

`pip install opencv-contrib-python`