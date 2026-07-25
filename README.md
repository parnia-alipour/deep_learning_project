
## datasets:

[detecting fired data](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)

[thalassemia data](https://www.kaggle.com/datasets/abhraghoshcmc/hplc-based-thalassemia-screening-data)

[kidney tumor data](https://figshare.com/articles/dataset/SMC-LUD_Liver_Ultrasound_Dataset_HCC_vs_Hemangioma_/31112716)

[nasa data predicting engine failure time](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)

[persian sentiment](https://github.com/phosseini/SentiPers/tree/master/data)
----

## How you can use my AI models and labs?:

(Download the datasets from the section above)

1. **after training is completed in the labs, copy and paste the code into a Python file**

2. **it's preferable to keep all the code as is, except for removing model.fit() and any plotting code (plt.show(), plt.figure(), etc.).**

3. **make sure you've saved your model,load your saved model using `load_model()`:**
```python

from tensorflow.keras.models import load_model
model=load_model("name_AI.keras",compile=False)

```
4. **apply a `for` or `while` loop to the code so you can test multiple times and get results**


---


AI for detecting breast cancer from images===[BreastMNIST](https://github.com/parnia-alipour/deep_learning_project/blob/master/BreastMNIST.ipynb)

AI for persian to english language translation===(Sequence-to-sequence)[Translator](https://github.com/parnia-alipour/deep_learning_project/blob/master/Translator.ipynb)

َََAI for predicting Bitcoin prices (trained with the LSTM algorithm)===[BTC](https://github.com/parnia-alipour/deep_learning_project/blob/master/BTC.ipynb)

AI for thalassemia disease prediction using a simple dense architecture only===[thalassemia](https://github.com/parnia-alipour/deep_learning_project/blob/master/thalassemia_prediction.ipynb)

AI for predicting engine failure time (using all 4 datasets with different scenarios)===[nasa](https://github.com/parnia-alipour/deep_learning_project/blob/master/NASA_engine_rul_prediction.ipynb)

AI for kidney tumor disease detection (implemented conv2D)===[kidney_tumor](https://github.com/parnia-alipour/deep_learning_project/blob/master/kidney_tumor_prediction.ipynb)

AI for predicting the location of coronary artery blockage in medical images (using segmentation masks and a resNet34 model without pre trained weights)===[heart](https://github.com/parnia-alipour/deep_learning_project/blob/master/Heart_disease_diagnosis.ipynb)

AI for detecting fire using the YOLOv8s model===[fire_detection](https://github.com/parnia-alipour/deep_learning_project/blob/master/Fire_detection.ipynb)


### In deep learning, what happens?


![demo](G.gif)

[Site address](https://playground.tensorflow.org/#activation=tanh&batchSize=18&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=10&networkShape=4,4,2&seed=0.66940&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
