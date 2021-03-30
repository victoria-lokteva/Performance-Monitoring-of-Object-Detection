# Performance Monitoring of Object Detection

This repository contains the pytorch implementation of the algorithm proposed by authors of the article "Per-frame mAP Prediction for Continuous Performance
Monitoring of Object Detection During Deployment":
https://arxiv.org/abs/2009.08650

The authors introduce a system "alert" designed to raise alarm when the performance of the deployed object detection model drops below a critical treshold. 
This approach allows to assess the performance of object detector without any ground-truth data, hence system performance is evaluated from input itself.
The algorithm uses the feature map received after a series of convolutions in a detector's backbone to extract features for alert system.

![network](https://user-images.githubusercontent.com/74068173/112883073-75e0cd00-90d6-11eb-9748-dd395fe12795.png)

## The alert system algorithm

For each channel of the feature map we apply mean pooling using the next formula:

<img src="https://render.githubusercontent.com/render/math?math=F_{mean} = \frac{\sum_{x=1}^{H} \sum_{y=1}^{W} f(x, y)}{W*H}">

Similarly, for each channel the max pooling is performed:

<img src="https://render.githubusercontent.com/render/math?math=F_{max}= \max_{\substack{x \in [1, H]}} \max_{\substack{y \in [1, W]}} f(x, y)">

Finally, we use the statistical pooling, calculating standard deviations for each channel and concatenating them:

<img src="https://render.githubusercontent.com/render/math?math=F_{std} = std(f_1) \bigoplus std(f_2) \bigoplus ... std(f_N)">

The results of these operations are concatenated:

<img src="https://render.githubusercontent.com/render/math?math=F_{mean\_max\_std} = F_{mean}  \bigoplus F_{max} \bigoplus F_{std}">

The resulting vector is passed as an input to a fully connected network. The network consists of 5 layers, relu is used as an activational function. 
The last layer is activated by sigmoid which gives a binary answer, where 1 stand for failure and 0 stands for success.

The alert system algorithm is illustrated on the flowchart below:

![flowchart](https://user-images.githubusercontent.com/74068173/112883131-8729d980-90d6-11eb-9e62-e15be1b894f0.png)

## Different types of poolings

* **Max Pooling**

Max pooling selects the maximum pixel value of the input matrix.
It allows to retain the most prominent features of the feature map like edges. However, it rejects a sufficient amount of information, 
missing out on some details.
Max pooling is also extremely effective in cases of objects placed on a dark background.

* **Average (Mean) Pooling**

Average pooling calculates the mean of pixel values of the input matrix.
This operation smooths out the image and has a blurring effect retaining the average values of features. However, average pooling might not be able
to extract important features because it tries to take everything into account. Unlike max pooling, it is irrespective of background.

* **Statistical Pooling**

Statistical pooling calculates the standard deviation of pixel values of the input matrix. It shows the diversity of features,
separating objects from background. Statistical pooling strongly activates parts of images with a lot of edges or lines.
