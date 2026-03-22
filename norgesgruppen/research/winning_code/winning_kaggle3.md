Skip to
content
Kaggle

Create
Home

Competitions

Datasets

Models

Benchmarks

Game Arena

Code

Discussions

Learn

More

Your Work


Viewed

CZII - CryoET Object Identification


1st place solution [segmentation with partly U-NET and ensembling part]


FathomNet 2025 @ CVPR-FGVC


1st Place Solution for the CVPR 2025 FGVC Competition


Only adding "database" to the coco/flickr Urls, allows to display FathomNet 2025 images.


Edited

Image Matching Challenge 2023 - Tom


View Active Events

Skip to
content
Kaggle
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
1st place solution [segmentation with partly U-NET and ensembling part] — CZII - CryoET Object Identification
1st place solution [segmentation with partly U-NET and ensembling part]

CZII - CryoET Object Identification

Solution Writeup · 1st place · Feb 24, 2025

Thanks to kaggle and everyone involved for hosting this exciting competition. It was a great learning experience and it was very interesting to see how much of our computer vision experience could also be applied to 3D imaging. Thanks to @bloodaxe for this great team experience.

TLDR
The solution in an ensemble of segmentation (3D Unets with ResNet & B3 encoders) and object detection models (SegResNet and DynUnet backbones) from MONAI. We also used MONAI for augmentations, and exported models via jit or TensorRT, which gave 200% speedup increase and enabled us to have a slightly larger ensemble. We did not use any external or simulated data!

This post covers the segmentation based approach and ensembling. For object detection part see @bloodaxe writeup: 1st place solution [Object Detection Part]

Cross validation
For segmentation approach 7 folds were used, simply by splitting by experiment. Using mean of f4-score of all 7 folds had some good correlation with LB. During model training I optimized individual class thresholds at the end of each epoch by simple grid-search on the validation experiment. After all 7 folds were trained we could re-calibrate thresholds, by taking OOF predictions. And fitting the threshold for one fold on the predictions of the other 6. Then we average the resulting f4-curves and take the best threshold.

Data preprocessing/ augmentations
3D images were normalized by standard normalization, i.e. for each 630x630x184 image, we substract mean and devide by standard deviation before splitting the images into patches.
Since models are trained from scratch, augmentations were essential to prevent overfitting.
We used RandomCrop, Flip on each axis, and rotation, which all are available with MONAI. Additionally I used my own implementation of MixUp which was highly effective to train longer and prevent overfitting.

Model
Modelling was quite a ride in this competition. I started with simple UNET with having 3D gaussian balls as segmentation target. For diversity I also tried object detection example from MONAI and realized its working very well out of the box. But when analyzing the different output feature maps and trying to isolate the perfomance gain over gaussian heatmap based segmentation I realized where the advantage was and adjusted my segmentation model accordingly. I learned that

the penultimate feature map has a higher accuracy than the last one, which is surprising at first
gains from box regression are negligable, as particles from the same type have mostly same size anyways.
Hence its sufficient to have a pixel-wise loss on the penultimate feature map output. I.e. use a partly UNET. The gaussian heatmap is not needed when suppressing background with a low class weight, and just use single pixels as targets. Using the same approach on lower level outputs (= deep supervision) is possible, but does not provide much gain. Input for the segmentation models are 96x96x96 image patches and the loss is calculated on the 48x48x48 output.



In general, we observed that relatively small models work really and the design of loss is the most important aspect. We used MONAIs FlexibleUnet with backbones resnet34 and efficientnet-b3. Six checkpoints of this architecture would finish under 2h and score 7th place on LB.

Training procedure
We used 7 classes (incl background) and weighted CrossEntropy as loss. Notably keeping beta-amylase as a class although it is not scored is quite helpful as model learns to differentiate beta-galactosidase from it. To account for low number of positive pixels, positive pixels are weighted by 256 and background has weight 1. Models were trained with a cosine learning rate schedule with peak LR of 0.001, mixed precision and an effective batch size of 32 samples. Training is based on Random crops and for validation the single experiment image is divided into patches and stored in RAM.

Ensembling
Ensembling was very challenging, as our two approaches are quite diverse. While in theory predictions from the segmentation models can be ensembled with feature map outputs of the object detection model before runnning the object detection postprocessing, in practice those feature maps have a very different distribution due to difference architectures and loss functions.



We were eager to find an elegant way to fix this scaling issue as we saw the potential of a possible ensemble. The scaling of our best submission works as following. To combine predictions A with predictions B, for each class sort all pixel values for A and B and replace values of B with the corresponding values of A of same rank. In code:



This results in both predictions having the same distribution, and hence we could simple blend the feature maps before performing object de
tection task.



What did not work
Using supplemental data (external or simulated)
Other augs
Other losses (Tversky, Dice)
Thanks for reading.

Edits:
training code: https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation
inference kernel: https://www.kaggle.com/code/christofhenkel/cryo-et-1st-place-solution?scriptVersionId=223259615

Authors
Eugene Khvedchenya
bloodaxe

Dieter
christofhenkel


Share

12

9
25 Comments
4 appreciation comments
Hotness
 
Comment here. Be patient, be friendly, and focus on ideas. We're all here to learn and improve!
This comment will be made public once posted.


Post Comment
Yassine Alouini
Posted a year ago

Well done and congrats. Can you provide some details on the training infrastructure please (how many GPUs, how long it trained, any issues on the losses…)?


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

I trained 6 checkpoints, each takes 2h on a A100 with bf16 mixed-precision. For my models mixup was crucial which more than doubles the number of epochs for training.


Reply

React
Yassine Alouini
Posted a year ago

Thanks for the details. bf16 didn't introduce any instabilities?


Reply

React
Đăng Nguyễn Hồng
Posted a year ago

· 70th in this Competition

Congratulations and thank you so much for your write-up, as always!
The CV scheme truly impressive. The ensemble approach seems like a form of histogram matching, isn't it?
For single model, it's remarkable how your team and the top solutions have made everything work so seamlessly, especially when I struggled to achieve just a poorer result with a "sound-like similar" setup. There are likely many implementation details that come naturally to you but still pose a bit of a challenge for me.
Really looking forward to seeing your train/infer code so I can better understanding of our problems and failures!


Reply

1
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

I was already 1st place on public LB two months ago, with a simple 3D-Unet with gaussian balls as target. Score was around 0.74 back then. It took us 2 month of hard work to improve and refine so the final solution can look "seamlessly" as you say.


Reply

2
Đăng Nguyễn Hồng
Posted a year ago

· 70th in this Competition

Thanks! You've really motivated us 😀
0.74 to 0.765+ for single model, i mean all things were improved, but beside design of loss is the most important aspect as you mentioned, could you share your thoughts on a few other key factors that contributed to this improvement? A short ranked list would be incredibly helpful.


Reply

React
Adrian Tif
Posted a year ago

Congratulations for the win! What did you mean by gaussian balls as target? Is it some sort of baseline?


Reply

React
Volodymyr Pivoshenko 🇺🇦
Posted a year ago

@christofhenkel congratulations! very interesting solution!


Reply

React
work work
Posted a year ago

· 559th in this Competition

congratulations , not fully understand the solution , but i learn a lot from this competition


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

If you have specific points where I should be clearer, I am happy to edit the post and elaborate more. My aim is that everybody can understand the solution.


Reply

React
Bartley
Posted a year ago

· 13th in this Competition

Thanks for the detailed write-up and nice ensemble strategy.

How was your version of MixUp different from the usual implementation?


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

Its not much different to other implementations. Some mix rather the losses, I like mixing the targets better. I like a simple pure torch implementation so you can have it as part of the model on GPU. And its flexible for 1D, 2D, 3D.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import Beta

class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Z=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        X_coeffs = coeffs.view((-1,) + (1,)*(X.ndim-1))
        Y_coeffs = coeffs.view((-1,) + (1,)*(Y.ndim-1))

        X = X_coeffs * X + (1-X_coeffs) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            Y = Y_coeffs * Y + (1 - Y_coeffs) * Y[perm]

        if Z:
            return X, Y, Z

        return X, Y


class Net(nn.Module):

    def __init__(self, cfg):
        super(Net, self).__init__()

        ...
        self.mixup = Mixup(cfg.mixup_beta)
        ...

    def forward(self, batch):

        x = batch['input']
        y = batch["target"]
        if self.training:
            if torch.rand(1)[0] < self.cfg.mixup_p:
                x, y = self.mixup(x,y)
        ....

Reply

3

1
Yassine Alouini
Posted a year ago

I guess he adapted an existing implementation to work with 3D patches? As far as my knowledge goes, MixUp works with two images with weights alpha and (1-alpha), maybe he did something clever to have images over the temporal dimension?


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

Mixup really just is averaging inputs and targets of a batch with a weight drawn from a Beta-distribution.
Actually we ( @philippsinger and @ilu000 ) coded this four years ago as part of the rainforest competition if I recall correctly. Mixup is really strong for audio. Our implementation was already flexible for arbitrary dimension input and was running on GPU, and I re-used in several competitions since then.


Reply

1
David List
Posted a year ago

· 11th in this Competition

I mean…who thinks of measuring the accuracy of the penultimate feature map? That or the way you managed to do the ensemble. Pretty cool stuff! Thanks for the great explanation!


Reply

React
Sinan Calisir
Posted a year ago

· 32nd in this Competition

Congratulations on the win! Curious to see if the number one spot will come back again shortly 👀

Your solo scores are amazing with no external data. It would be greatly appreciated if you plan to open source your approach. I got even inspired from your solution on: https://github.com/Project-MONAI/tutorials/tree/main/competitions/kaggle/RANZCR/4th_place_solution while learning about MONAI.


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

I guess you will be happy to hear that we will add our solution to MONAI as a tutorial


Reply

2

1

2
Sinan Calisir
Posted a year ago

· 32nd in this Competition

Oh, that's such a great news!! I am looking forward to reading your tutorial when published.


Reply

React
Ma Edward
Posted a year ago

· 24th in this Competition

Congratulations!!!
Would you mind sharing the single model performance in the final ensemble models?


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

for segmentation:

resnet34 backbone: 0.766
resnet34 backbone + deep supervision: 0.767
effnet-b3 backbone: 0.765
combined: 0.778


Reply

React
Ms. Nancy Al Aswad
Posted a year ago

Thanks for sharing 🙏


Reply

React
Ma Edward
Posted a year ago

· 24th in this Competition

Thank you so much for sharing!


Reply

React
Sagar Sarkar
Posted a year ago

Congratulations!! Thank you, it was a fantastic learning opportunity, and your work is greatly appreciated!


Reply

React
Linhan Wang
Posted a year ago

· 34th in this Competition

Thanks for the detailed writeup. Using x2 downsampled output can speed up the model greatly. Could you share how you do postprocessing? If you use maxpooling, can you share the kernel size. I am concerned about downsampling might hurt the localization accuracy.


Reply

React
Tyrian Tong
Posted a year ago

Thanks for the detailed write-up and nice ensemble strategy.💪


Reply

React
Lavanyabanga25
Posted a year ago

Thanks for the detailed write-up and nice ensemble strategy.


Reply

React
work work
Posted a year ago

· 559th in this Competition

@christofhenkel congratulations to be 1 ranking other time, can you share with us what environement that use to develop your solution ( IDE, HARDWARE…)


Reply

React
Dieter
Topic Author
Posted a year ago

· 1st in this Competition

For this competition I switched from jupyter notebook as IDE to using VSCODE, simply because it enables the use of Copilot.


Reply

React
work work
Posted a year ago

· 559th in this Competition

thank you for your reply


Reply

React
Yassine Alouini
Posted a year ago

At some point, we will get cursor/copilot within notebooks I believe. 👌


Reply

React
work work
Posted a year ago

· 559th in this Competition

for pycharm IDE , it is available in professional version not free version


Reply

React

Appreciation (4)
Bhumika Mahajan
Posted a year ago

very informative

Harsaihaj Singh Gill
Posted a year ago

Very informative

Saaransh Gupta
Posted a year ago

Very nice!!!!!!!!!!!!!!

YMJA
Posted a year ago

Interesting🤔

