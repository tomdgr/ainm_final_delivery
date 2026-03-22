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

RSNA 2024 Lumbar Spine Degenerative Classification


1st place solution


CZII - CryoET Object Identification


1st place solution [segmentation with partly U-NET and ensembling part]


FathomNet 2025 @ CVPR-FGVC


Edited

Image Matching Challenge 2023 - Tom


View Active Events

Skip to
content
Kaggle
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
1st place solution — RSNA 2024 Lumbar Spine Degenerative Classification
1st place solution

RSNA 2024 Lumbar Spine Degenerative Classification

Solution Writeup · 1st place · Oct 28, 2024

First of all, I would like to express my sincere gratitude to the competition host and the Kaggle staff for organizing such a fascinating competition. I thoroughly enjoyed this competition and learned a great deal in the process!

Furthermore, I'd like to thank @hengck23 and @brendanartley. @hengck23 's discussion and notebook were the starting point for my solution, and @brendanartley 's this dataset helped my coordinate prediction models. I was deeply impressed by their contributions to the Kaggle community.

This is my first solution write-up, so please feel free to leave any comments or suggestions for improvement!

Summary
My solution is 2 stage approach, creating test_label_coordinates.csv and predicting severity. Furthermore, I separated 1st stage into instance_number prediction and coordinate prediction. Therefore I prepared 3 type of model, instance_number prediction model, coordinate prediction model and severity prediction model. The pipeline is shown in the following figure.

pipeline

1st stage: test_label_coordinates creation
In the 1st stage, I use 2 type of models, 3D convolution model and 2D convolution model. These models are very simple, encoder + level-separated heads.

instance_number prediction (sagittal)
In this part, I used simple 3D ConvNeXt to predict instance_number for each level. Data that is fed into models is just normalized from 0 to 1, sorted by dicom's metadata and padded 32 to depth direction to align shape. Data preprocessing is shown in the following figure (scs example).

scs_volume_example

In training models, I trained models 2 tasks, regression and classification, and I used L1 Loss and Cross Entropy Loss respectively. In the classification task, these heads output (bs, 32) shape logits for each level. In the regression task, these heads output (bs, 3) shape vectors for each level. (bs, 3) shape vector means (x, y, z) and I used z for depth prediction, (x, y) were used auxiliary loss. In the regression task, I normalized coordinate labels 0 to 1 for stabilizing models during training. Concretely, I used label (x', y', z') = (x/width, y/height, z/32). The model architecture is shown in the following image (scs example). I implemented 3D ConvNeXt for this task (to implement 3D ConvNeXt, I referred to this repo).

instance_number_prediction_scs_example

The results of instance_number prediction models are shown in the following table (sagt2, scs).

model/error	+-0	+-1	+-2	error>+-2
cls	71.08%	27.04%	1.43%	0.44%
reg	67.48%	30.59%	1.61%	0.31%
I ensembled this 2 type of predictions using median for each level (actually I used 5 fold for each task).

coordinate prediction(sagittal)
In coordinate prediction task, I used 2d encoder + level-separated heads, almost same as instance_number regression model. Data is 3 channel image. The image is picked up using median of instance_number of L1 ~ S1. Then the data processed normalization and reshaping (512x512). Labels are (x', y') = (x/width, y/height)
for each level, same as instance_number regression, and also I used L1 loss. The model architecture is shown in the following figure.

coordinate_prediction_model_scs_example

I used ConvNeXt-base and Efficientnet-v2-l for this task. Before I train these models, I trained these models using @brendanartley 's dataset. These pretrained models were slightly better than pretrained models that were trained using imagenet. I ensembled these predictions using mean.

instance_number calculation and coordinate prediction (axial)
For instance_number prediction of axial, I borrowed @hengck23 's method (notebook is here). Then I predicted coordinates of axial, same as coordinate prediction for sagittal.

2nd stage: severity prediction
For the 2nd stage, I attempted simple 2.5D model and MIL. 2.5D model can be implemented easily, however, MIL was better than simple 2.5D at final.

preprocessing
Cropping method
My preprocessing strategy is cropping. For example, I cropped sagt2 image for scs;

pick up 5 images (center is an image that was assigned instance_number)
reshape 512x512
crop images using the coordinate (96 pix left and 32 pix right from coordinate x, 40 pix upper and 40 pix lower from coordinate y)
After cropping an image, the image can be like the figure below (sagt2 for scs, L1/L2).

scs_cropped_image

sagt2, sagt1 and axial were cropped for each classification task. The following tables are representing cropping range from (x, y) coordinate.

for scs

type	left	right	upper	lower
sagt2	96	32	40	40
axial	96	96	96	96
Note that when I crop images from axial, I picked up left or right subarticular stenosis coordinate randomly, and for adjusting cropping point, I added +-20 to ss coordinate x. As a result, cropping range can be like the following figure (the example is right ss coordinate x + 20).

axial_for_scs_cropping

for nfn

type	left	right	upper	lower
sagt1 (both left and right)	96	64	32	32
axial (right)	144	48	96	96
axial (left)	48	144	96	96
for ss

type	left	right	upper	lower
axial (right)	144	48	96	96
axial (left)	48	144	96	96
The following image is the range of cropping axial for right subarticular stenosis.

axial_cropping_for_ss_right

data augmentations
I used several augmentations like below;

Before cropping

random shift of coordinate x and y (-10~+10 pix)
random shift of instance_number (-2~+2. shifting probability was decided error probability of each instance_number prediction models)
After cropping

RandomBrightnessContrast(p=0.25)
ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.1), rotate_limit=20, p=0.5)
Especially, random shift of instance_number was crucial for robustness of error of 1st stage.

model architecture
My model architectures are shown in following figures.
[EDITED] I have updated the figure illustrating the model architecture to correct an error in the previous version. aux_attn_score in the code below is fed into cross entropy loss directly.

severity_prediction_model_scs_fixed
severity_prediction_model_ss_fixed

I used ConvNeXt-small and Efficientnet-v2-s as the encoder. After implementing Attention-based MIL, my public LB score was improved from 0.37 -> 0.35. Then, adding bi-LSTM, aux losses and ensembling improve my score from 0.35 to 0.33. bi-LSTM + Attention-based MIL was implemented like below.

class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)

        return weighted_instances, aux_attn_scores
what didn't work
MAMBA and Self-Attention instead of bi-LSTM
sharing weight between aux_attention layer and attention layer
sagt1 image for scs, sagt1 and sagt2 image for ss, sagt2 image for nfn
long epochs (I used 7 epochs for convnext-small and 14 epochs for efficientnet-v2-s)
large models (convnext-large < convnext-base < convnext-small in my experiments)
vision transformers (I think this was my problem. but convolution models were better than vits in my experiments)
code
All training code is implemented in google colaboratory. All models are used for this inference code. Following links are pairs of model name & training notebook link. You can check these model name in the inference code.

You can train on google colaboratory environment with T4 + high memory.

models
instance number prediction models (SCS)
scs_depth_1024_ssr: notebook
scs_depth: notebook
scs_depth_1024_ssr_l1: notebook
instance number prediction models (NFN)
nfn_depth_1024_ssr: notebook
nfn_depth: notebook
nfn_depth_1024_ssr_l1: notebook
coordinate prediction models (SCS)
scs_detect_pre: notebook
scs_detect_pre_effv2l: notebook
coordinate prediction models (NFN)
nfn_detect_pre: notebook
nfn_detect_pre_effv2l: notebook
coordinate prediction models (SS)
ss_detect: notebook
severity prediction models (SCS)
_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp: notebook
_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_effv2s_for_exp: notebook
severity prediction models (NFN)
nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_convnext-s: notebook
nfn_classify_5ch_axsagt1-lstm-mil_auxloss_auxdepth_2shift_effv2s: notebook
severity prediction models (SS)
ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_effv2s: notebook
ss_classify_5ch_ax-lstm-mil_auxloss_auxdepth_convnext-s: notebook
coordinate pretrained models
this notebook is pre-training code for coordinate prediction models with this dataset. model checkpoints are in this dataset

Author
NANACHI
wadakoki


Share

24

7
19 Comments
6 appreciation comments
Hotness
 
Comment here. Be patient, be friendly, and focus on ideas. We're all here to learn and improve!
This comment will be made public once posted.


Post Comment
Bartley
Posted a year ago

· 2nd in this Competition

Great first write up and congrats on the strong finish!

Did the Attention-based MIL improvements improve your local CV as much as the LB? Also, do you know how much of the improvement was due to the auxillary loss?


Reply

1
NANACHI
Topic Author
Posted a year ago

· 1st in this Competition

Thank you Bartley! Also, congratulations on winning a gold medal!

Attention-based MIL improvements improve your local CV as much as the LB?

Improvements of my local cv by MIL are smaller than the improvement of LB. I lost log of actual improvements of cv, but I remember the improvements are less than 0.012. However public LB improved 0.3729 -> 0.3588 (diff is 0.0141) and private LB improved 0.4259 -> 0.4062 (diff is 0.0197).

do you know how much of the improvement was due to the auxillary loss?

Actually, I submitted bi-LSTM+aux loss model at the same time, so I don't have the result of the improvements on LB by aux losses. The table below is the result of validation data of scs (same seed, same preprocess, same hyper params and same architecture with/without aux head)

fold	without aux	with aux
0	0.254	0.240
1	0.283	0.254
2	0.267	0.254
3	0.264	0.252
4	0.244	0.261
mean	0.2624	0.2522

Reply

React
Bartley
Posted a year ago

· 2nd in this Competition

Interesting, the improvement for aux loss is a more significant than I thought. Thanks!


Reply

1
samu2505
Posted a year ago

· 438th in this Competition

In the classification task, these heads output (bs, 32) shape logits for each level.; what targets were used since there are 5 levels for each image?


Reply

React
Mingjie Wang
Posted a year ago

Congratulations! Could you explain in detail why auxiliary loss is effective? I understand it should be nearly equivalent to the original architecture.


Reply

React
NANACHI
Topic Author
Posted a year ago

· 1st in this Competition

Thank you!

In my opinion, auxiliary loss makes it possible to add context to feature vectors. For example, I added depth aux head after bi-LSTM. This is because I wanted to incorporate the context of which slice the annotators were focusing on into the feature vectors output by the LSTM. I think this allows the main attention layer to function more effectively.


Reply

React
tim062912
Posted a year ago

Could you kindly provide the necessary file? Without it, we are unable to run the notebook.


Reply

React
Switch9527
Posted a year ago

Hello! I’d like to ask about the three model notebooks in the instance number prediction models (SCS). I assume that these three models aim to achieve the same objective, with the only difference being their model structures—is that correct? Will these three models be integrated in the inference code? If my understanding is incorrect, could you please point it out? Thank you! Apologies for my slow progress with reading the code; so far, I’ve only reviewed these three notebooks.


Reply

React
homiecal
Posted a year ago

Hi @wadakoki what hardware did you use for the competition? Was it just the free quota on Google Colab Notebooks?


Reply

React
Kenny
Posted a year ago

thanks for sharing. Great solution and explanation.


Reply

React
Malabh Bakshi
Posted a year ago

· 419th in this Competition

Amazing Presentation @wadakoki


Reply

React
SUZUKI SEIYA
Posted a year ago

You're number one!


Reply

React
Yisak Birhanu Bule
Posted a year ago

congra man you deserve it


Reply

React
dragon zhang
Posted a year ago

· 152nd in this Competition

thanks for sharing. Piece of artist solution cake.


Reply

React

Appreciation (6)
Vinothkumar Sekar
Posted a year ago

Good work.!!! 👍

Rudrersh Asagodu
Posted a year ago

Congratulations!

Muhammed Tausif
Posted a year ago

Thanks for sharing your approach.

Usaid Ahmad
Posted a year ago

good work!

This comment has been deleted.

This comment has been deleted.

