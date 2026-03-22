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

BYU - Locating Bacterial Flagellar Motors 2025


1st Place - 3D U-Net + Quantile Thresholding


CatBoost | LGBM | XGB | Churn Prediction | 0.912


Advanced EDA Technique


Predict Customer Churn


Edited

Image Matching Challenge 2023 - Tom


View Active Events

Skip to
content
Kaggle
Kaggle uses cookies from Google to deliver and enhance the quality of its services and to analyze traffic.
Learn more
OK, Got it.
1st Place - 3D U-Net + Quantile Thresholding — BYU - Locating Bacterial Flagellar Motors 2025
1st Place - 3D U-Net + Quantile Thresholding

BYU - Locating Bacterial Flagellar Motors 2025

Solution Writeup · 1st place · Jun 30, 2025

Thanks to BYU and Kaggle for hosting this competition. It was nice to have another well-run tomography competition and the hosts were awesome. I can't believe the result!

TLDR
My solution uses a 3D U-Net trained with heavy augmentations and auxiliary loss functions. During inference, I rank each tomogram based on the max predicted pixel value and use quantile thresholding to determine if a motor is present.

Cross Validation
For validating models, the competition data is split into 4 folds. Local CV strongly correlates with the LB up to about 0.93. Beyond that, I used the public LB for validation. It was important to use quantile thresholding to get reliable feedback from the LB. More on this in the post-processing section.

Preprocessing
Tomograms from the competition data and the CryoET Data Portal are used to create a training set. Each tomogram is resized to (128, 704, 704) using scipy.ndimage.zoom(), and tomograms without motors are discarded. As others noted, the competition data is quite noisy, so Napari was used to manually add missing motors. I will add the updated data here.

For the labels, I use a Gaussian heat map centered on each motor. Similar to @bloodaxe and @christofhenkel's solution in the CZII competition, the resolution of the heatmap is reduced by 8x. This works especially well for this competition as there is a high tolerance for distance error in the metric. This means that predicting the exact pixel is not as important as predicting motor presence. If you are not convinced, the following plot shows roughly how much error is allowed around each motor when voxel spacing equals 10.

tomogram_image

Model
The model is a 3D U-Net (sort of). The encoder is a pre-trained ResNet200 from Kenoshara’s repository here. For most experiments, I used the ResNet101 variant, but increasing the capacity of the encoder yields better performance. In addition, stochastic dropout is applied for regularization, and gradient checkpointing is used to reduce vRAM usage during training. The decoder uses a single deconvolution block before the segmentation head.

model_image

Loss
The model is trained using SmoothBCE loss with 3 contributions. The main segmentation head predicts the output logits, a deep supervision head is applied to the second last feature map, and a max pooled loss (kernel size and stride of 4) is applied on the main segmentation head. Moreover, the pooled loss encourages high probabilities around the motor region, while reducing the penalty for small localization errors.

loss_image

Augmentations
Heavy augmentations enabled training for 400 epochs without overfitting. Although, I could probably have trained longer there was no change in the public LB scores beyond 250 epochs.

Mixup (100%)
Rescale/Zoom (100%)
Rotate90/180/270 (100%)
Axis Flips (100%)
Axis Swap (100%)
Coarse Dropout (50%)
Color inversion (25%)
Simple Cutmix (15%)
Loading tomograms from disk is slow, which limits the time for augmentations on the CPU. To address this, all augmentations but rescaling are applied on the GPU. To keep rescaling as fast as possible scipy.ndimage.zoom(..., order=0) is used.

Inference
Initially, the same preprocessing pipeline was applied during inference. This worked well, but it was 4x faster to match the patch height and width, and only slide over the depth. This allows more time for TTA and a very high overlap (0.875). Both approaches scored about the same, but my final solution uses the latter.

All edge predictions are down-weighted using the roi_weight_map parameter. The middle 40% of the logits are weighted as 1.0 and other logits are weighted as 0.001 when aggregating the sliding window.

Ensembling
The final submission uses an 8-seed ensemble. Sigmoid is applied to each model output and the logits are summed. Inference takes ~10 hrs.

Postprocessing
Like many others, I found that fixed thresholds were unstable. Instead, I use quantile thresholding to determine motor presence.

To apply this, all tomograms are ranked based on their max predicted pixel value. Then, predictions for the lowest quantile are removed. I tuned the quantile on the public LB and then prayed to the Kaggle gods that the private LB was similar. On the public LB the optimal threshold was 0.565 and on private it was 0.560.

LB_image

Final Note
Thanks for reading, and thanks to everyone who showed their appreciation for the external dataset.

External data here
Github repository here
Metadata here

Happy Kaggling!

Author
Bartley
brendanartley


Share

27

12
58 Comments
6 appreciation comments
Hotness
 
Comment here. Be patient, be friendly, and focus on ideas. We're all here to learn and improve!
This comment will be made public once posted.


Post Comment
Jeroen Cottaar
Posted 9 months ago

· 27th in this Competition

Congratulations, and thanks for the detailed writeup!

I actually spent my first month on this competition on a very similar 3D UNet approach, building from my CZII competition solution. A very simple first version already got to ~0.85 CV in the first few days, but never scored above 0.2 on the leaderboard (and usually 0.0). I spent a month trying to figure out what was going on, but failed and fell back to YOLO.

Did you face anything similar at any point? I really want to know what was going on…


Reply

React
IAmParadox
Posted 9 months ago

· 17th in this Competition

Yup, myself and @sersasj as well faced similar issues independently while trying to work with 3D-UNet. The best I could get on the public LB was around ~0.22 and for sersasj it was around ~0.5.


Reply

React

3 more replies
Profile picture for wym2024
Profile picture for MLArt
Profile picture for Bartley
Lonnie Wibberding
Posted 7 months ago

Thanks for the write up and congratulations on the win. As someone fairly new to data science I appreciate people like you who share their techniques to help me learn faster.


Reply

React
Sadeep Dilshan Kasthuriarachchi
Posted 9 months ago

Congratulations!!, and particularly thanks for your effort in providing the external dataset—made a significant impact on the entire community. This recognition is well-deserved, and your contribution truly set the stage for success.❕


Reply

React
Bhavya Garg
Posted 9 months ago

· 113th in this Competition

Congratulations!! A lot to learn from you🙂


Reply

React
Guillermo Perez G
Posted 9 months ago

· 645th in this Competition

Congratulations! Your merit is greater for competing alone, and with such close scores. I wonder if it's possible to experiment with your work in a Jupyter notebook and Anaconda?
All the best


Reply

React
Charvak Upadhyay
Posted 9 months ago

Massive congrats on the 1st place finish—what a brilliant piece of work. The 3D U-Net, all that heavy augmentation, and the smart use of quantile thresholding really paid off. well deserved and super inspiring.


Reply

React
Champ
Posted 9 months ago

· 385th in this Competition

Congrats @brendanartley on winning and thanks for the writeup!
I have a question regarding the quantile thresholding: What exactly is the advantage in your opinion? Because I am not sure I have understood it completly and as far as I understand it, you still have to tune the threshold and the threshold would still change depending on the used model right?
Is it because a quantile threshold gives somewhat of a more smooth threshold on how many predictions are counted?

It will probably be clear, when your inference code is released but still wanted to ask.


Reply

React
tenzy123
Posted 9 months ago

Thanks for this solution,its really helpful.


Reply

React
Victor
Posted 9 months ago

· 13th in this Competition

Hi,
The aux head shouldn't be applied on the last feature map of the encoder as you only use one decoder block?
In the Loss image you shared, it is applied on the decoder block instead.


Reply

React
alex5051
Posted 9 months ago

· 89th in this Competition

Congrutulations,thanks for your solution.It's amazing.


Reply

React
Jiahao Liu
Posted 9 months ago

Congratulations, and thanks for detailed writeup…!


Reply

React
Sarah Arshad
Posted 9 months ago

Congratulations, and thanks for detailed writeup…!


Reply

React
Caleb Yenusah
Posted 9 months ago

· 431st in this Competition

Congrats! Is it possible to share this solution's training and inference source code?


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Cheers @calebyenusah.

I just released the training code here. The inference pipeline is coming soon.


Reply

2

1
Sweksha Sinha
Posted 5 months ago

@brendanartley would love to get the inference pipeline to understand your approach to the problem


Reply

React
Cyrus
Posted 9 months ago

· 26th in this Competition

Congrats! I think the thresholding trick you have done has a lot of weight in the final results, you did a great job nailing that!


Reply

React
tennogh
Posted 9 months ago

· 686th in this Competition

Hi and congrats, I have some questions seeing as I used a similar approach but didn't get past .7 LB:

Did you use all of the external data? I found that the tomograms with the smallest voxel spacings hindered my models significantly
What crop size did you use for training? I tried to go as large as possible but I think it was a mistake.
How long did training and especially validation take? I had to reduce the size of my validation sets because it was difficult to train for a lot of epochs
Anectodally I estimated that around 35% of the test samples were positive, and settled for a threshold around the 58th percentile.


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Hi @tennogh,

I used all tomograms.

The crop size was (64, 674, 674) - I think you were on the right track with this.

Small models trained in 2 - 8 hours, while the final ensemble models took 35 - 40 hours each.


Reply

1
Ángel Jacinto Sánchez Ruiz
Posted 9 months ago

· 640th in this Competition

Congratulations. Thanks for share.

Rotate90/180/270 (100%)

So no free rotations at all. Do you think apply free rotations as I did could been perjudicial because the small size of the actual target, the motor?


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Hi @sacuscreed, thanks for your comment.

Its hard to say without validation, but some pipelines will be more susceptible to noise than others and maybe free rotations were too much for yours. I did not try them myself, but maybe I should have!


Reply

1
Ian Pan
Posted 9 months ago

Congratulations! Great solution.


Reply

React
wym2024
Posted 9 months ago

Hello Brendanartley, your work is awesome! I have some questions, I want to ask you, as follows:
Do you have prior knowledge of medical images, so can you correct the label manually?
Did you mention discarding negative samples because you think the results should focus more on recalls or because of experience?
Thank you for your reply!


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Hi @wym2024, thanks for the comment.

My only experience with tomograms or medical images comes from ML competitions like the CZII one here. I discarded negative samples due to concerns about missed annotations, but in hindsight this had no impact on the private LB.


Reply

React
wym2024
Posted 9 months ago

Thank you for your reply!


Reply

React
Vasilis
Posted 9 months ago

· 192nd in this Competition

Congratulations for the rank 1 solution and extra congratulations for sharing the extra data! I saw you get all tomographs at 128, 704, 704. How do you downsample the Z dim? if they are like 300 or 500 slices do you keep one every 3 for example (and i guess a bit more around the mottor?


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Thanks @vasileioscharatsidis. All tomograms were resized with scipy.ndimage.zoom(). I have updated the preprocessing code here.


Reply

React
Carlos Pérez
Posted 9 months ago

· 43rd in this Competition

Congrats in your solo gold medal! I've seen you in past competitions and I knew you were going to make it soon! Happy Kaggling!


Reply

React
NarayanNarayan
Posted 9 months ago

· 197th in this Competition

Congratulations. I'm so glad to see the winning solution isnt YOLO. Also, Hats off to you for the dataset you provided. Kudos to you again


Reply

React
siwooyong
Posted 9 months ago

· 29th in this Competition

Congratulations!
What patch size, label radius did you use?


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Thanks @siwooyong!

The patch size was (64, 704, 704) and the kernel_size was 7 (on the 8x downsampled label). Here is a label overlaid on a tomogram to show the scale.

Cropper


Reply

1
ynhuhu
Posted 9 months ago

· 444th in this Competition

Congratulations, PrivateLB has a slight improvement over PublicLB, I think it is due to your data processing and 3D model. Can the model run on Kaggle?


Reply

React
Bartley
Topic Author
Posted 9 months ago

· 1st in this Competition

Hi @ynhuhu, thanks for the comment.

The final model and patch size are too large to train on Kaggle, but there is more than enough time for inference.


Reply

React
yyyy0201
Posted 9 months ago

· 42nd in this Competition

Thank you for a very good solution and having excellent generalizability, learning a lot!🥰


Reply

React
aaaditya56
Posted 9 months ago

Congratulations! Thank you for this external dataset, as a beginner in Data Science and Machine Learning, there was a lot of learning when reading this.


Reply

React
