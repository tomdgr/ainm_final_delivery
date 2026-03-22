https://github.com/liaopeiyuan/TransferDet


Thank you for host and competitors.
It is interesting competition for me.

TL;DR
EfficientDetB3 5Fold -> 10 image Test with Pseudo Labeling
EfficientDetB5 All Training(EMA) -> Prediction MultiScaleTraning(1280,768,1024+4Flip) 0.766
EfficientDetB5 All + EfficientDetB4 All + EfficientDetB5 All(With Pseudo) Public 0.7752
Solution
1st Phase
EfficientDetB3 for test 10 images pseudo labeling
Test image is "test source", I want to get test source knowledge

2nd Phase
I used EMA for the ensemble technique in 2nd phase.

AdamW 100epoch 640 batchsize 4(for bn parameter turning)
AdamW 40epoch 1024 batchsize 1(bn layer freeze)
use Cosine Annealing

Augmentation

Mixup
Mosaic
Scale
Hue
Random Brightness
Cutout
GridMask
3rd Phase(Kernel)
Prediction
EfficientDetB5 All + Predict MultiScale(768, 1024, 1280 * 4Flip) -> Pseudo Labeling of All test.

Pseudo Labeling
Traning Parameter as follows

EfficientDet B5 image size 1024
Epoch 5
Use EMA
Mixed Precision(AMP)
Ensemble
EfficientDetB5 All + EfficientDetB4 All + EfficientDetB5 All(With Pseudo) using WBF

Why Shakedown?
failed Threshold turning.(lower threshold is better in private)
When setting threshold 0.1, the score go to 0.695. but I can not get selection.
Use more heavy augmentation for robustness




VinbidData

Thanks to Kaggle and hosts for this very interesting competition with a too challenging dataset. This has been a great collaborative effort. Please give your upvotes to @fatihozturk @socom20 @avsanjay Congrats to The Winners.

TLDR
@fatihozturk explained in his post how was his path through this competition, this post is intended as an overall view of our final solution which achieved 0.354/0..314 PublicLB/PrivateLB. It is worth mentioning that we also had a better solution with 0.330/0.321 PublicLB/PrivateLB (which unfortunately we didn't select). As you all already know the main magic we have used for this competition was our Ensembling procedure, also there were lots of other important findings that we mention in this post.

CV strategy: This was our main challenge, we didn't have a unique validation scheme. Since we started the competition as different teams, we all were using different CV splits to train our models and to build our separated best ensembles. To overcome this difficulty we finally ended up using the public LB as an extra validation dataset. We felt that this strategy was a bit risky, so we also retrained some of our best models using a common validation split and then we ensemble them (this last approach didn't have the best score on the private LB by itself). We think that our last ensemble, using the public LB as a validation source, helped out our final solution to fit better the consensus method used to build the test dataset.

Our Validation Strategy
Our strategy for the final submission can be divided into 3 stages :

Fully Validated Stage
Partially Validated Stage
Not validated Stage (Just validated using the Public LB)
Models, we used for Ensembling
Fully Validated Stage:

Detectron2 Resnet101 (notebook by @corochann, Thanks)
YoloV5
EffDetD2
Partially Validated Stage:

YoloV5 5Folds
Not Validated Stage:

2 EffdetD2 model
3 YoloV5 (1x w/ TTA , 2x w/o TTA)
16 Classes Yolo model
5 Folds YoloV5 by @awsaf49
New Anchor Yolo
Detectron2 Resnet50 (notebook by @corochann, Thanks)
YoloV5 @nxhong93 (https://www.kaggle.com/nxhong93/yolov5-chest-512)
Yolov5 with image size (640)
Ensembling Strategy
To ensemble all the partial models we mainly used @zfturbo's ensembling repository https://github.com/ZFTurbo/Weighted-Boxes-Fusion. Our ensemble technic is shown in the following figure:

enter image description here

As you can see at the end of the Not Validated Stage, we use WBF+p_sum. This last blending method is a variant of WBF proposed by @socom20 and was intended to simulate the consensus of radiologists used in this particular test dataset. By using WBF+p_sum improved our Not Validated stage from 0.319 to 0.331 in the PublicLB.

Modified WBF: It has more flexibility than WBF, it has 4 bbox blending technics, we found that only 2 of them were useful in this competition:

"p_det_weight_pmean", has a behaviour similar to WBF, it finds the set of bboxes with iou > thr, and then weights them using their detection probabilities (detection confidences). The new p_det (detection confidence) is an average over the detection probabilities of detections with iou > thr.
"p_det_weight_psum": does the same with the bboxes. In this case, the new p_det is a sum over the detection probability of bboxes with iou > thr.
Because of the sum of the p_dets, "p_det_weight_psum" needs a final normalization step. The idea behind psum is to try to replicate "the consensus of radiologists" using the partial models. When adding the probabilities of detection, we weigh more those boxes in which more models/radiologists agree.
In our final submission, the not validated stage had a high weighting, which lead to an overfitted problem in our final ensemble we got 0.354/0.314 Public/Private. If we had reduced the weighting for this not validated stage, the LB score would have become 0.330/0.321 Public/Private as we previously have mentioned in the post.

Some Interesting Findings
After spending more time understanding the competition metric, we realized that there is No Penalty For Adding More Bbox as @cdeotte explains clearly here: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229637. Having more confident boxes is good but, at the same time, having many low confidence bboxes improves the Recall and can improve the mAP as well. We tested this approach by using two models, a Model A with a good mAP, and a Model B with a good Recall for all the classes. The following figure shows on the left the Precision vs Recall curves of Model A. It can be seen that many class tails end with a maximum recall = 0.7. We managed to add to Model A all the bboxes predicted for Model B which doesn't overlap with any bbox of predicted for Model A, the result is in the plot of the right. It can be seen that the main shape of the PvsR curves is still the same, but the tails now reach over recall=0.8. We didn't use this approach in our final model because the final ensemble already had a very good recall for all the classes.
enter image description here

Last competition days, we spent analyzing the predictions made for the ensemble both by submitting and also visually inspecting predicted bboxes. We found that our improved submissions were indeed improving for almost all classes including the easy and the rare ones.
enter image description here

Things didn't work
Multi-Label Classifier: We trained a multilabel classifier using as output the 14 classes. We used sigmoid as activation because we needed the probability of each disease and many images have more than one disease. The idea was to remove from the ensemble all the bboxes not predicted by this classifier model, This approach led to a reduction in performance. We also tried to switch the pooling layer and to use PCAM pooling (it is an attention pooling layer used in the Top1 Solution of CheXpert), but it didn't work either.
Bbox Filter: We trained an EfficientNetB6 which classifies whether the detected bboxes belong to a selected class or not just by seeing the crop selected by the ensemble. The model couldn't understand the difference between each disease, perhaps it needed more information on the whole image.
Usage of External Dataset: We tried to use the NIH dataset. We first trained a backbone as a Multi-Label Classifier using NIH classes, and then we transfer its weights to the detector's backbone. Finally, we trained only the detection heads using this competition's dataset. We couldn't detect any appreciable improvement in the particular detector score.
ClassWise Model: we trained different models, one for each detection class. We used clsX + cls14 as each model's dataset. In this case, each model had its own set of anchors prepared for detecting the class of interest. Some classes improved their CV performance but others didn't. Joining all the models we got a very good model, unfortunately, the validation split used in this model was different from the others and it was unreliable to add it to the Validated Stage so we decided to include it in the Partially Validated Stage, but still, didn't give any improvements. We had a final ensemble using this model, which wasn't finally selected, but it had the same score as our winning submission.
Models we choose.
Best at local validation (around 0.47+ mAP on CV), Public: 0.300, private: 0.287
Best at LB. Public: 0.354, private: 0.314
Our best submission in private was 0.321 which was 0.330 in Public
Hardware we used
4x Quadro GV100
4x Geforce GTX 1080
4x Titan X
Ryzen 9 3950x
Nvidia RTX 3080
Quadro RTX 6000
EDIT 1: The solution is reproducible by one GPU itself.


Part 1: Initial models
At first I started to try different OD models for this dataset. I tried: EffDet (B5), CenterNet (HourGlass), Yolo_v5, Retinanet (RsNet50, ResNet101, ResNet152), Faster RCNN

Main initial tricks:

To train models I used WBF on train.csv with IoU: 0.4. I usually used this file with most of models for training and for validation.
Using validation I find out that it’s critical to use lower values for IoU in NMS which filters boxes on output of models. So I used NMS IoU: 0.25 for Yolo_v5 and NMS IoU: 0.3 for RetinaNet. Looks like, it was better for high confident boxes to consume more boxes around it.
I didn’t want to change anchors for RetinaNet so I created version of train.csv with boxes which is close to square, because default Retina doesn’t like elongated boxes.
Best models were the following:

Yolo_v5: ~0.251 Public LB (Std + Mirror images 5KFold, 640px)
RetinaNet (ResNet101): ~0.246 Public LB (Std + Mirror images 5KFold, 1024px)
RetinaNet (ResNet152): ~0.222 Public LB (Std + Mirror images 5KFold, 800px)
CenterNet (HourGlass): ~0.196 Public LB(Std + Mirror images 5KFold, 512px)
EffDet – same code as Wheat competition, but works pretty poor on this dataset. With Faster-RCNN I think I have some problem with code, because it has the very bad score.

Using WBF ensemble of these models + 2 public models (Detectron2 and Yolo_v5):

0.221: https://www.kaggle.com/corochann/vinbigdata-detectron2-prediction
0.204: https://www.kaggle.com/awsaf49/vinbigdata-2-class-filter?scriptVersionId=51562634
I was able to achieve 0.300 on Public LB (0.289 Private LB). Then I merge with my teammates. They had more mmdetection models as well as Yolo_v5, but together we were able to got only 0.301.

Part 2: Investigation
While we were stuck we decided to get individual score for each class and compare with our OOF validation. For 3 days we got the following table:

Class 0: 0.014 (mAP LB: 0.210 Valid: 0.900)
Class 1: 0.009 (mAP LB: 0.135 Valid: 0.335)
Class 2: 0.017 (mAP LB: 0.255 Valid: 0.278)
Class 3: 0.046 (mAP LB: 0.690 Valid: 0.915)
Class 4: 0.019 (mAP LB: 0.285 Valid: 0.438)
Class 5: 0.013 (mAP LB: 0.195 Valid: 0.376)
Class 6: 0.010 (mAP LB: 0.150 Valid: 0.465)
Class 7: 0.003 (mAP LB: 0.045 Valid: 0.400)
Class 8: 0.011 (mAP LB: 0.165 Valid: 0.372)
Class 9: 0.004 (mAP LB: 0.060 Valid: 0.162)
Class 10: 0.027 (mAP LB: 0.405 Valid: 0.548)
Class 11: 0.012 (mAP LB: 0.180 Valid: 0.343)
Class 12: 0.037 (mAP LB: 0.555 Valid: 0.388)
Class 13: 0.009 (mAP LB: 0.135 Valid: 0.432)
Class 14: 0.063 (mAP LB: 0.945 Valid: 0.772)
Sum: 0.294
As it can be seen LB mAP the very different from local mAP. We started from investigation of Class 0 difference (it was most suspicious). We have near perfect prediction on validation while LB score is so small. The first thing we tried is to calculate validation mAP based on Radiologist ID.

Class ID: 0 Validation file: ensemble_iou_0.4._mAP_0.454_extracted_classes_0.csv
Radiologist: R8 mAP: 0.991058
Radiologist: R9 mAP: 0.954593
Radiologist: R10 mAP: 0.971534
Radiologist: R11 mAP: 0.379283
Radiologist: R12 mAP: 0.019819
Radiologist: R13 mAP: 0.511284
Radiologist: R14 mAP: 0.527909
Radiologist: R15 mAP: 0.279870
Radiologist: R16 mAP: 0.506167
Radiologist: R17 mAP: 0.447092
Interesting, we have perfect match for R8, R9 and R10, but very poor on other radiologists. Bad thing is that there are too small entries by other radiologists in train:

R11 - Class:  0 Entries:    30 AP: 0.379283
R12 - Class:  0 Entries:    12 AP: 0.019819
R13 - Class:  0 Entries:    32 AP: 0.511284
R14 - Class:  0 Entries:    74 AP: 0.527909
R15 - Class:  0 Entries:    25 AP: 0.279870
R16 - Class:  0 Entries:    23 AP: 0.506167
R17 - Class:  0 Entries:    10 AP: 0.447092
One more thing that R8, R9 and R10 never markup with any other radiologists. And what if test set marked up by some other radiologists (may be even by R13, R14 etc) and making boxes too similar to these 3 we overfit to train? And what if with better models we went further from test mark up?

I checked by eyes images of class 0 in train marked up by R8, R9 and R10 and by rare radiologists to check the difference. I was lucky to see that other radiologists often use much larger boxes comparing to R8, R9 and R10. So I made a small trick by hand:
x1 -= 100 y2 += 100 for all boxes with class 0
And checked local validation, which improved a lot for other radiologists, while R8, R9 and R10 became worse:

Radiologist: R10 Class:  0 Entries:  2349 AP: 0.559458
Radiologist: R11 Class:  0 Entries:    30 AP: 0.559355
Radiologist: R12 Class:  0 Entries:    12 AP: 0.351739
Radiologist: R13 Class:  0 Entries:    32 AP: 0.483730
Radiologist: R14 Class:  0 Entries:    74 AP: 0.623019
Radiologist: R15 Class:  0 Entries:    25 AP: 0.341545
Radiologist: R16 Class:  0 Entries:    23 AP: 0.506167
Radiologist: R17 Class:  0 Entries:    10 AP: 0.452887
I made the same trick on our 0.301 submission and got 0.309 on LB. This showed us the right direction. We finetuned our models using only rare radiologists. And it improved LB score for all of models:

RetinaNet (ResNet101): 0.246 -> 0.264
RetinaNet (ResNet152): 0.222 -> 0.237
Yolo_v5: 0.251 -> 0.271
Ensemble for all models including original moved us to Public LB > 0.330

Part 3 (Single classes):
At this point it became hard to add new models to ensemble without proper validation. So we decided to switch to single class improvements. It was much easier to find proper stop epochs for this case. Also training procedure became more complicated. We added NIH dataset boxes for classes where it was possible. We prepared NIH dataset in format of VinBigData Chest X-ray competition. We added it as separate radiologist R99. We also prepeared dataset for Pneumothorax, based on SIIM competition, but it didn't help.

We used only one RetinaNet-R101 model for finetuning. We started from best previous weights obtained for multiclass case. So finetuning was pretty fast. We changed sampling in following way:

We first select random radiologist, then select random class (target_class ot empty image), then select random image which were marked by selected radiologist and selected class.
Validation was also slightly changed. We excluded all non-empty images which were marked by R8, R9 and R10 radiologists.
In parallel we fine-tuned some models on very high resolution (2048px). Some classes from these models catch small boxes better than previous models. It improved our local validation.
This approaches together allowed us to improve even more: >0.36 on public

Part 4. Final steps
Actually we decided to create 2 submissions: best on validation and best on LB. We made first submission pretty early. So at the late stage of competition we mostly work on LB-overfitting. That was a little bit risky, but after private LB was revealed we found out it was right thing to do. )

Models we choose.
Best at local validation (around 0.51 mAP locally), Public: 0.304, private: 0.304
Best at LB. Public: 0.365, private: 0.307
Our best model at private has 0.319 score, but we had no chance to choose it because it had too low score on public - only 0.292. And it wasn't best at local validation.
Code
You can find code for out solution on github:
https://github.com/ZFTurbo/2nd-place-solution-for-VinBigData-Chest-X-ray-Abnormalities-Detection


We would like to thank the competition host(s) and Kaggle for organizing the competition and congratulate all the winners, and anyone who benefited in some way from the competition. Special thanks to my teammates @erniechiew, @css919, @zehuigong and @stephkua

Solution Components:
Detection models
Specialized detector for aortic enlargement
Multi-label classifier-based post-processing
1.Detection models
Pre-processing of the bounding box annotations
I believe that most participants used wbf to merge the original boxes, as this notebook(https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset). I apply the same idea that averaging the coordinate of boxes if their IOUs are higher than a threshold, e.g., 0.2. Then using this pre-processed boxes as my annotations in this competition.
Our YOLO-V4 Detector
• Optimizer: SGD, BS = 32, LR=0.001, CosineAnnealing.
• Image size 1280 for P6, 1536 for P7.
• Augmentations:
(1) RandomFlipping (p=0.5)
(2) RandomScaling (0.2-1.8)
(3) Mosaic
(4) Mixup(p=0.2)
(5) Random translation (0.5)

1.1 Training on the boxes of a single radiologist (LB 0.274)
We notice that the processing method of annotation between training and test set are different (see this notebook https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/211035). So we analyze how many images each radiologist annotates, and found that rad 8, 9, and 10 annotate the majority of bounding boxes. So we train three detectors for rad 8, 9, and 10, separately. The score of the separate detector are showed as follow: (CV/LB/Private)
Rad 8: 0.498/0.252/0.233
Rad9: 0.376/0.213/0.234
Rad10: 0.384/0.228/0.226

And then I think, why not try to merge them all? Then after I merge them, I got 0.274 on LB, and a private score of 0.248.

1.2 To the LB 0.280.
Then, I try to train the detectors on all radiologists. For this part, YOLO-V4 P6 and P7 version are employed. (LB/Private)
P6: 0.278/0.261
P7: 0.264/0.263
After I got these models, I merge them all with the step one submission, and we achieve 0.280 on the public leaderboard, and 0.272 on the private.

Things not work:
(1) Albu augmentation, e.g., CLAHE, RandomBrigheness
(2) Class aware sampling
(3) Training with larger image 1536 for P6
(4) Focal loss + BCE
(5) Pretraining on RSNA dataset and finetune on training set of this competition
(6) Pseudo labeling on the test set
(7) Train specialized detector on class ILD only

Things that we haven’t tried for details:
(1) Pseudo labeling the RSNA or other dataset, denoted as PseData, and training on PseData + TrainSet, this improve the CV from 0.498 to 0.55+, but not improve the LB, however, it did make a improvement on the private score, 0.250 vs. 0.255.
(2) Crop the boxes from abnormal images and paste them on normal images, with different sampling ratios for different classes.

2. Specialized Detector for Aortic Enlargement
As highlighted in the discussion forums during the competition, there is a big discrepancy between CV and LB scores. To understand this discrepancy better, we decided to compare LB and local CV scores for each individual class. The AP scores of two classes -- aortic enlargement (class 0) and cardiomegaly (class 3) stood out to us. Below are the individual class AP scores from one of our earlier detectors:

- Aortic enlargement: 0.712 CV / 0.24 LB
- Cardiomegaly: 0.772 CV / 0.81 LB

Intuitively, we felt that LB performance on these classes should follow the same trends since they are i) “easy” (based on their high local CV scores), ii) relatively popular assuming the test set follows a similar disease distribution as the training set. The observation that aortic enlargement performs drastically worse on LB warranted a further investigation.

We found that CV scores for aortic enlargement on images annotated by non-R8/R9/R10 radiologists were unusually low (similar to our LB score for this class). Upon further analysis, we found that:

Aortic enlargement bboxes annotated by non-R8/R9/R10 rad_ids tend to be larger than those annotated by R8, R9, R10.
Our model has a non-negligible tendency to mixup Aortic enlargement and calcification for images annotated by non-R8/R9/R10 rad_ids.
And so we decided to fine-tune one of our detectors on non-R8/R9/R10 rad_ids only. We also trained this detector only on aortic enlargement and calcification classes in an attempt to make the model focus on better distinguishing these two classes.

We then replaced our final predictions for aortic enlargement from our main detector with predictions from this specialized detector. This yielded an increase in overall mAP of approximately 0.020 on public LB, and 0.010 on private LB from our original score.

After reading the great solutions posted by other teams, it is comforting to know that several teams also leveraged this data peculiarity in similar ways!

3. Multi-label Classifier-based Post-processing
A detailed breakdown of this step has been posted in a separate discussion(https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/discussion/229636).
code is https://github.com/Scu-sen/VinBigData-Chest-X-ray-Abnormalities-Detection

Final Results
Ensemble of 11 detection models with 2-class classifier Public: 0.290 Private:0.284
+Specialized Aortic detector Public:0.310 Private: 0.294
+ Multi-label Classifier-based Post-processing Public: 0.348 Private: 0.305


Code object detection competition:

https://github.com/Sense-X/TSD

