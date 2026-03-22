================================================================================
ERROR ANALYSIS — NM i AI 2026 — 3-Model WBF Ensemble
================================================================================

Overall Results:
Competition Score: 0.9400
  Detection mAP@0.5:       0.9698 (× 0.7 = 0.6788)
  Classification mAP@0.5:  0.8705 (× 0.3 = 0.2612)
  Predictions: 51313, Ground truth: 22731

Total predictions: 51313
Total GT annotations: 22731

================================================================================
A) PER-CLASS CLASSIFICATION AP (WORST 30 CLASSES)
================================================================================

Rank  Cat ID   AP@0.5     GT Count   Name
------------------------------------------------------------------------------------------
1     26       0.0099     1          OB PROCOMFORT NORMAL 16ST
2     57       0.0099     1          MORENEPOTETER GULE 650G BJERTNÆS&HOEL
3     60       0.0099     2          KNEKKEBRØD URTER&HAVSALT GL.FRI 190G
4     69       0.0099     1          DAVE&JON'S DADLER SOUR COLA 125G
5     76       0.0099     1          BRUSCHETTA LIGURISK 130G OLIVINO
6     91       0.0099     1          SANDWICH PESTO 37G WASA
7     95       0.0099     1          EGG M/L ØKOLOGISKE 10STK VILJE
8     104      0.0099     3          SMØR USALTET 250G TINE
9     149      0.0099     1          KNEKKEBRØD SESAM&HAVSALT GL.FRI 240G
10    154      0.0099     1          KRYDDERMIKS SHISH KEBAB 10G POS HINDU
11    199      0.0099     1          GREEN CEYLON TE ØKOLOGISK 24POS CONFECTA
12    211      0.0099     1          BJØRN HAVREMEL 1KG AXA
13    227      0.0099     1          STORFE ENTRECOTE 180G FIRST PRICE
14    230      0.0099     1          JARLSBERG 27% SKIVET 120G TINE
15    234      0.0099     1          LANO SÅPE 2X125G
16    241      0.0099     1          SANDWICH CHEESE GRESSLØK 37G WASA
17    242      0.0099     1          BOG 390G GILDE
18    263      0.0099     1          GRANOLA RASPBERRY  500G START!
19    276      0.0099     2          HVITLØK 100G PK
20    279      0.0099     1          KNEKKEBRØD NATURELL GL.FRI 240G WASA
21    291      0.0099     1          PREMIUM DARK ORANGE 100G FREIA
22    300      0.0099     1          
23    335      0.0099     1          STORFE SHORT RIBS GREATER OMAHA LV
24    317      0.2079     1          POWERKNEKKEBRØD GL.FRI 225G SCHÄR
25    133      0.2079     5          ZOEGAS KAFFE SKÅNEROST 450G
26    116      0.2574     1          FROKOSTBLANDING LION 350G NESTLE
27    101      0.3350     13         SANDWICH CHEESE HERBS 30G WASA
28    156      0.3366     3          GRANOLA CRAZELNUT 500G START!
29    183      0.4059     5          KNEKKEBRØD GODT FOR DEG OST 190G SIGDAL
30    290      0.4644     29         SANDWICH CHEESE TOMAT&BASILIKUM 40G WASA


BEST 20 CLASSES:
Rank  Cat ID   AP@0.5     GT Count   Name
------------------------------------------------------------------------------------------
1     2        1.0000     6          FROKOSTRINGER FRUKTSMAK 350G ELDORADO
2     4        1.0000     10         Økologiske Egg 6stk
3     10       1.0000     12         Jacobs 10 Gårdsegg
4     16       1.0000     6          CAPPUCCINO 8KAPSLER DOLCE GUSTO
5     18       1.0000     13         MELANGE MARGARIN U/MELK/SALT/SOYA 500G
6     31       1.0000     17         TASSIMO EVERGOOD DARK ROAST 16KAPSLER
7     32       1.0000     67         SOLBÆRTODDY 10PK 320G FREIA
8     41       1.0000     11         VITA HJERTEGO MYK PLANTEMARGARIN 370G
9     43       1.0000     25         MEIERISMØR 500G TINE
10    44       1.0000     2          KNEKKS KJEKS HAVRE 190G RØROS
11    51       1.0000     5          Tørresvik Gårdsegg 6stk
12    59       1.0000     8          MÜSLI BLÅBÆR 630G AXA
13    65       1.0000     27         PULVERKAFFE INSTANT 200G FIRST PRICE
14    71       1.0000     28         AMERICANO 16 KAPSLER DOLCE GUSTO
15    77       1.0000     4          BAKEKAKAO 250G REGIA
16    79       1.0000     1          PANNEKAKER GROVE STEKTE 480G ÅMLI
17    81       1.0000     1          GRØNN TE CHAI 25POS TWININGS
18    90       1.0000     2          GRANOLA PEKAN GL.FRI 325G SYNNØVE FINDEN
19    93       1.0000     23         GRANOLA KAKAO KOKOS&MANDEL BARE BRA
20    94       1.0000     1          HAVREGRYN STORE GLUTENFRI 1KG AXA

AP Distribution:
  Mean: 0.8705
  Median: 0.9667
  Std: 0.2577
  Min: 0.0099, Max: 1.0000
  Classes with AP=0: 23
  Classes with AP<0.5: 30
  Classes with AP<0.8: 54
  Classes with AP>=0.9: 276

================================================================================
B) CLASSIFICATION CONFUSION ANALYSIS
================================================================================

Detection matching (IoU >= 0.5):
  Correctly classified: 21489 (94.5% of GT)
  Wrong class (detected but misclassified): 882 (3.9% of GT)
  Missed entirely (false negatives): 360 (1.6% of GT)

Top 30 confusion pairs (GT -> Predicted):
Count    GT Cat   Pred Cat   GT Name                             Pred Name
--------------------------------------------------------------------------------------------------------------
28       345      92         KNEKKEBRØD URTER&HAVSALT 220G SIGDA KNEKKEBRØD GODT FOR DEG 235G SIGDAL
21       325      146        NESCAFE GULL 100G                   NESCAFE GULL 200G NESTLE
16       27       189        YELLOW LABEL TEA 50POS LIPTON       YELLOW LABEL TEA 25POS LIPTON
13       240      47         SUPERGRANOLA GLUTENFRI 350G BARE BR GRANOLA EPLE&KANEL 430G BARE BRA
12       355      12         unknown_product                     Leksands Rutbit
10       292      137        NESCAFE AZERA AMERICANO 100G        NESCAFE AZERA ESPRESSO 100G
10       21       39         KNEKKEBRØD RUNDA SESAM&HAVSALT 290G KNEKKEBRØD RUNDA KANEL 330G WASA
9        84       86         HAVRE KNEKKEBRØD ØKONOMI 600G WASA  HAVRE KNEKKEBRØD 300G WASA
9        338      246        HUSMAN KNEKKEBRØD 520G WASA         HUSMAN KNEKKEBRØD 260G WASA
8        97       181        CHEERIOS MULTI 375G NESTLE          CHEERIOS HAVRE 375G NESTLE
8        246      338        HUSMAN KNEKKEBRØD 260G WASA         HUSMAN KNEKKEBRØD 520G WASA
7        56       259        SJOKOLADEDRIKK 10PK 320G REGIA      REGIA KAKAO ORIGINAL 260G
7        48       128        SMØREMYK 600G ELDORADO              SMØREMYK LETT 400G ELDORADO
7        186      268        SUPERGRØT SKOGSBÆR 57G BARE BRA     SUPERGRØT KANEL&PUFFET QUINOA 54G
7        61       315        MUSLI BLÅBÆR 630G AXA               4-KORN 675G AXA
7        86       84         HAVRE KNEKKEBRØD 300G WASA          HAVRE KNEKKEBRØD ØKONOMI 600G WASA
7        188      307        MAISKAKER POPCORN 125G FRIGGS       MAISKAKER OST 125G FRIGGS
6        304      341        EVERGOOD CLASSIC KOKMALT 250G       EVERGOOD CLASSIC HELE BØNNER 500G
6        100      304        EVERGOOD CLASSIC FILTERMALT 250G    EVERGOOD CLASSIC KOKMALT 250G
6        325      106        NESCAFE GULL 100G                   NESCAFE GOLD KOFFEINFRI 100G
6        160      171        ALI ORIGINAL KOKMALT 250G           ALI ORIGINAL FILTERMALT 250G
6        117      78         GRANOLA HASSELNØTT&KOKOS 450G BARE  MUSLI HAVRE&MANDEL 400G BARE BRA
6        181      97         CHEERIOS HAVRE 375G NESTLE          CHEERIOS MULTI 375G NESTLE
6        132      271        FRUKOST FULLKORN 320G WASA          FRUKOST KNEKKEBRØD 240G WASA
6        349      296        KNEKKEBRØD SPORT+ 210G WASA         FIBER BALANCE 230G WASA
6        158      250        SURDEIG KNEKKEBRØD WASA             LEKSANDS KNEKKE NORMALT STEKT 200G
5        171      160        ALI ORIGINAL FILTERMALT 250G        ALI ORIGINAL KOKMALT 250G
5        125      169        SJOKOLADEDRIKK 512G RETT I KOPPEN   SJOKOLADEDRIKK REFILL 512G RETT I K
5        305      47         GRANOLA KAKAO&BRINGEBÆR 430G BARE B GRANOLA EPLE&KANEL 430G BARE BRA
5        244      52         RUGSPRØ HAVRE 180G WASA             RUGSPRØ 200G WASA

================================================================================
C) SPATIAL ANALYSIS — WHERE DO DETECTIONS FAIL?
================================================================================

Region               Total GT     Missed     Miss Rate
-------------------------------------------------------
right_10pct          1252         27         0.022
left_10pct           1025         18         0.018
center               10945        172        0.016
bottom_quarter       4088         55         0.013
bottom_10pct         838          9          0.011
top_quarter          3615         34         0.009
top_10pct            771          3          0.004

Miss rate by object size:
Size Bin                  Total GT     Missed     Miss Rate
-------------------------------------------------------
tiny (<32x32)             33           22         0.667
small (32-64)             791          120        0.152
medium (64-128)           6233         124        0.020
large (>128)              15674        83         0.005

================================================================================
D) FALSE NEGATIVES — COMPLETELY MISSED GT OBJECTS
================================================================================

Categories with most missed objects:
Missed   GT Total   Miss%    Cat ID   Name
--------------------------------------------------------------------------------
16       374        4.3      109      KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA
14       271        5.2      21       KNEKKEBRØD RUNDA SESAM&HAVSALT 290G WASA
13       247        5.3      92       KNEKKEBRØD GODT FOR DEG 235G SIGDAL
13       322        4.0      246      HUSMAN KNEKKEBRØD 260G WASA
12       307        3.9      271      FRUKOST KNEKKEBRØD 240G WASA
10       260        3.8      233      LEKSANDS KNEKKE GODT STEKT 200G
9        167        5.4      342      HAVREFRAS 375G
7        398        1.8      86       HAVRE KNEKKEBRØD 300G WASA
7        271        2.6      250      LEKSANDS KNEKKE NORMALT STEKT 200G
7        13         53.8     101      SANDWICH CHEESE HERBS 30G WASA
7        300        2.3      132      FRUKOST FULLKORN 320G WASA
6        34         17.6     106      NESCAFE GOLD KOFFEINFRI 100G
6        195        3.1      188      MAISKAKER POPCORN 125G FRIGGS
6        227        2.6      52       RUGSPRØ 200G WASA
6        223        2.7      345      KNEKKEBRØD URTER&HAVSALT 220G SIGDAL
5        368        1.4      100      EVERGOOD CLASSIC FILTERMALT 250G
5        190        2.6      97       CHEERIOS MULTI 375G NESTLE
5        85         5.9      0        FRØKRISP KNEKKEBRØD ØKOLOGISK 170G BERIT
5        283        1.8      307      MAISKAKER OST 125G FRIGGS
5        185        2.7      152      HAVREKNEKKEBRØD TYNT 265G WASA
4        5          80.0     133      ZOEGAS KAFFE SKÅNEROST 450G
4        422        0.9      355      unknown_product
4        126        3.2      186      SUPERGRØT SKOGSBÆR 57G BARE BRA
4        240        1.7      239      MAISKAKER CHIA/HAVSALT 130G FRIGGS
4        115        3.5      235      FLATBRØD ØKOLOGISK 190G RØROS

================================================================================
E) FALSE POSITIVES — PREDICTIONS WITH NO MATCHING GT
================================================================================

Total false positives: 28942

Categories with most false positives:
FP Count   Cat ID   Name
----------------------------------------------------------------------
1781       355      unknown_product
936        109      KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA
692        85       RIS PUFFET 260G FIRST PRICE
519        92       KNEKKEBRØD GODT FOR DEG 235G SIGDAL
373        100      EVERGOOD CLASSIC FILTERMALT 250G
370        233      LEKSANDS KNEKKE GODT STEKT 200G
362        275      EGG FRITTGÅENDE 18STK S/M FIRST PRICE
360        7        PUFFET HAVRE 340G FIRST PRICE
351        52       RUGSPRØ 200G WASA
347        246      HUSMAN KNEKKEBRØD 260G WASA
334        280      RISKAKER 100G FIRST PRICE
324        171      ALI ORIGINAL FILTERMALT 250G
321        86       HAVRE KNEKKEBRØD 300G WASA
317        338      HUSMAN KNEKKEBRØD 520G WASA
314        271      FRUKOST KNEKKEBRØD 240G WASA
312        243      RYVITA KNEKKEBRØD RUG 400G
309        345      KNEKKEBRØD URTER&HAVSALT 220G SIGDAL
268        239      MAISKAKER CHIA/HAVSALT 130G FRIGGS
262        80       FROKOSTEGG FRITTGÅENDE L 12STK PRIOR
262        261      MÜSLI HASSELNØTT 600G AXA

================================================================================
F) PREDICTION SCORE DISTRIBUTION
================================================================================

All predictions score stats:
  Count: 51313
  Mean: 0.3604
  Median: 0.1322
  Std: 0.3740

Threshold    Preds Above     FP Above     FP Rate    TP Above    
-----------------------------------------------------------------
0.01         38907           16553        0.425      22354       
0.05         29334           7039         0.240      22295       
0.10         26671           4448         0.167      22223       
0.15         25221           3074         0.122      22147       
0.20         24307           2251         0.093      22056       
0.25         23595           1690         0.072      21905       
0.30         22987           1280         0.056      21707       
0.40         21831           768          0.035      21063       
0.50         20554           488          0.024      20066       
0.60         18844           304          0.016      18540       
0.70         16305           177          0.011      16128       
0.80         12704           46           0.004      12658       
0.90         2809            2            0.001      2807        

================================================================================
G) WORST IMAGES BY DETECTION PERFORMANCE
================================================================================

20 Worst images (by F1 score at conf=0.1):
Img ID   F1      Prec    Rec     TP    FN    FP    GT    Pred   File
------------------------------------------------------------------------------------------
209      0.675   0.519   0.966   28    1     26    29    54     img_00209.jpg
295      0.727   0.581   0.971   132   4     95    136   227    img_00295.jpg
89       0.753   0.604   1.000   64    0     42    64    106    img_00089.jpg
304      0.767   0.685   0.871   122   18    56    140   178    img_00304.jpg
120      0.780   0.654   0.967   87    3     46    90    133    img_00120.jpg
297      0.788   0.732   0.852   115   20    42    135   157    img_00297.jpg
22       0.796   0.672   0.978   45    1     22    46    67     img_00022.jpeg
127      0.798   0.685   0.956   87    4     40    91    127    img_00127.jpg
319      0.798   0.683   0.960   95    4     44    99    139    img_00319.jpg
229      0.800   0.733   0.881   96    13    35    109   131    img_00229.jpg
294      0.817   0.752   0.896   103   12    34    115   137    img_00294.jpg
122      0.824   0.712   0.977   42    1     17    43    59     img_00122.jpg
328      0.825   0.734   0.942   146   9     53    155   199    img_00328.jpg
306      0.828   0.724   0.967   89    3     34    92    123    img_00306.jpg
227      0.830   0.723   0.974   112   3     43    115   155    img_00227.jpg
55       0.831   0.716   0.990   96    1     38    97    134    img_00055.jpeg
1        0.835   0.717   1.000   99    0     39    99    138    img_00001.jpg
248      0.835   0.728   0.980   99    2     37    101   136    img_00248.jpg
152      0.837   0.762   0.928   64    5     20    69    84     img_00152.jpeg
310      0.839   0.749   0.954   146   7     49    153   195    img_00310.jpg

================================================================================
H) CATEGORIES IN GT BUT NEVER PREDICTED (or very rare)
================================================================================

Cat ID   GT Count   Pred Count   Name
--------------------------------------------------------------------------------
26       1          0            OB PROCOMFORT NORMAL 16ST
57       1          0            MORENEPOTETER GULE 650G BJERTNÆS&HOEL
60       2          0            KNEKKEBRØD URTER&HAVSALT GL.FRI 190G
69       1          0            DAVE&JON'S DADLER SOUR COLA 125G
76       1          0            BRUSCHETTA LIGURISK 130G OLIVINO
91       1          0            SANDWICH PESTO 37G WASA
95       1          0            EGG M/L ØKOLOGISKE 10STK VILJE
104      3          0            SMØR USALTET 250G TINE
116      1          0            FROKOSTBLANDING LION 350G NESTLE
133      5          1            ZOEGAS KAFFE SKÅNEROST 450G
154      1          0            KRYDDERMIKS SHISH KEBAB 10G POS HINDU
156      3          0            GRANOLA CRAZELNUT 500G START!
183      5          1            KNEKKEBRØD GODT FOR DEG OST 190G SIGDAL
199      1          0            GREEN CEYLON TE ØKOLOGISK 24POS CONFECTA
211      1          0            BJØRN HAVREMEL 1KG AXA
227      1          0            STORFE ENTRECOTE 180G FIRST PRICE
230      1          0            JARLSBERG 27% SKIVET 120G TINE
234      1          0            LANO SÅPE 2X125G
241      1          0            SANDWICH CHEESE GRESSLØK 37G WASA
242      1          0            BOG 390G GILDE
263      1          0            GRANOLA RASPBERRY  500G START!
276      2          0            HVITLØK 100G PK
279      1          0            KNEKKEBRØD NATURELL GL.FRI 240G WASA
291      1          0            PREMIUM DARK ORANGE 100G FREIA
300      1          0            
317      1          0            POWERKNEKKEBRØD GL.FRI 225G SCHÄR
335      1          0            STORFE SHORT RIBS GREATER OMAHA LV
================================================================================
I) ACTIONABLE RECOMMENDATIONS
================================================================================

## Key Findings Summary

- Val competition score: 0.9400 (det mAP 0.9698, cls mAP 0.8705)
- Detection is strong (97% mAP) — the main gap is classification (87%)
- 94.5% of GT objects correctly classified, 3.9% misclassified, 1.6% missed
- 23 classes have AP near zero (mostly 1-sample classes)
- 276 out of 356 classes have AP >= 0.9
- Most FPs are low-confidence: at threshold 0.25, FP rate drops to 7.2%
- Small objects (<64px) have 15.2% miss rate; tiny (<32px) 66.7% miss rate

## 1. CONFIDENCE THRESHOLD OPTIMIZATION

Current: using all predictions (conf >= 0.01 from WBF skip_box_thr)

- At conf=0.20: 93% FP reduction vs 0.01, lose only ~300 TPs (22056 vs 22354)
- At conf=0.15: 81% FP reduction, lose only ~207 TPs
- RECOMMENDATION: Test optimal threshold around 0.15-0.25 globally.
  This should improve precision substantially with minimal recall loss.
- Per-class thresholds unlikely to help much — most errors are between
  similar products, not systematic per-class score issues.

## 2. CLASSIFICATION CONFUSION FIXES (highest ROI)

The top confusion pairs are all VISUALLY SIMILAR products:
- Same brand, different size (NESCAFE GULL 100G vs 200G, HUSMAN 260G vs 520G)
- Same brand, different variant (CHEERIOS MULTI vs HAVRE, ALI KOKMALT vs FILTERMALT)
- Same packaging style (KNEKKEBRØD variants from Sigdal, WASA, Leksands)

RECOMMENDATIONS:
a) **Product image gallery classifier**: Use the product_images/ reference photos
   as an embedding gallery. After detection, crop each detection and match against
   the gallery using CLIP or similar embeddings. This could fix ~50% of the 882
   misclassifications since they're between visually distinct products when viewed
   up close.
b) **Increase input resolution**: Many confusions are between products that differ
   only in small text. Training at 1280px (with a model that fits in VRAM, e.g.,
   YOLOv8l) could help distinguish these. However, per CLAUDE.md, YOLOv8x doesn't
   fit at 1280px on T4.
c) **WBF label voting**: Currently WBF averages labels. For confused pairs, using
   majority voting or probability-weighted voting on the probs tensor could help.
   The 3 models may disagree on confusable pairs — using the full 356-class
   probability vectors to vote could recover correct labels.

## 3. FALSE NEGATIVE REDUCTION

Only 360 missed GTs (1.6%) — relatively minor. Main patterns:
- Small objects (32-64px): 120 missed — higher-res inference or SAHI could help
- Tiny objects (<32px): 22 missed — likely annotation noise, hard to improve
- Edge objects (left/right 10%): slightly higher miss rate — TTA with crops could help
- Categories with most misses are common knekkebrød/cracker products that appear
  frequently — these are likely occluded or partially visible instances

RECOMMENDATIONS:
a) **SAHI (Sliced Inference)**: Already implemented in sahi.py. Running with
   slice_size=640, overlap=0.2 on the 3 models could recover small objects.
   Risk: adds many FPs. Test with higher conf threshold for SAHI detections.
b) Keep current approach — 1.6% miss rate is already very good.

## 4. FALSE POSITIVE REDUCTION

28,942 FPs at conf>=0.01 — but most are low confidence:
- At conf>=0.20, only 2,251 FPs remain
- Biggest FP category: "unknown_product" (1,781 FPs) — model hallucinates
  unknown products or assigns unknown_product to background regions
- Other high-FP categories (KNEKKEBRØD, RISKAKER, etc.) are common products
  where the model generates duplicate/extra detections

RECOMMENDATIONS:
a) **Increase skip_box_thr in WBF**: Currently 0.01. Try 0.10-0.15 to filter
   low-confidence boxes before fusion. This is the single easiest win.
b) **Post-WBF confidence filter**: Apply a final conf threshold of 0.15-0.20
   after WBF fusion.
c) **NMS after WBF**: Apply a secondary NMS pass to remove remaining duplicate
   boxes that WBF didn't merge (e.g., nearby boxes with different labels).

## 5. WBF PARAMETER TUNING

Current: iou_thr=0.55, skip_box_thr=0.01

RECOMMENDATIONS:
- **Lower iou_thr to 0.45-0.50**: More aggressive merging could reduce FPs
  from nearby duplicate detections.
- **Raise skip_box_thr to 0.05-0.10**: Filter weak detections before fusion.
- **Weighted models**: If one model generalizes better, give it higher weight
  in WBF (weights parameter).

## 6. SPECIFIC CLASS IMPROVEMENTS

- 23 zero-AP classes are mostly single-instance — can't fix with training
- The 7 classes with AP 0.20-0.50 are more actionable:
  - cat 317 (POWERKNEKKEBRØD GL.FRI): 1 sample, confused with other knekkebrød
  - cat 133 (ZOEGAS KAFFE): 5 samples, 4 missed — possibly confused with similar coffee
  - cat 116 (LION FROKOSTBLANDING): 1 sample
  - cat 101 (SANDWICH CHEESE HERBS): 13 samples, 7 missed — small sandwich products
  - cat 156 (GRANOLA CRAZELNUT): 3 samples, confused with other granola
  - cat 183 (KNEKKEBRØD GODT FOR DEG OST): 5 samples, confused with cat 92 (same brand)
  - cat 290 (SANDWICH CHEESE TOMAT): 29 samples — larger class, worth investigating

## 7. PRIORITY ACTION ITEMS (ranked by expected impact on test set)

1. **POST-PROCESSING CONF THRESHOLD**: Test 0.15, 0.20, 0.25 — free improvement,
   no retraining needed. Can test immediately.
2. **WBF skip_box_thr**: Increase to 0.05 or 0.10 — easy parameter change.
3. **PROBABILITY-WEIGHTED CLASSIFICATION**: Use the full probs tensors from 3
   models to vote on class labels instead of relying on WBF label fusion.
   Average the 356-dim probability vectors across models for each fused box.
4. **PRODUCT GALLERY CLASSIFIER**: Post-detection CLIP/embedding matching
   against product reference images. High effort but could fix 200+ misclassifications.
5. **SAHI for small objects**: Low priority — only 120 small objects missed.

## 8. IMPORTANT CAVEAT

Per CLAUDE.md: val mAP does NOT reliably predict test performance. The val set is
only ~37 images. Best leaderboard score (0.7802) came from the baseline yolov8x_640,
not from higher-val-mAP models. Any changes should be validated via Docker + submission,
not just val scores. Focus on changes with strong theoretical justification (e.g.,
confidence thresholding, better label fusion) rather than chasing val metrics.
