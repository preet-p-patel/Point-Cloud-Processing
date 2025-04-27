# Learning for 3D Vision - Assignment 5: Point Cloud Processing

## Q1. Classification Model

The model architecture is same as one implemented in PointNet paper trained for 250 epochs (best model at epoch 76). 

__Test Accuracy__: 98.32%

Visualisations:

<table>
    <tr>
        <th>Index</th>
        <th>Predicted Class</th>
        <th>Ground Truth Class</th>
        <th>Ground Truth Visualisation</th>
    </tr>
    <tr>
        <th>124</th>
        <th>Chair</th>
        <th>Chair</th>
        <td><img src="output\cls\visual\gt_Q1_124.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>665</th>
        <th>Vase</th>
        <th>Vase</th>
        <td><img src="output\cls\visual\gt_Q1_665.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>724</th>
        <th>Lamp</th>
        <th>Lamp</th>
        <td><img src="output\cls\visual\gt_Q1_724.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>406</th>
        <th>Lamp</th>
        <th>Chair</th>
        <td><img src="output\cls\visual\gt_Q1_406.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>651</th>
        <th>Lamp</th>
        <th>Vase</th>
        <td><img src="output\cls\visual\gt_Q1_651.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>832</th>
        <th>Vase</th>
        <th>Lamp</th>
        <td><img src="output\cls\visual\gt_Q1_832.gif" alt="Grid Visualization"></td> 
    </tr>
</table>


**Interpretation** : Our model seems to be accurately predicting the class for most of the cases, and it gets confused only for cases where the object has many features resembling another class and not a common shape of its own class. For eg index 651, the vase has structure very similar to the lamp from index 724 and in general similar to a desk lamp. 


## Q2. Segmentation Model

Similar to the classification model, the segmentation model is also an implementation of the PointNet segmentation model trained for 250 epochs (best model @ 73 epoch).

__Test Accuracy__ : 89.64%

Visualisations:

<table>
    <tr>
        <th>Index</th>
        <th>Prediction Accuracy</th>
        <th>Prediction Visualisation</th>
        <th>Ground Truth Visualisation</th>
    </tr>
    <tr>
        <th>0</th>
        <th>94.78%</th>
        <td><img src="output\seg\Q3_1\0\pred_rotate_0_0.gif" alt="Grid Visualization"></td>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_0.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>20</th>
        <th>97.81%</th>
        <td><img src="output\seg\Q3_1\0\pred_rotate_0_20.gif" alt="Grid Visualization"></td>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_20.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>170</th>
        <th>94.11%</th>
        <td><img src="output\seg\Q3_1\0\pred_rotate_0_170.gif" alt="Grid Visualization"></td>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_170.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>90</th>
        <th>61.67%</th>
        <td><img src="output\seg\Q3_1\0\pred_rotate_0_90.gif" alt="Grid Visualization"></td>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_90.gif" alt="Grid Visualization"></td> 
    </tr>
    <tr>
        <th>26</th>
        <th>49.93%</th>
        <td><img src="output\seg\Q3_1\0\pred_rotate_0_26.gif" alt="Grid Visualization"></td>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_26.gif" alt="Grid Visualization"></td> 
    </tr>
</table>

**Interpretation** : Here aswell the model is able to work with objects with highly distinguishable features and segment pretty well. But when the objects dont have clear distinction, the model fails to accurately segment points near intersection as seen for index 90 and 26.


## Q3. Robustness Analysis

#### Experiment 1 - Rotating point cloud about x-axis

__Procedure:__ The input point clouds are rotated by 10, 30 and 60 degrees to check classification and segmentation accuracies. 

CLASSIFICATION TASK:

<table>
    <tr>
        <th>Degree Rotated</th>
        <th>0</th>
        <th>10</th>
        <th>30</th>
        <th>60</th> 
    </tr>
    <tr>
        <th>Test Accuracy</th>
        <th>98.32%</th>
        <th>96.64%</th>
        <th>76.80%</th>
        <th>28.5%</th> 
    </tr>
</table>

Visualisation:

<table>
    <tr>
        <th>Index</th>
        <th>True Class</th>
        <th>Prediction after 0 degree rotation</th>
        <th>Prediction after 10 degree rotation</th>
        <th>Prediction after 30 degree rotation</th>
        <th>Prediction after 60 degree rotation</th>
        <th>Visualisation</th> 
    </tr>
    <tr>
        <th>0</th>
        <th>Chair</th>
        <th>Chair</th>
        <th>Chair</th>
        <th>Lamp</th>
        <th>Lamp</th>
        <td><img src="output\cls\Q3_1\30\gt_trial_rotate_0.gif" alt="Grid Visualization"></td>
    </tr>
    <tr>
        <th>724</th>
        <th>Lamp</th>
        <th>Lamp</th>
        <th>Lamp</th>
        <th>Lamp</th>
        <th>Vase</th>
        <td><img src="output\cls\Q3_1\30\gt_trial_rotate_724.gif" alt="Grid Visualization"></td>
    </tr>
    <tr>
        <th>883</th>
        <th>Lamp</th>
        <th>Vase</th>
        <th>Vase</th>
        <th>Vase</th>
        <th>Lamp</th>
        <td><img src="output\cls\Q3_1\30\gt_trial_rotate_883.gif" alt="Grid Visualization"></td>
    </tr>
</table>

__Inference:__ Rotation of point cloud seems to have a adverse effect on test accuracy. We observe that it drastically drops with larger angles. 
Surprisingle for index 883, which completely looks like a vase, is a lamp and only model with 60 degrees rotation correclty classified it. It seems like a random guess, as at ~29% accuracy there is not much calculated classification done by the model.


SEGMENTATION TASK:

<table>
    <tr>
        <th>Degree Rotated</th>
        <th>0</th>
        <th>10</th>
        <th>30</th>
        <th>60</th> 
    </tr>
    <tr>
        <th>Test Accuracy</th>
        <th>89.64%</th>
        <th>85.91%</th>
        <th>69.02%</th>
        <th>43.67%</th> 
    </tr>
</table>

Visualisation:

<table>
    <tr>
        <th>Index</th>
        <th>Ground Truth</th>
        <th>Segmentation after 0 degree rotation</th>
        <th>Segmentation after 10 degree rotation</th>
        <th>Segmentation after 30 degree rotation</th>
        <th>Segmentation after 60 degree rotation</th> 
    </tr>
    <tr>
        <th>0</th>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_0.gif" alt="Grid Visualization" ></td>
        <td>94.78%<img src="output\seg\Q3_1\0\pred_rotate_0_0.gif" alt="Grid Visualization" ></td>
        <td>92.36%<img src="output\seg\Q3_1\10\pred_rotate_10_0.gif" alt="Grid Visualization" ></td>
        <td>74.41%<img src="output\seg\Q3_1\30\pred_rotate_30_0.gif" alt="Grid Visualization" ></td>
        <td>47.78%<img src="output\seg\Q3_1\60\pred_rotate_60_0.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>41</th>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_41.gif" alt="Grid Visualization" ></td>
        <td>67.11%<img src="output\seg\Q3_1\0\pred_rotate_0_41.gif" alt="Grid Visualization" ></td>
        <td>64.21%<img src="output\seg\Q3_1\10\pred_rotate_10_41.gif" alt="Grid Visualization" ></td>
        <td>55.31%<img src="output\seg\Q3_1\30\pred_rotate_30_41.gif" alt="Grid Visualization" ></td>
        <td>51.26%<img src="output\seg\Q3_1\60\pred_rotate_60_41.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>170</th>
        <td><img src="output\seg\Q3_1\0\gt_rotate_0_170.gif" alt="Grid Visualization" ></td>
        <td>94.11%<img src="output\seg\Q3_1\0\pred_rotate_0_170.gif" alt="Grid Visualization" ></td>
        <td>90.08%<img src="output\seg\Q3_1\10\pred_rotate_10_170.gif" alt="Grid Visualization" ></td>
        <td>76.20%<img src="output\seg\Q3_1\30\pred_rotate_30_170.gif" alt="Grid Visualization" ></td>
        <td>47.04%<img src="output\seg\Q3_1\60\pred_rotate_60_170.gif" alt="Grid Visualization" ></td>
    </tr>
</table>


__Inference:__ Again the accuracy drastically drops with increasing angles. We also notice that the model seems to be segmenting points based on height. In 60 degree segmentations we can still see that the model is trying to classify points at a particular height into seat, points on the side as arm-rest etc. Thus, the model seems to have learnt features based on global features rather than local relationships among points. This could also be due to the fact that training was done only on objects in a particular orientation, thus model had never seen data with different orientations of objects.

#### Experiment 2 - Varying number of points per sample

__Procedure:__ The input point clouds which were by default sampled with 10000 points are tested with lesser points per sample. Thus for experiment 5000, 1000 and 100 points per sample are used. 

CLASSIFICATION TASK:

<table>
    <tr>
        <th>Number of Points per sample</th>
        <th>10000</th>
        <th>5000</th>
        <th>1000</th>
        <th>100</th> 
    </tr>
    <tr>
        <th>Test Accuracy</th>
        <th>98.32%</th>
        <th>98.32%</th>
        <th>98.01%</th>
        <th>92.13%</th> 
    </tr>
</table>

Visualisation:

<table>
    <tr>
        <th>Index</th>
        <th>10000 points</th>
        <th>5000 points</th>
        <th>1000 points</th>
        <th>100 points</th>
    </tr>
    <tr>
        <th>0</th>
        <td><img src="output\cls\gt_Q1_0.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\5000\gt_points_5000_0.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\1000\gt_points_1000_0.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\100\gt_points_100_0.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>124</th>
        <td><img src="output\cls\visual\gt_Q1_124.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\5000\gt_points_5000_124.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\1000\gt_points_1000_124.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\100\gt_points_100_124.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>724</th>
        <td><img src="output\cls\visual\gt_Q1_724.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\5000\gt_points_5000_724.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\1000\gt_points_1000_724.gif" alt="Grid Visualization" ></td>
        <td><img src="output\cls\Q3_2\100\gt_points_100_724.gif" alt="Grid Visualization" ></td>
    </tr>
</table>

__Inference:__ There is not much affect to the accuracy with decreasing points. Initially I planned to test with 5000, 2500, and 1000 points, but observing that accuracy does not decrease much, I wanted to check when does the accuracy start dropping. Still we see that even with only 100 points, the model has pretty high accuracy of ~92%. 

SEGMENTATION TASK:

<table>
    <tr>
        <th>Number of Points per sample</th>
        <th>10000</th>
        <th>5000</th>
        <th>1000</th>
        <th>100</th> 
    </tr>
    <tr>
        <th>Test Accuracy</th>
        <th>89.64%</th>
        <th>89.68%</th>
        <th>89.35%</th>
        <th>79.80%</th> 
    </tr>
</table>

Visualisation:

<table>
    <tr>
        <th>Index</th>
        <th>10000 points</th>
        <th>5000 points</th>
        <th>1000 points</th>
        <th>100 points</th>
    </tr>
    <tr>
        <th>0</th>
        <td><img src="output\seg\pred_exp.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\5000\pred_points_5000_0.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\1000\pred_points_1000_0.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\100\pred_points_100_0.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>117</th>
        <td><img src="output\seg\pred_sample_117.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\5000\pred_points_5000_117.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\1000\pred_points_1000_117.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\100\pred_points_100_117.gif" alt="Grid Visualization" ></td>
    </tr>
    <tr>
        <th>41</th>
        <td><img src="output\seg\pred_failure_cases_41.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\5000\pred_points_5000_41.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\1000\pred_points_1000_41.gif" alt="Grid Visualization" ></td>
        <td><img src="output\seg\Q3_2\100\pred_points_100_41.gif" alt="Grid Visualization" ></td>
    </tr>
</table>

__Inference:__ Once again, the number of points does not have much affect of the segmentation accuracy. It could be possible that, as discussed earlier the model relies on global position of each point, and hence reducing points does not affect its performance. In contrast, if the model depended on relationships between points, with fewer points, it would be difficult to segment and in that case the accuracy might have been affected by number of points. But it would also ensure actual robust performance.  


## Q4 Locality 

#### Classification Task:

__Procedure:__ Implemented a PointNet++ based locality feature using model that finds K nearest neighbours to find local features and uses them along with global position. Due to memory and time issues I was only able to train it for 30 epochs (best model @ 29 epoch), to acheive test accuracy of 96.7% (used 5000 points per sample instead of 10000 due to memory issues).

<table>
    <tr>
        <th>Model</th>
        <th>Normal Test Accuracy</th>
        <th>30 degree Rotate Test Accuracy</th> 
    </tr>
    <tr>
        <th>Classification model (without locality)</th>
        <th>98.32%</th>
        <th>76.80%</th> 
    </tr>
    <tr>
        <th>Classification model (with locality)</th>
        <th>96.7%</th>
        <th>76.39%</th> 
    </tr>
</table>

We observe that model with locality even though trained for fewer epochs and tested on lower number of points per sample, had less decrease in accuracy with rotated point cloud. Thus, our model was able to learn local features to some extent.  

#### Segmentation Task:

__Procedure:__ Similar to previous model, here also tried using K nearest neighbours to help the model learn local features. But due to memory issues (CUDA out of memory), I was not able to train the segmentation model with locality. Model architecture - local_seg_model() is given in models.py. 

Segmentation task would have given better insight on how well the model has learned local features when inferencing output for 30 degree rotated point cloud. 