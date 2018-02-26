# Noise Resistant Adaboost

* Studying the distribution of weights in Adaboost on noisy datasets.
* Regularize the weights to create a noise resistant Adaboost version.
* Capable of detecting wrongly-labeled data points.

## Visualization

We look at the weight distribution of Adaboost, in log-scale compared to the initial weighting scheme.

Left image : Adaboost weight distribution

Righ image : Adaboost Cumulative weight distribution

Both informations tells us how many points account for the most loss weight. On a noisy setting, very few points (the outliers - mislabeled points) take all the weights. Our regularization find those points and penalize them.

1. Adaboost

![Adaboost Weights](images/adaboost_weight_distribution.png =250x) ![Adaboost Cumulative Weights](images/adaboost_cumulative_weight_distribution.png =250x)

2. Noise resistant Adaboost

![NRAdaboost Weights](images/newadaboost_weight_distribution.png =250x) ![NRAdaboost Cumulative Weights](images/newadaboost_cumulative_weight_distribution.png=250x)

## Data

We looked for datasets possibly subject to wrong labeling. Wrong labeling can happens on datasets hard for a human, even for a specialist, to label.

We took 3 medical datasets :
* Chronic Kidney Disease
* Liver disorder
* Prima indians diabetes

as well as some datasets subject to label noise :
* twonorm
* spambase
* waveform

## Procedure

Given a clean train/test set, let's add noise label to the train set, learn Adaboost and report its performance. Then we compared to our Noise Resistant Adaboost and report the performance increase.
