![](C:\Users\User\Google Drive\Projects\time_series_anomaly_detection\result_tadgan.png)



# Time Series Anomaly Detection via Prediction and Reconstruction

This is a personal project to implement examples of two approaches to **time series anomaly detection**, one using **prediction** methods and one using **reconstruction** methods.

For each approach, we have selected a particular deep neural network model, and demonstrated the end-to-end procedure of anomaly detection using the [NYC taxi data](https://github.com/numenta/NAB/blob/master/data/realKnownCause/nyc_taxi.csv) from the [Numenta Anomaly Benchmark repository](https://github.com/numenta/NAB).

In both cases, with only a small effort of hyperparameter search, we obtain results with the desired properties of recovering most of the known anomalies (**high recall**) without raising too many false alarms (**acceptable precision**). [A sample result is illustrated at the top of the page.]



## Prediction Approach with DeepAR

This approach relies on a method of predicting (or forecasting) a segment of a time series based on its history. Ideally, if the predictions characterize expected behaviours, any significant divergence in the actual values would indicate an anomaly.

The prediction model we consider is [DeepAR](https://arxiv.org/abs/1704.04110), developed by Amazon and available in AWS SageMaker. DeepAR has the benefit of producing probabilistic predictions, offering a natural way to quantify the normality of each actual value.

For the demonstration, we use SageMaker SDK for DeepAR training and inference.

**[DEMO NOTEBOOK](https://colab.research.google.com/github/pokman/time_series_anomaly_detection/blob/main/demo_deepar.ipynb)**



## Reconstruction Approach with TadGAN

This approach relies on a method of reconstructing a time series segment from a low-dimensional latent space representation. Ideally, if only the non-anomalous parts are closely reconstructed, any significant divergence from the original values would indicate an anomaly.

The reconstruction model we consider is [TadGAN](https://arxiv.org/abs/2009.07769v3), developed by researchers at MIT. This GAN-based model strives for the reconstruction goal through an adversarial relation between its components -- generators learning to synthesize realistic data, and critics learning to differentiate real and synthetic data.

For the demonstration, we have re-implemented TadGAN in TensorFlow 2, in order to better utilize GPU acceleration and introduce modifications.

**[DEMO NOTEBOOK](https://colab.research.google.com/github/pokman/time_series_anomaly_detection/blob/main/demo_tadgan.ipynb)**

