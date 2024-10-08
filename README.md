

<div align="center">

# Spatio-Temporal Deep Learning for Final Infarct Prediction using Acute Stroke CT Perfusion Data

  
</div>

TensorFlow implementation of our method for the ISLES 2024 Challenge: "Spatio-Temporal Deep Learning for Final Infarct Prediction using Acute Stroke CT Perfusion Data."

## Abstract
Accurate prediction of the tissue outcome is crucial for guiding treatment decisions in acute ischemic stroke (AIS). Spatio-temporal (4D) Computed Tomography Perfusion (CTP) provides detailed insights into cerebral blood flow dynamics, which are essential for predicting final infarct regions. However, its high-dimensional and noisy nature presents challenges for direct prediction. In this study, we evaluate a deep learning model that fully leverages 4D CTP data for predicting tissue outcomes. The model integrates a shared-weight convolutional neural network (CNN) encoder, a Transformer encoder, and a CNN decoder to capture both spatial and temporal dependencies within the data. We evaluated this approach on a multicenter dataset of 143 patients from the ISLES 2024 challenge. The results reveal a Dice score of 0.20, an absolute volume difference of 17 ml, a lesion count difference of 19, and a lesion-wise F1-Score of 0.02, underscoring both the potential and challenges of directly utilizing 4D CTP data for final infarct prediction.

<p align="center">
<img src="https://github.com/kimberly-amador/ISLES24-PrediCTP/blob/main/figures/model_architecture.png" width="750">
</p>


## Usage

#### Installation

Recommended environment:

- Python 3.8.1
- TensorFlow GPU 2.4.1
- CUDA 11.0.2 
- cuDNN 8.0.4.30

To install the dependencies, run:

```shell
$ git clone https://github.com/kimberly-amador/ISLES24-PrediCTP
$ cd ISLES24-PrediCTP
$ pip install -r requirements.txt
```

#### Data Preparation
1. Download the ISLES24 dataset from [the official challenge page](https://isles-24.grand-challenge.org/dataset/) on the Grand Challenge website.
2. Preprocess the dataset using [the included preprocessing script](python/data_processing/preproc.py).

#### Model Training

1. Modify the model configuration. The default configuration parameters are in `./model/config.py`.
2. Run `python ./model/main_unimodal.py` to train the model.

#### Inference and Evaluation
1. Convert the model output back to images using [the included postprocessing script](python/data_processing/postproc.py).
2. Visualize the images using your favorite image viewer, or calculate the metrics used in the ISLES24 challenge using [the included evaluation script](python/evaluation/evaluate_challenge_metrics.py).

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
Part of the code is adapted from open-source codebase:
* Transformer: https://github.com/keras-team/keras-io/blob/master/examples/vision/video_transformers.py
* Dice Loss: https://github.com/voxelmorph/voxelmorph/blob/legacy/ext/neuron/neuron/metrics.py
* Evaluation Metrics: https://github.com/ezequieldlrosa/isles24/tree/main
