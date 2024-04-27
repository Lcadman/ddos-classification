# ddos-classification

For cs535 term project spring 2024

This repo contains code to perform binary classification on the CIC-DDoS2019 dataset. The file `ddos_process_dataset.py` will perform the dataset preprocessing which consists of sample balancing and feature selection. Next, the two files `ddos_binary_classifier_MLP.py` and `ddos_binary_classifier_1DCNN.py` will perform training of both a multilayer perceptron and an 1-dimensional for the preprocessed dataset respectively (each can be launched distributedly with the two Slurm files). Next, the files `ddos_sensitivity_analysis_MLP.py` and `ddos_sensitivity_analysis_1DCNN.py` will allow the user to perform sensitivity analysis for both models by passing the name of the attack class (or benign) as the first command line argument. Finally, the files `ddos_metrics_MLP.py` and `ddos_metrics_1DCNN.py` will perform metric analysis including precision, recall, F1-score, and inference time for both of the models.

## References

<https://ieeexplore.ieee.org/document/8888419>
<https://www.sciencedirect.com/science/article/pii/S016740482300161X>
<https://ieeexplore.ieee.org/document/9476932>
<https://ieeexplore.ieee.org/document/9664685>
<https://www.researchgate.net/publication/358884690_Using_Machine_Learning_Techniques_to_Detect_Attacks_in_Computer_Networks>
<https://www.mdpi.com/2079-9292/10/11/1257>
<https://www.kaggle.com/datasets/rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files>
<https://github.com/ryanjob42/CSX75-HPC-Demo/tree/main?>
