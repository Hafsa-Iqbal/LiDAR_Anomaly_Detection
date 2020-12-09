#####################################
FIRST PART (ENCODER AND LSTM)
#####################################
REQUIREMENTS (MATLAB):
MATLAB r2020a
deep learning toolbox
computer vision toolbox

All datasets employed in this project are available in the following link:

https://drive.google.com/drive/u/1/folders/1jTtc1648Wro-nQkFWMVm54NjkZXsk8Rk

1- Create a new folder called "dataset" and unzip all dataset files in there

2- Run "Pre_trainednetwork_Modelnet40.m" to generate a pretrained network.
   This process may take many hours, we have uploaded an already pretrained
   network called "modelnet40PretrainNetwork.mat" so you can skip this step

3- Run "TransferLearning.m" to provide initial information of our data to the network using transfer learning

4- Run "FeatureExtractioniCAB.m" to get the features of the training and testing data

5- Run "LSTM.m" to compute the prediction

#####################################
SECOND PART (DECODER)
#####################################
REQUIREMENTS (PYTHON):
python 3.6.9
tensorflow 2.3.1
numpy 1.17.3
h5py 2.10.0
mat4py 0.4.3
matplotlib 3.1.0

1- Run the algorithm "reconstruction_net_3D.py" to generate the first version of
the decoder network "network_lr0.0001_mse_3D.h5" and its log file "network_lr0.0001_mse_3D.pkl"

2- Run the algorithm "reconstruction_net_3D_fine_tuning.py" to generate the fine-tuned version
of the decoder network "network_lr0.0001_mse_3D_fine_tuned.h5" and its log data "network_lr0.0001_mse_3D_fine_tuned.pkl"

Note: The folder "outputs_decoder/" already contains a trained version of the networks produced in points 1 and 2

3- Run the algorithm "network_outputs_saving.py" to obtain the reconstructed test data based on the initial decoder
and the fine-tuned version. You can find the reconstructions of both networks in the following links:

Initial decoder test outputs (point clouds):
https://drive.google.com/drive/folders/1hsNWCbSLEccnQy6yeEQqz7xAPwCWidIy

Fine-tuned decoder test outputs (point clouds):
https://drive.google.com/drive/folders/1DiOBiypgBHYc2lK8NEMmNuRC5OUUp0iM?usp=sharing

4- Run the algorithm "results_analysis.py" to generate qualitative results to evaluate the trained networks
