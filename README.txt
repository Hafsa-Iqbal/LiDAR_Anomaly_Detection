REQUIREMENTS:
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
