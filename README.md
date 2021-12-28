# Modelling Perception in Autonomous Vehicles via 3D Convolutional Representations on LiDAR



LiDAR is assumed to be an exteroceptive sensor that allows the autonomous vehicles to have dynamic 3D scene perception of their surroundings. We employ an encoder-decoder architecture based on 3D-Convolutional layers called 3D Convolution Encoder-Decoder (3D-CED), together with a transfer learning strategy to extract a set of features from point clouds which are relevant in the context of autonomous driving. The resulting features allow to infer the future point cloud data and detect multiple abstraction level anomalies in controlled scenarios by utilizing a probabilistic switching dynamic model called High Dimensional Markov Jump Particle Filter (HD-MJPF). Moreover, a comparison is provided between piecewise linear, piecewise nonlinear, and nonlinear predictive models for anomaly detection at multiple abstraction levels. Our approach is evaluated with data collected from the LiDAR sensors of the autonomous vehicle while performing certain tasks in a controlled environment.

The block diagram of the proposed methodology is as follows;

![blockDiagram_revised](https://user-images.githubusercontent.com/56120865/147563831-b81e6afc-1405-47a6-bf3e-f4f918b956ce.jpg)



### Citation

If you find our work useful in your research, please cite our work:

```
@article{iqbal2021modeling,
  title={Modeling Perception in Autonomous Vehicles via 3D Convolutional Representations on LiDAR},
  author={Iqbal, Hafsa and Campo, Damian and Marin-Plaza, Pablo and Marcenaro, Lucio and G{\'o}mez, David Mart{\'\i}n and Regazzoni, Carlo},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```
