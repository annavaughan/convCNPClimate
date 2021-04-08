# convCNPClimate
Implementation of convolutional conditional neural processes for statistical downscaling following https://gmd.copernicus.org/preprints/gmd-2020-420/. 
![alt text](graz_tmax.png)

## Examples
Trained models are included in examples/. The notebooks
- precip_marginal_prediction.ipynb
- tmax_marginal_prediction.ipynb

Demonstrate using trained models to make predictions at held out stations

## Models
Source code for different sections of the paper can be found in
- (marginal distributions): models used in this section are in models/elev_models
- (multivariate downscaling): models for predicting p(precipitation|temperature) in models/multivar_models_tmax_init, and for p(temperature|precipitation) models/multivar_models_precip_init

## References
Elements of the code in this implementation were adapted from Yann Dubois' neural process repo (https://yanndubs.github.io). Specifically, these are:
- Parts of the Encoder class, particularly the ProbabilityConverter class for converting the density channel to a probability.
- The implementation of the ResNet decoder architecture, including ResConvBlock and function to make the convolution depth separable were adapted from Yann's work.
