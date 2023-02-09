## Towards Interpretable Solar Flare Prediction with Attention-based Deep Neural Networks
In this work, we developed an attention-based deep learning model as an improvement over the standard CNN pipeline to perform full-disk binary flare predictions for the occurrence of ≥M1.0-class flares within the next 24 hours.

### Architecture of Standard CNN Pipeline based Full-disk Model

![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/no_attn_arch.png?raw=true)

### Architecture of the Attention-based Full-disk Model
![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/attn_arch.png?raw=true)

### Data Labeling
We labeled our data with a prediction window of 24 hours. The images are labeled based on the maximum peak X-ray flux values,
converted to NOAA/GOES flare classes observed in the next 24 hours. See figure below:
![alt text](https://github.com/chetrajpandey/fulldiskAttention/blob/main/readme_resoc/data_label.png?raw=true)
