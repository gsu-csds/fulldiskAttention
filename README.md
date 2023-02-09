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

---
### Source Code Documentation

##### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". 
There are two functions inside this module.
First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : [Helioviewer](https://api.helioviewer.org/docs/v2/api/api_groups/jpeg2000.html)
Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms

##### 2. data_labeling:
Run python labeling.py : Contains functions to generate labels, binarize, filtering files, and creating 4-fold CV dataset.
Reads goes_integrated_flares.csv files from data_source.
Generated labels are stored inside data_labels.
---