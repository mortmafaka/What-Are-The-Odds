# Serving up Stats

First off, a large thanks to Jeff Sackmann for making avaliable an open-source tennis dataset.

# The Repository

This repository contains a copy of ATP match data from the Jeff Sackmann dataset from the years 2000 to 2023. This is used in the 'data_preprocessing' notebook. 

The 'processed_data' folder contains two different csv files. The 'sortedh2h.csv' file is the first checkpoint before any of the feature engineering happen. It is a combination of all the individual csv files contained in the 'data' folder. The csv file has been cleaned and all irrelevant columns dropped. The 'ml_workflow' notebook is the different models that my brother and I used to create the prediction algorithms. 