# Website usability: Minimum number of participants for collecting reliable data from closed card sorting experiments

This is the code developed for [my undergraduate thesis (PDF)](https://ikee.lib.auth.gr/record/358314/files/Dougalis.pdf).

## Data Preprocessing

This folder contains the following:
- dataSkroutz.py: All the data gathered from the first closed card sorting experiment in JSON.
- dataTourismos.py: All the data gathered from the second closed card sorting experiment in JSON.
    - The data in JSON format was gathered using [this open soruce card sorting tool](https://github.com/indigane/cardsort) hosted on a personal server.
- JSONtoCSV.py: The script that transforms the data from JSON to a .csv file given the above mentioned as data.
- skroutz.csv: The .csv file created with the data from the first experiment.
- tourismos.csv: The .csv file created with the data from the second experiment.

## Data Analysis

This folder contains the data from the two experiments in the .csv files, the code used for the data analysis and the visualization of the results, as well as the results both in .csv and in graph form inside the "results" sub-folder:
- skroutz.csv: All the sorts gathered from 191 users in the first closed-type card sorting experiment.
- tourismos.csv: All the sorts gathered from 185 users in the second closed-type card sorting experiment.
- FINAL.py is the code developed to perform statistical analysis of the data and the visualization of the results. This analysis was performed using a resampling procedure and two analysis methods that calculated the correlation between the data of all the participants and data from 100 random samples for each possible sample size from 1 to the maximum N with a step of 1. The first method is called the Mantel test and it calculated the correlation coefficient of the distance matrices of the two sets, while the second is called the Element Centric Clustering Similarity Test and it calculated the degree of similarity of the non-overlapping clusters of the two sets.
- Mantel.py is an implementation of the Mantel test [https://github.com/jwcarr/mantel](https://github.com/jwcarr/mantel)