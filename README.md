# Predict Optimal Time to Purchase Flight Ticket

> We aim to find the optimal day to book a particular flight for which the flight cost will be minimum between the current date and date of departure. (CSE343 Machine Learning Project at Indraprastha Institue of Information Technology, Delhi)

This project was trained on data collected for top 7 domestic routes in India:

- Mumbai → New Delhi
- New Delhi → Mumbai
- New Delhi → Goa
- Bengaluru → New Delhi
- Mumbai → Goa
- Mumbai → Bengaluru
- New Delhi → Kolkata

Contributors: Shashwat Aggarwal (2018097), Shikhar Sheoran (2018099), Sidhant S Sarkar (2018102) [Group 4]

## Requirements

1. `Python 3.6+`
2. `selenium`
3. `bs4`
4. `scikit-learn`
5. `numpy`
6. `pandas`
7. `tqdm`
8. Google chrome and chrome driver version must match. Please check your google chrome browser version and download the chrome driver accordingly.
9. You can download the correction version of chrome driver from the link - http://chromedriver.chromium.org/downloads.

## Setup

1. `git clone https://github.com/shashwataggarwal/ml-flight-predictor.git`
2. `wget https://github.com/shashwataggarwal/ml-flight-predictor/releases/download/v1.1/processed_data.zip`
3. `unzip processed_data.zip -d ./ml-flight-predictor/Processed_Data`
4. `cd ml-flight-predictor`

## Data Scrapper

1. `cd ml-flight-predictor/Scrapper`
2. Update `main.py`
3. Run `main.py`
4. `Crontab` can be setup to automate the data collection pipeline
5. `searchandkill.sh` to kill stray process during automation

## Running Pre-trained Model

```
$python3 app.py
```

1. Follow the menu driven program.
2. `Output = 1 ` means buy ticket now
3. `Output = 0 ` means buy ticket at a later date

## File Descriptions

| Name           |                           Descripton                            |
| -------------- | :-------------------------------------------------------------: |
| app.py         |           Menu driven program to run pretrained model           |
| preprocess.py  |                    Preprocess scrapped data                     |
| train_clf.py   |         Train classification models and generate scores         |
| Weights        |             Weights folder containing saved models              |
| Data           |                 Contains orginal scrapped data                  |
| Processed_Data |        Contains preprocessed data (available as release)        |
| Graphs         |                      Contains graph images                      |
| \*.ipynb       | Jupyter notebooks containing EDA, grid search, graph generation |

## Queries

Contact shashwat18097@iiitd.ac.in, shikhar18099@iiitd.ac.in, sidhant18102@iiitd.ac.in