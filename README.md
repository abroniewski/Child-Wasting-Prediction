# Improving Child Wasting Prediction for Zero Hunger Labs

![Current Version](https://img.shields.io/badge/version-v0.5-blue)
![GitHub contributors](https://img.shields.io/github/contributors/abroniewski/Child-Wasting-Prediction)
![GitHub stars](https://img.shields.io/github/stars/abroniewski/README-Template?style=social)
![GitHub activity](https://img.shields.io/github/commit-activity/w/abroniewski/Child-Wasting-Prediction?logoColor=brightgreen)

## Table of contents

- [Getting Started](#getting-started)
- [Requirments](#tools-required)
- [Running the code](#running-the-code)
- [Development](#Development)
- [Authors](#authors)
  - [Adam Broniewski](#adam-broniewski)
  - [Chun Han (Spencer) Li](#chun-han-spencer-li)
  - [Himanshu Choudhary](#himanshu-choudhary)
  - [Luiz Fonseca](#luiz-fonseca)
  - [Tejaswini Dhupad](#tejaswini-dhupad)
  - [Zyrako Musaj](#zyrako-musaj)
- [License](#license)

## Getting Started

This project contains is part of Data challenge 3: Improving Child Wasting Prediction for Zero Hunger Labs.

The project code follows the structure below:

```
	Child-Wasting-Prediction
	├── README.md
	├── LICENSE.md
	├── requirments.txt
	└── src
		├── all executbale scripts
	└── results
		├── contains all results in csv files
	└── data
		└── contains raw and processed data
```
## Requirments
- pandas
- sklearn
- numpy
- matplotlib

## Running the Code

### Main Code

1. Unzip the folder Child-Wasting-Prediction.zip and switch to the folder
    ```
    cd Child-Wasting-Prediction
    ```

2. Install the dependencies:
    ```
    pip install -r requirments.txt
    ```

3. Once all dependencies are installed, we can run the ZHL Baseline model file **(model_1)**. The output is saved in result folder. 
    ```
    python3 src/Baseline.py
    ```
    **Expected Outcome -** 
    ```
    no. of trees: 74
    max_depth: 6
    columns: ['district_encoded']
    0.05629900026844118 0.849862258953168
    ```

4. The below code will generate the extracted features data file from conflict data. The output data is saved in acled folder.

```
python3 src/feature_engineering.py
```

5. The below code will run our main file, it will generate baseline and conflict data model (model with conflict features) results.  
    ```
    python3 src/dc3_main.py {model_2/model_3}
    ```
    The parameters "**model_2**" and "**model_3**" can be passed with the above script to generate **our baseline** and **combined conflict data model** results respecively. By default it is running on our combined conflict data model (model_3).  

    **Expected Outcome with model_2 -** 
    ```
    Total no of district after preproecssing are - 55 
    number of observations for training are - 275 and for testing are - 110 
    MAE(Mean Absolute Error) score for model_2 model on training data is - 0.021391660328296973
    MAE(Mean Absolute Error) score for model_2 model on test data is - 0.0512394083425881 
    ```
    **Expected Outcome with model_3 -** 
    ```
    Total no of district after preproecssing are - 55 
    number of observations for training are - 275 and for testing are - 110 
    MAE(Mean Absolute Error) score for model_3 model on training data is - 0.021117145120009326
    MAE(Mean Absolute Error) score for model_3 model on test data is - 0.05013554272083966 
    ```

6.  The below code combine all the results from model_1 (ZHL Baseline),model_2 (Our Baseline) and model_3 (conflict data combined model). 
    ```
    python3 src/combine_results.py
    ```

This above codes will run the full data-preperation, model building and prediction generation using the data provided in [data](https://github.com/abroniewski/Child-Wasting-Prediction.git/data).


## Development

The objective of this project is to work with various ****stakeholders**** to understand their needs and the impact modeling choices have on them. Additionally, the design choices are assessed through a lens of **ethical impact**.

The objective of the **data analytics model** to explore whether a better (more accurate or more generally applicable) forecasting model for predicting child watage can be developed, by researching one of the following two questions:
1. Is the quality of the additional data sources sufficient to improve or expand the existing GAM forecasting model? Are there additional, public data sources that allow you to improve or expand the existing GAM forecasting model?
2. Are there other techniques, different than additional data sources, that would lead to an improved GAM forecasting model on the data used in the development of the original GAM forecasting model?


## Authors

#### Adam Broniewski [GitHub](https://github.com/abroniewski) | [LinkedIn](https://www.linkedin.com/in/abroniewski/) | [Website](https://adambron.com)
#### Chun Han (Spencer) Li
#### Himanshu Choudhary
#### Luiz Fonseca
#### Tejaswini Dhupad [GitHub](https://github.com/tejaswinidhupad) | [LinkedIn](https://www.linkedin.com/in/tejaswinidhupad/) 
#### Zyrako Musaj

## License

`Child-Wasting-Prediction` is open source software [licensed as MIT][license].