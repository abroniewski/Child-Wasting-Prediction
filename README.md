# Improving Child Wasting Prediction for Zero Hunger Labs

![Current Version](https://img.shields.io/badge/version-v0.5-blue)
![GitHub contributors](https://img.shields.io/github/contributors/abroniewski/Child-Wasting-Prediction)
![GitHub stars](https://img.shields.io/github/stars/abroniewski/README-Template?style=social)
![GitHub activity](https://img.shields.io/github/commit-activity/w/abroniewski/Child-Wasting-Prediction?logoColor=brightgreen)

## Table of contents

- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [Tools Required](#tools-required)
- [Development](#development)
- [Authors](#authors)
  - [Adam Broniewski](#adam-broniewski)
  - [Chun Han (Spencer) Li](#chun-han-spencer-li)
  - [Himanshu Choudhary](#himanshu-choudhary)
  - [Luiz Fonseca](#luiz-fonseca)
  - [Tejaswini Dhupad](#tejaswini-dhupad)
  - [Zyrako Musaj](#zyrako-musaj)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

This project has a stable branch called `main`. Different branches are created for during major changes to the model or for feature development.

The project follows the structure below:

```
	Child-Wasting-Prediction
	├── README.md
	├── LICENSE.md
	├── .gitignore
	└── notebooks
	└── src
		├── all executbale script files
	└── docs
		└── support documentation and project descriptions
	└── data
		├── raw
		└── processed
```
## Tools Required
- Python
- Pip

## Running the App

1. Clone the project to a directory of your choice
    ```bash
    git clone https://github.com/abroniewski/Child-Wasting-Prediction.git
    ```
2. Pipenv is used to manage dependencies. If you do not have pipenv installed, run the following:
    ```bash
    pip install pipx
    pip install pipenv
    ```
3. Install dependencies using the included pipfile. Run the following from the parent directory.
    ```bash
    pipenv install
    pipenv run clean_notebook
    ```
4. Once all dependencies are installed, we can run the main file. 
    ```bash
    python3 src/dc3_main.py
    ```
    The parameters "**old**" and "**new**" can be passed with the above script to generate **baseline** and **modified approach** results. By default it isrunning on our new model.  


This will run the full data-preperation, model building and prediction generation using the data provided in [data](https://github.com/abroniewski/Child-Wasting-Prediction.git/data).

### Tools Required

No tools currently specified

## Development

The objective of this project is to work with various ****stakeholders**** to understand their needs and the impact modeling choices have on them. Additionally, the design choices are assessed through a lens of **ethical impact**.

The objective of the **data analytics model** to explore whether a better (more accurate or more generally applicable) forecasting model for predicting child watage can be developed, by researching one of the following two questions:
1. Is the quality of the additional data sources sufficient to improve or expand the existing GAM forecasting model? Are there additional, public data sources that allow you to improve or expand the existing GAM forecasting model?
2. Are there other techniques, different than additional data sources, that would lead to an improved GAM forecasting model on the data used in the development of the original GAM forecasting model?


## Results
```
python3 src/dc3_main.py old
```
- Total no of district after preproecssing are - 55 
- number of observations for training are - 275 and for testing are - 110 
- MAE(Mean Absolute Error) score for old model on training data is - 0.021391660328296973
- MAE(Mean Absolute Error) score for old model on test data is - 0.0512394083425881 


```
python3 src/dc3_main.py new
```
- Total no of district after preproecssing are - 55 
- number of observations for training are - 275 and for testing are - 110  
- MAE(Mean Absolute Error) score for new model on training data is - 0.021117145120009326
- MAE(Mean Absolute Error) score for new model on test data is - 0.05013554272083966


## Authors

#### Adam Broniewski [GitHub](https://github.com/abroniewski) | [LinkedIn](https://www.linkedin.com/in/abroniewski/) | [Website](https://adambron.com)
#### Chun Han (Spencer) Li
#### Himanshu Choudhary
#### Luiz Fonseca
#### Tejaswini Dhupad [GitHub](https://github.com/tejaswinidhupad) | [LinkedIn](https://www.linkedin.com/in/tejaswinidhupad/) 
#### Zyrako Musaj

## License

`Child-Wasting-Prediction` is open source software [licensed as MIT][license].

## Acknowledgments

....

[//]: #
[license]: https://github.com/abroniewski/LICENSE.md
