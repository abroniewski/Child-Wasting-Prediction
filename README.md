# Project Title

![Current Version](https://img.shields.io/badge/version-v0.1-blue)
![GitHub contributors](https://img.shields.io/github/contributors/abroniewski/README-Template)
![GitHub stars](https://img.shields.io/github/stars/abroniewski/README-Template?style=social)

## Table of contents

- [Getting Started](#getting-started)
  - [Tools Required](#tools-required)
  - [Installation](#installation)
- [Development](#development)
  - [Part 1: Name](#part-1-landing-zone)
    - [Subsection](#sub-section)
  - [Part 2: Name2](#part-2-formatted-and-exploitation-zone)
- [Running the App](#running-the-app)
- [Authors](#authors)
  - [Adam Broniewski](#adam-broniewski)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Getting Started

This project has a single branch: `main`

The project is small and will follow the structure below:

```
	IdelCompute-Data-Management-Architecture
	├── README.md
	├── .gitignore
	└── src
		├── all executbale script files
	└── docs
		├── support documentation and project descriptions
	└── data
		├── raw
		└── processed
```

## Running the App

Steps and commands for running the app will be included here

* Example steps:
  ```
    Example command
  ```

1. clone the project
2. install dependencies using pipfile
3. run main.py. 

This will use the data provided in [data/raw](https://github.com/abroniewski/README-Template/tree/main/data/raw) and run through the full data pipeline from landing zone, formatted zone and exploitation.

### Tools Required

You would require the following tools to develop and run the project:

* Access to UPC Virtual Machine
* HDFS

### Installation

Installation steps for virtual environment and HDFS are locate in [here in /docs](https://github.com/abroniewski/IdleCompute-Data-Management-Architecture/tree/main/docs)

## Development

The goal of this project is to develop a theoretical understanding of data management architecture along with a practical implementation of design choices. The work is completed in 2 stages:

1. Landing zone architecture and external data source identification
2. Formatted and exploitation zone architecture

### Part 1: Landing Zone

The scope is defined by breaking out all the relevant compoenents that need to be implemented.

#### Sub-heading

We will use datasets from kaggle that are ~5 MB with ~30 attributes. Datasets will be labelled data with a classification problem challenge.

- CSV
- JSON
  - Potential 1: [arXiv scholarly articles dataset - 3.34GB](https://www.kaggle.com/datasets/Cornell-University/arxiv)
  - Potential 2: [Automatic Ticket Classification - 83.4MB](https://www.kaggle.com/datasets/arunagirirajan/automatic-ticket-classification/code)
- **Assumption:** Dataset could have multiple other sources, types, and formats.

### Part 2: Formatted and Exploitation Zone

This scope has not yet been developed.

## Authors

#### Adam Broniewski

* [GitHub](https://github.com/abroniewski)
* [LinkedIn](https://www.linkedin.com/in/abroniewski/)
* [Website](https://adambron.com)

## License

`IdleCompute Data Management Architecture` is open source software [licensed as MIT][license].

## Acknowledgments

....

[//]: #
[license]: https://github.com/abroniewski/LICENSE.md
