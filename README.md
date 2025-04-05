# Flight Price Prediction System

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Model Training](#model-training)
   - [Model Evaluation](#model-evaluation)
   - [Deployment with Streamlit](#deployment-with-streamlit)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Overview

The Flight Price Prediction System is a machine learning-based application designed to predict flight prices based on various input features. This system aims to help users make informed decisions by providing accurate price predictions for flights. The application uses historical flight data to train a regression model, which can then predict prices for new flight queries.

## Features

- **Data Cleaning and Preprocessing**: Automatically handles data cleaning, including removal of unnecessary columns, conversion of date formats, and transformation of categorical variables.
- **Exploratory Data Analysis (EDA)**: Provides visualizations to understand the distribution of flight prices and relationships between different features.
- **Model Training**: Utilizes regression algorithms such as Linear Regression and Random Forest to build the prediction model.
- **Model Evaluation**: Evaluates model performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Hyperparameter Tuning**: Implements GridSearchCV for hyperparameter optimization to improve model accuracy.
- **Deployment with Streamlit**: Offers a user-friendly web interface for real-time price predictions based on user inputs.

## Dataset

The dataset used for this project is the [Goibibo Flight Data](https://www.kaggle.com/datasets/iamavyukt/goibibo-flight-data), available on Kaggle. It contains information about flights, including airline, departure and destination cities, number of stops, duration, and price.





   ```

## Usage

### Data Preprocessing

The `preprocess.py` script handles data cleaning and preprocessing. It performs the following tasks:
- Removes unnecessary columns.
- Converts date columns to datetime format and extracts meaningful features.
- Transforms categorical variables into numerical representations.
- Converts duration and price columns to numerical formats.

Run the script using:
```bash
python preprocess.py
```

### Exploratory Data Analysis (EDA)

The `eda.py` script provides visualizations to understand the data better. It includes:
- Distribution of flight prices.
- Relationship between flight duration and price.
- Impact of stops on flight price.

Run the script using:
```bash
python eda.py
```

### Model Training

The `train.py` script trains the prediction model using the preprocessed data. It splits the dataset into training and test sets and trains a regression model.

Run the script using:
```bash
python train.py
```

### Model Evaluation

The script evaluates the model using metrics like MAE and RMSE. It also performs hyperparameter tuning using GridSearchCV to optimize the model's performance.

### Deployment with Streamlit

The `app.py` script creates a Streamlit application where users can enter flight details and get real-time price predictions. The app displays the predicted price based on user inputs and allows users to visualize model accuracy with evaluation metrics.

Run the Streamlit app using:
```bash
streamlit run app.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the contributors and maintainers of the dataset and libraries used in this project.
- Special thanks to the open-source community for their continuous support and contributions.

---

This detailed README file provides comprehensive information about the Flight Price Prediction System, including installation instructions, usage guidelines, and contribution details. You can customize it further based on your specific requirements and project structure.
