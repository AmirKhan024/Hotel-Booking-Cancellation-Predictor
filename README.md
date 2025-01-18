# Hotel Booking Cancellation Predictor

## Project overview
 Hotel bookings are a significant part of the hospitality industry. One of the challenges faced by hotels is the cancellation of bookings, which leads to lost revenue and operational inefficiencies. Predicting booking cancellations can help hotels manage their resources better and mitigate losses. In this project, we aim to predict whether a hotel booking will be canceled or not, based on various features such as hotel type, lead time, previous cancellations, and more.

![image](https://github.com/user-attachments/assets/725e7b02-afc5-48e8-88cf-75f35bde94bc)


 ## Objective 
 The objective of this project is to create a machine learning model that can predict whether a hotel booking will be canceled or not, using a variety of features available in the dataset. The project goes through the following steps:
 
-- Data cleaning and preprocessing
 
-- Exploratory Data Analysis (EDA) to understand patterns and trends

-- Building a machine learning model

-- Evaluating model performance and improving it through techniques like SMOTE and Hyperparameter tuning

-- Deploying the model in a Flask-based web application for real-time predictions

## Workflow
#### 1. Data Cleaning:
The initial step involved cleaning the dataset by handling missing values, fixing data inconsistencies, and converting data types where necessary. This ensures the data is in a usable form for further analysis.

#### 2. Exploratory Data Analysis (EDA)
A comprehensive EDA was performed to understand the relationships between various features and the target variable is_canceled. 

Key insights included:

-- The distribution of bookings by hotel type and cancellation status.

-- The trend of cancellations across different months and countries.

-- Insights into lead time, family size, market segments, and booking behavior, which all contribute to understanding cancellation patterns.

#### 3. Model Building
Initially, a Decision Tree model was created, which showed some useful results but lacked performance. Subsequently, we switched to the Random Forest model, which improved the prediction accuracy significantly. Random Forest is an ensemble learning method, which helps in reducing overfitting and enhancing the model's generalization.

#### 4. Handling Class Imbalance
As the dataset had an imbalanced distribution of canceled vs. non-canceled bookings, we used SMOTEENN (Synthetic Minority Over-sampling Technique + Edited Nearest Neighbors) to generate synthetic samples and balance the dataset. This helps in improving the model's performance, especially for the minority class.

#### 5. Hyperparameter Tuning
To improve the performance of the Random Forest model, hyperparameter tuning was performed using GridSearchCV. This step optimizes the model's parameters to achieve the best possible performance.

#### 6. Model Saving
Once the model was trained and tuned, it was saved using joblib, which allows us to reuse the model without retraining every time.

#### 7. Flask Web Application
A Flask-based web application was created to allow users to interact with the model. The app allows users to input hotel booking details, and the model predicts whether the booking will be canceled or not. 

The Flask app includes:

-- Input forms for users to enter data such as hotel type, lead time, special requests, and more.

-- Model prediction output displayed on the webpage.

-- Clear form functionality to reset the input fields.

#### 8. Deployment
The model and Flask app were deployed on a cloud platform (e.g., Render) for real-time predictions. The app can be accessed online, where users can enter booking information and get predictions on whether the booking will be canceled.

## Hosted Online
This project is hosted online and can be accessed via the following link: 

[Hotel Booking Cancellation Prediction App](https://hotel-booking-cancellation-predictor.onrender.com/)
