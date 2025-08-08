# Titanic Survival Prediction

## Project Overview

This project analyzes the classic Titanic dataset from Seaborn to predict whether a passenger survived the disaster. The notebook walks through the entire data science workflow, including data loading, exploratory data analysis (EDA), data preprocessing, feature engineering, model training, and evaluation. Two machine learning models, XGBoost Classifier and Random Forest Classifier, are built and compared to determine the best model for this prediction task.

## Dataset

The analysis uses the `titanic` dataset, which is available directly through the Seaborn library. It contains information about individual passengers, such as their age, sex, passenger class, fare paid, and whether they survived.

## Key Findings from Exploratory Data Analysis

The initial analysis of the data revealed several key insights that influence survival chances:

* **Passenger Class:** Survival rates were strongly correlated with passenger class. First-class passengers had a significantly higher chance of survival compared to those in second and third class. The majority of non-survivors were from the third class.

* **Gender:** Gender was a critical factor. A much higher proportion of females survived compared to males, confirming the "women and children first" protocol.

* **Embarkation Port:** The majority of passengers boarded from Southampton. Passengers from Cherbourg had a slightly higher survival rate compared to those from Southampton and Queenstown.

* **Fare and Family:** Higher fares correlated positively with survival, which is also linked to being in a higher passenger class. The presence of family members (not being alone) also showed a slight positive correlation with survival.


## Project Workflow

The notebook follows these steps:

1.  **Data Loading:** The Titanic dataset is loaded into a Pandas DataFrame.
2.  **Exploratory Data Analysis (EDA):** Visualizations are created using Matplotlib and Seaborn to understand relationships between different features and survival rates.
3.  **Data Preprocessing & Feature Engineering:**
    * Handled missing values in `age`, `deck`, and `embark_town` columns using median imputation and filling with appropriate placeholders.
    * Converted categorical features (`sex`, `alone`, `class`, `deck`, `embark_town`) into numerical format using Label Encoding, mapping, and One-Hot Encoding.
    * Dropped redundant or unnecessary columns.
4.  **Model Training:**
    * The dataset was split into training (80%) and testing (20%) sets.
    * Two models were trained: **XGBoost Classifier** and **Random Forest Classifier**.
    * `GridSearchCV` was employed to perform hyperparameter tuning and find the optimal parameters for both models.
5.  **Model Evaluation:**
    * The models were evaluated on the test set using classification reports and confusion matrices.
    * Both models performed well, but the XGBoost Classifier showed a slight edge in performance.
6.  **Model Saving:** The trained and optimized models were saved to `.pkl` files using `joblib` for future use.

## Modeling Results

Both models achieved good accuracy, with XGBoost performing slightly better.

| Model | Accuracy | Precision (Survived) | Recall (Survived) | F1-Score (Survived) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **83%** | 0.82 | 0.76 | 0.79 |
| **Random Forest** | 82% | 0.82 | 0.72 | 0.76 |

### Confusion Matrix Comparison

-   **XG Boost:**
    ```
    [[93 12]
     [18 56]]
    ```
-   **Random Forest:**
    ```
    [[93 12]
     [21 53]]
    ```

The XGBoost model had fewer False Negatives (18 vs. 21), meaning it was slightly better at correctly identifying passengers who survived. Both models had the same number of False Positives (12).

## Conclusion

The analysis successfully identified key predictors of survival on the Titanic. A machine learning pipeline was built to preprocess the data and train predictive models. The final **XGBoost model achieved an accuracy of 83%** on the test data, demonstrating its effectiveness in this classification task.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install the required libraries:**
    ```
    pip install pandas matplotlib seaborn numpy scikit-learn xgboost joblib
    ```

3.  **Run the Jupyter Notebook:**
    ```
    jupyter notebook "titanic (1).ipynb"
    ```
