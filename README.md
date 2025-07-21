# Loan Default Risk Prediction with Business Cost Optimization

## Overview

This project focuses on predicting the likelihood of loan default and optimizing classification thresholds to minimize business cost. Instead of relying on conventional performance metrics alone (like accuracy or AUC), this project integrates a cost-sensitive approach where financial consequences of false positives and false negatives are explicitly considered. 

## Objective
The main objective is twofold:
1. Predict the likelihood of a loan applicant defaulting using machine learning.
2. Optimize the decision threshold to minimize the total cost to the business from false positives and false negatives.

## Dataset Description
The dataset used includes detailed information about loan applicants, including demographic, financial, and credit history features.

**Key Features:**
- `person_age`: Age of the applicant
- `person_income`: Annual income
- `person_home_ownership`: Home ownership status (e.g., RENT, OWN)
- `person_emp_length`: Length of employment (in months)
- `loan_intent`: Purpose of the loan (e.g., EDUCATION, MEDICAL)
- `loan_grade`: Loan grade assigned by the institution
- `loan_amnt`: Loan amount requested
- `loan_int_rate`: Interest rate assigned
- `loan_status`: Whether the applicant defaulted or not (1 = defaulted)
- `loan_percent_income`: Loan amount as percentage of income
- `cb_person_default_on_file`:  Previous default record
- `cb_person_cred_hist_length`: Credit history length in years

## My Approach

1. **Data Preprocessing**
   - **Missing Values**: Handled any missing or inconsistent entries.
   - **Categorical Encoding**:
      - For **Logistic Regression**, applied one-hot encoding.
     - For **CatBoost**, used raw categorical variables (CatBoost handles these internally).
   - **Feature Scaling**:
      - Applied standardization to numerical features for Logistic Regression.
     - No scaling needed for CatBoost.

2. **Model Training**
   
     Trained and compared two models:
   - **Logistic Regression**
      - Applied after preprocessing (encoding + scaling).
     - Offers interpretability and serves as a baseline.
   - **CatBoostClassifier**
      - Chosen for its performance on tabular data and native support for categorical features.
     - Tuned hyperparameters and trained on the same dataset.

3. **Business Cost Function** 
   - Defined a cost function based on business logic:
     - **False Positive (FP)**: $1,000 (missed opportunity from rejecting a good applicant)
     - **False Negative (FN)**: $10,000 (loss from approving a defaulter)

4. **Threshold Optimization**
   - Evaluated thresholds from 0 to 1 using the predicted probabilities.
   - Calculated total business cost for each threshold using:<br>
       total_cost = FP × cost_fp + FN × cost_fn
   - Selected the threshold that minimized the total cost.


5. **Model Evaluation**
   - Confusion matrix and classification metrics (precision, recall, F1-score) were analyzed at the optimized threshold.
   - ROC AUC was used to confirm model discrimination performance.
   - A cost vs. threshold plot was created to visually highlight the impact of threshold selection.

## Key Insights
- **Default Threshold (0.50):**
   - Cost: $2,369,000

- **Optimized Threshold (0.21):**
   - Cost: $1,802,000
   - Cost Saved: $567,000

- Despite some increase in false positives, the large reduction in false negatives significantly reduced overall business loss.
- Optimizing threshold based on cost delivers better real-world outcomes than relying on standard metrics alone.


## Technologies Used

- Python (NumPy, Pandas, Matplotlib, Scikit-learn)
- LogisticRegression
- CatBoost
- Jupyter Notebook

## Skills Demonstrated

- Data preprocessing and feature engineering
- Supervised Machine Learning for classification problems
- Cost-sensitive decision making in predictive modeling
- Threshold optimization using custom evaluation metrics
- Business impact analysis using financial costs
- Data visualization and interpretation of model performance
- Model evaluation with classification reports and ROC analysis
