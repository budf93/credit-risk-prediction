# 🏦 Credit Risk Prediction Model (Lending Company Project)

## **1. Introduction and Goal**

This project's objective was to develop a robust machine learning solution for a lending company to predict the **credit risk** of loan applicants. The ultimate goal is a **Binary Classification** model that predicts whether a loan is likely to become a "Bad Loan" (Charged Off, Default, Late) versus a "Good Loan" (Fully Paid, Current).

The model and resulting insights are intended to serve as a new, data-driven system for underwriting and portfolio management.

-----

## **2. Methodology & Data Preparation**

The solution followed the standard Data Science pipeline, with crucial steps taken to manage data quality and severe class imbalance.

### **2.1. Feature Engineering & Cleaning**

| Step | Action Taken | Rationale |
| :--- | :--- | :--- |
| **Imputation** | Missing **Numerical** values filled with **Median**. Missing **Categorical** values filled with **Mode**. | Provided clean data while maintaining distribution robustness against outliers. |
| **Scaling** | All numerical features (e.g., `loan_amnt`, `dti`) were scaled using **StandardScaler**. | Ensured all features contributed equally to distance-based and gradient-descent models. |
| **High Cardinality** | Features with $\le 20$ unique values (e.g., `purpose`, `grade`) were **One-Hot Encoded (OHE)**. | Preserved information for low-cardinality features. |
| **Low Cardinality** | Features with $> 20$ unique values (e.g., `zip_code`, `addr_state`, date columns) were **Label Encoded (LE)**. | Reduced feature space complexity and computational load. |
| **Leakage/Redundancy** | Highly correlated features ($\text{corr} > 0.90$) and outcome-dependent variables (e.g., `total_pymnt`, `loan_status` components) were removed. | Mitigated multicollinearity and prevented target leakage. |

### **2.2. Target Definition & Imbalance Handling**

  * **Target Variable ($\mathbf{y}$):** A new binary column, `is_bad_loan`, was derived from `loan_status`.
      * **Class 0 (Good):** Fully Paid, Does Not Meet Policy: Fully Paid, **Current**.
      * **Class 1 (Bad):** Charged Off, Default, Late, **In Grace Period**.
  * **Data Split:** Data was split into 80% Training ($\mathbf{X}_{train}$) and 20% Testing ($\mathbf{X}_{test}$) using **Stratified K-Fold** to maintain the integrity of the minority class proportion across the sets.
  * **Imbalance Ratio ($\text{IR} \approx 7.4$):** The severe class imbalance was handled by injecting the calculated **$\text{scale\_pos\_weight}$** into the boosting models.

-----

## **3. Key Exploratory Data Analysis (EDA) Insights**

The analysis revealed clear and actionable risk clusters, confirming the power of credit utilization and delinquency recency.

  * **Financial Strain is Clustered:** The highest concentration of **Bad Loans** occurs in the segment defined by **high Revolving Utilization ($\ge 60\%$**) coupled with **high Interest Rates**.
  * **Strongest Predictor (DTI/Utilization):** Both the median **Debt-to-Income (DTI)** ratio and **Revolving Utilization** are significantly higher for defaulted loans compared to healthy ones, confirming high debt usage as the **primary financial strain indicator**.
  * **Recency of Issues:** Borrowers with **recent major derogatory events** (low value of `mths_since_last_major_derog`) and those with **public records** are substantially more likely to default.
  * **Geographic Risk:** States like **Nevada (NV), Florida (FL), and Georgia (GA)** exhibit a disproportionately high default rate compared to the high-volume states (CA, TX).

-----

## **4. Modeling, Tuning, and Performance**

The best predictive model was identified through hyperparameter tuning across five different algorithms.

  * **Algorithms:** **XGBoost, CatBoost, AdaBoost**, and baselines **Logistic Regression** and **MLP Classifier**.
  * **Tuning Strategy:** **RandomizedSearchCV** was used with **Stratified K-Fold** cross-validation on the training data ($\mathbf{X}_{train}$) to find the optimal settings for each model.
  * **Evaluation:** The best model was tested on the **unseen $\mathbf{X}_{test}$ set** to provide an unbiased measure of real-world performance.

### **Model Performance Comparison**

*(Note: Actual F1 scores are dependent on the final trained models.)*

| Model | F1 Score (Weighted) | Key Balancing Parameter |
| :--- | :--- | :--- |
| **XGBoost** | [Highest F1/Accuracy] | `scale_pos_weight = 7.427` |
| **CatBoost** | [High F1/Accuracy] | `auto_class_weights = 'Balanced'` |
| **Logistic Regression** | [Baseline F1] | `class_weight = 'balanced'` |

-----

## **5. Final Recommendations and Business Solutions**

The final solution provides clear actionable steps to update the lending company’s underwriting rules.

1.  **Implement the Predictive Model:** Deploy the best-performing model (likely XGBoost or CatBoost) directly into the approval workflow to assign a **precise risk probability** to every new loan application.
2.  **Set Hard Stop on High-Leverage Borrowers:** Establish strict underwriting rules (auto-reject or senior review) for applicants with **DTI** or **Revolving Utilization** exceeding the median thresholds observed in the defaulted loan segment.
3.  **Adjust for Geographic & Temporal Risk:** Instantly flag applications from high-default states (e.g., NV, FL, GA) and those showing **recent severe credit issues** (`mths_since_last_major_derog`) for enhanced scrutiny.
4.  **Develop a Feature-Driven Scorecard:** Create a transparent, objective scorecard using the model's **Feature Importance** to rationalize all acceptance and rejection decisions, improving compliance and client communication.

-----

## **6. Setup and Dependencies**

To replicate this project and run the models, you must have the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost lightgbm catboost
```