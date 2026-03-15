# CDC Diabetes Health Indicators - Predictive Analysis

This project aims to predict the presence of diabetes or pre-diabetes in patients using healthcare statistics and lifestyle survey information. By leveraging machine learning models, we analyze key indicators such as BMI, smoking habits, and blood pressure to assist in early misconception detection and health risk assessment.

##  Dataset
The dataset is sourced from the **UCI Machine Learning Repository (CDC Diabetes Health Indicators)**.
- **Size:** 253,680 instances
- **Features:** 21 (Categorical and Integer)
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes)

##  Methodology
This analysis follows a structured machine learning pipeline:
1. **Data Acquisition:** Fetching the dataset directly via the UCI ML Repository API.
2. **Preprocessing:** Handling imbalanced classes and splitting data into training (20%) and testing (80%) sets.
3. **Modeling:**
   - **Decision Trees:** Baseline and depth-optimized models.
   - **Random Forest:** Baseline classification.
4. **Optimization:** - Hyperparameter tuning using **GridSearchCV**.
   - Advanced Bayesian optimization using **Optuna (TPE Search)** to maximize **Recall**, ensuring the model captures as many diabetic cases as possible.

##  Key Results
- **Decision Tree (Optimized):** 86.4% Accuracy at a depth of 6.
- **Random Forest:** 86.1% Baseline Accuracy.
- **Best Performance:** Achieved **100% Recall** on testing data using Optuna-tuned hyperparameters, addressing the critical need for sensitivity in health diagnostics.

##  Tech Stack
- **Python**
- **Scikit-learn** (Modeling & Evaluation)
- **Optuna** (Bayesian Optimization)
- **Matplotlib/Seaborn** (Visualization)
- **Pandas/NumPy** (Data Manipulation)

##  Repository Structure
- `Random_forest.ipynb`: Main Jupyter Notebook containing data analysis, model training, and optimization.
