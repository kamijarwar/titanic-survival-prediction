# titanic-survival-prediction
End-to-end ML project on the Titanic dataset: data cleaning, feature engineering, and baseline comparison using Logistic Regression.
# Titanic Feature Engineering Analysis

An end-to-end machine learning project on the Titanic dataset: data cleaning, feature engineering, and a baseline comparison to test whether the engineered features actually help.

## Objective

Predict whether a passenger survived the Titanic (891 records) using demographic and travel data, and measure the real impact of feature engineering against a baseline model.

## Key Findings

| Finding | Result |
|---|---|
| Survival rate, travelling with family | **50.6%** |
| Survival rate, travelling alone | **30.4%** |
| Accuracy without engineered features (baseline) | **81.0%** |
| Accuracy with engineered features | **80.4%** |

**Takeaway:** The family signal in the data is real, but my engineered features (`Family_Size`, `ISalone`) did not improve accuracy. They repackaged `SibSp` and `Parch`, which the model already had. A new feature only helps when it adds information the model does not already have.

## Workflow

1. **Data understanding:** `info()`, `isnull().sum()`, `head()`
2. **Missing values:**
   - `Age` (177 missing): filled with median
   - `Embarked` (2 missing): filled with mode
   - `Cabin` (687 missing): dropped
3. **Feature construction:**
   - `Family_Size = SibSp + Parch + 1`
   - `ISalone = 1 if Family_Size == 1 else 0`
4. **Feature selection:** dropped `PassengerId`, `Name`, `Ticket`
5. **Encoding:** one-hot encoding for `Sex` and `Embarked`
6. **Train-test split:** 80/20, `random_state=42`
7. **Scaling:** `StandardScaler` on `Age`, `Fare`, `Family_Size`, fitted on training data only to prevent data leakage
8. **Modeling:** Logistic Regression, evaluated with accuracy score on unseen test data
9. **Baseline comparison:** same model retrained without the engineered features

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- Jupyter Notebook

## Project Structure

```
titanic-feature-engineering-analysis/
|-- titanic_feature_engineering.ipynb
|-- train.csv
|-- README.md
```

## How to Run

``bash
git clone https://github.com/<your-username>/titanic-feature-engineering-analysis.git
cd titanic-feature-engineering-analysis
pip install pandas numpy scikit-learn jupyter
jupyter notebook titanic_feature_engineering.ipynb

## Next Steps

- Extract passenger titles (Mr., Mrs., Miss, Master) from the `Name` column and test whether this hidden signal improves accuracy
- Try other models (Random Forest, Decision Tree) and compare
- Use cross-validation for a more reliable accuracy estimate than a single train-test split

## Author

**Kamran Jarwar**
IT Student, Quaid-e-Awam University of Engineering, Science and Technology
