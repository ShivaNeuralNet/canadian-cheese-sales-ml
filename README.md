# 🧀 Canadian Cheese Fat Level Classification (Final ML Project)

This machine learning project applies classification models to predict the **fat level** (Lower Fat or Higher Fat) of Canadian cheese products based on a range of categorical and numeric features. The dataset is publicly available via Canada's Open Government Portal and includes variables like cheese category, milk treatment, moisture percentage, and manufacturing region.

The project was built to solidify core machine learning concepts, including EDA, preprocessing, model evaluation, and hyperparameter tuning.

---

## 🎯 Objective

To classify cheese products into **Lower Fat** or **Higher Fat** categories using machine learning algorithms and evaluate their performance with appropriate metrics—especially in the presence of **class imbalance**.

---

##  Libraries & Tools

- `pandas`, `numpy`, `scipy`
- `scikit-learn`: classifiers, pipelines, transformers, metrics
- `altair`
- `RandomizedSearchCV`, `GridSearchCV` for tuning

---

##  Workflow Summary

### 1. Data Splitting & Setup
- The dataset was split into `train_df` (80%) and `test_df` (20%) using `random_state=123`
- Target variable: `FatLevel`
- Imbalance noted: more "Lower Fat" samples in training data

###  2. Exploratory Data Analysis (EDA)
- Used `Altair` for visualizing categorical and numerical feature distributions
- Observed dominance of Quebec in cheese production
- Lower Fat cheese and pasteurized cow milk were most common
- Organic cheese had very few samples

###  3. Preprocessing
- Built pipelines using `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, and `StandardScaler`
- Applied column transformers for categorical, ordinal, and numerical features
- Converted target labels to binary (1 = Lower Fat, 2 = Higher Fat)

###  4. Baseline Model
- Created a `DummyClassifier` (strategy = "most_frequent") for performance comparison

###  5. Modeling & Evaluation
Trained and compared the following models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- RBF SVM

Used:
- 5-fold Cross-Validation (`cross_validate`)
- `accuracy`, `precision`, `recall`, `f1_score`
- `plot_confusion_matrix` and `classification_report`

**Class imbalance** was addressed using `class_weight='balanced'`.

---

##  Hyperparameter Tuning

Performed hyperparameter optimization using `RandomizedSearchCV` for:
- Decision Tree (`max_depth`)
- KNN (`n_neighbors`)
- Logistic Regression (`C`, `penalty`)

Scoring metric: **F1 Score (weighted average)**

---

## 📈 Model Performance Summary

| Model                        | Accuracy | Weighted Recall | Weighted Precision |
|-----------------------------|----------|------------------|--------------------|
| Decision Tree               | 0.8296   | 0.8395           | 0.8949             |
| Decision Tree (Tuned)       | 0.8283   | 0.8283           | 0.8247             |
| KNN                         | 0.8247   | 0.8778           | 0.8592             |
| KNN (Tuned)                 | 0.8038   | 0.8038           | 0.8045             |
| Logistic Regression         | 0.7647   | 0.7865           | 0.8450             |
| Logistic Regression (Tuned) | 0.7844   | 0.7844           | 0.7854             |
| RBF SVM                     | 0.7803   | 0.7829           | 0.8704             |
| Random Forest               | 0.8211   | 0.9216           | 0.8592             |
| Random Forest (Tuned)       | 0.7656   | 0.7656           | 0.7680             |
| Dummy Classifier            | —        | —                | —                  |

📌 **Best tuned model:**  
`DecisionTreeClassifier(class_weight='balanced', max_depth=42)`  
→ `Test Accuracy: 0.8654`

---

##  Insights

- Decision Tree provided the best overall performance on the test set but showed signs of **overfitting**
- Random Forest had more balanced recall/precision, but tuning reduced its performance
- Logistic Regression underperformed despite being well-suited for binary classification
- The dataset’s **small size** and strong **class imbalance** had a noticeable impact on all models

---

##  Next Steps

- Investigate overfitting in Decision Tree via feature pruning
- Apply feature selection to test impact on model accuracy
- Try ensemble tuning with `VotingClassifier`
- Test additional metrics such as **MAPE** or ROC-AUC
- Build a dashboard or Streamlit app for interactive model use

---

## 🌐 Resources
- [Canadian Open Cheese Dataset](https://open.canada.ca/data/en/dataset)
- [Cheese Types Overview – Wikipedia](https://en.wikipedia.org/wiki/Canadian_cheese)

---

## 👩‍💻 Author

**Shiva Dorri**  
[LinkedIn Profile](https://linkedin.com/in/shiva-dorri)

---

