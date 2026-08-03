# Predicting Liver Cirrhosis Patient Outcomes to Support Earlier Clinical Intervention
## A machine learning and deep learning analysis of 418 primary biliary cirrhosis patients
**Author**: Siwar Ehwass

### Business Problem

Liver cirrhosis is a disease where the liver becomes permanently scarred over time. As the damage gets worse, the liver cannot work properly, making it harder to remove toxins, produce proteins, and help blood clot.

This dataset follows patients with **primary biliary cirrhosis (PBC)**. Some patients remain stable, some need a liver transplant, and some die from the disease.

The goal of this project is to predict whether a patient will be:

- **C** – Alive / Stable
- **CL** – Received a liver transplant
- **D** – Died

Since missing a patient who is likely to die is the most serious mistake, the models were optimized to detect **D (death)** cases instead of simply maximizing overall accuracy.

![Liver Cirrhosis Progress](visuals/Cirrhosis-image-2.jpg)

---

### Data:
fedesoriano. (August 2021). Cirrhosis Prediction Dataset. Retrieved Jul. 12, 2026. [Dataset](https://www.kaggle.com/fedesoriano/cirrhosis-prediction-dataset).
- **418 patient records** (312 from a randomized clinical trial + 106 additional patients followed for survival), each with **19 clinical/lab features** plus the target, `Status`.
**Target Class:**
- **C (Censored / Alive)** — the patient was still alive and stable when the study ended. To a doctor, this is the "safe" group — no urgent action needed.
- **CL (Censored, Liver Transplant)** — the patient's disease got bad enough that they needed a liver transplant. This is serious, but the patient survived and got treated in time.
- **D (Death)** — the patient died from the disease. This is the outcome doctors want to prevent the most.


---

## Methods

- Removed `ID` and `N_Days` to avoid data leakage.
- Filled missing categorical values with `"MISSING"` and numerical values with the median.
- Ordinal encoded `Edema` and `Stage`; one-hot encoded the remaining categorical features.
- Compared **Logistic Regression**, **Random Forest**, and **Neural Network** models.
- Used **SMOTE** for class imbalance, **KMeans** for feature engineering, and **backward feature selection** to keep the most useful features.

---

## Results

### Bilirubin vs. Patient Outcome

![Bilirubin by Status](visuals/Bilirubin-vs-status.png)

> Patients with higher `Bilirubin` levels were much more likely to die. Death rates increased from **13.3%** in the lowest group to **72.4%** in the highest group. This matches what doctors already know, since high bilirubin is a sign of poor liver function.

### Prothrombin vs. Patient Outcome

![Patient status across Prothrombin levels](visuals/Prothrombin-vs-Status.png)

> Higher `Prothrombin` time was also linked to higher death rates. As clotting time increased, death rates rose from **20.2%** to **69.2%**, showing that worse liver function leads to poorer outcomes.

### Platelets vs. Patient Outcome

![Platelets by Status](visuals/Platelets-vs.-status.png)

> Lower `Platelets` were linked to higher death rates. Patients with higher platelet counts generally had better outcomes, although the very-high group showed a small increase because it contained only a few patients.

---

## Feature Importance

![Feature Importance](visuals/Feature-Importance.png)

> `Bilirubin` and `Platelets` were the most important features in the final model. `Age` and `Prothrombin` were also strong predictors. These features are well-known clinical signs of liver disease, which shows the model is learning meaningful patterns.

---

## Model

Three models were compared: **Logistic Regression**, **Random Forest**, and a **Neural Network**.

The final model is a **Neural Network** because it detects more patients who eventually die, which is the main goal of this project.

![Final Model Performance Comparison](visuals/RF-vs.-NN.png)

> The Random Forest had slightly better overall performance (Macro F1: **0.55** vs. **0.52**), but the Neural Network detected more death cases (**73% recall** vs. **68%**). Since finding high-risk patients is more important than maximizing overall accuracy, the Neural Network was chosen.

Both models had difficulty predicting the rare **CL (transplant)** class because there were very few training examples.

---

## Recommendations

- Use the model as a **decision support tool**, not a replacement for doctors.
- Pay close attention to patients with high **Bilirubin**, long **Prothrombin** times, low **Platelets**, and older **Age**.
- Continue using normal clinical procedures to identify transplant patients because the model does not predict the **CL** class well.
- Use the patient clusters found during analysis to help decide how closely patients should be monitored.

---

## Limitations & Next Steps

- The dataset is small (**418 patients**) and comes from one older study, so the model should be tested on newer data.
- The **CL** class is very small, making it difficult for any model to learn.
- Future work could test the model on larger datasets and use tools such as **SHAP** to explain individual predictions.

---

### For further information
For the full analysis, code, and additional visualizations, see the accompanying [notebook](Cirrhosis_Prediction.ipynb) in this repository. For any additional questions, please contact **siwarehwass@gmail.com**.
