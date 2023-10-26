# Model Audit of COMPAS Dataset

In this project, the `model_audit.ipynb` notebook contains a comprehensive review and improvement of an existing machine learning model originally created by another analyst. The overarching aim was to identify and rectify any existing issues, specifically focusing on areas like preprocessing, documentation, feature selection, model selection, and potential biases.

## Contents

- [Introduction](#introduction)
- [Dataset Details](#dataset-details)
- [Findings & Analysis](#findings--analysis)
- [Attempted Model Improvement](#attempted-model-improvement)
- [Conclusion](#conclusion)

### Introduction

The intention behind this audit was to provide constructive feedback, suggest potential enhancements to the model, and, importantly, to ensure the mitigation of any biases the original model might introduce.

### Dataset Details

The model leverages the **ProPublica COMPAS dataset**. This dataset encompasses all criminal defendants who were subject to COMPAS screening in Broward County, Florida, during 2013 and 2014. Along with this, various features related to the defendant’s demographic information (e.g., gender and race), criminal history (e.g., number of prior offenses), and administrative details about the case (like case number, arrest date, and risk of recidivism as predicted by the COMPAS tool) are available. It also indicates whether the defendant recidivated or not.

### Findings & Analysis

#### Exploratory Data Analysis 

Evidently not much was done in this regard.

```python
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile
```

ProfileReport can be instrumental in familiarizing oneself with the dataset, showcasing correlations within features, null values, duplicates, etc.

#### Feature Selection/Engineering

**Features that could/should have been dropped:**
- `age`: High correlation with `age_cat`, redundant.
- `juv_other_count`: Not a misdemeanor or felony, ambiguous.
- `decile_score.1`: Identical to `decile_score`.
  
**Features that could have been considered:**
- `r_charge_degree`: Potentially relevant when combined with `c_charge_degree`.

***Potential Data Leak:***
Upon closer examination, I noticed that the values of "is_recid" and
"two_year_recid" are almost identical. Initially, I assumed that "is_recid"
indicated whether an individual had previously committed a crime before the
current evaluation. The absence of this feature's removal from the original
analysis might have been due to this presumption. However, without explicit
documentation, this remains speculative. To be cautious, I suggest excluding
the "is_recid" feature. Its presence could potentially explain the model's
exceptionally high accuracy, hinting at a possible data leakage.

##### One Hot Encoding

After one hot encoding, redundant columns should be discarded to avert overfitting. For instance, the `Female` column is superfluous if `Male` is binary.

#### Model Evaluation

The individual opted for an SVC model with default parameters. SVC is a fine model, but can be prone to overfitting.
The data should have been scaled. A better solution might
have been to start with a simple logistic regression model with cross validation and
hyperparameter tuning, and compare those results with the SVC model (and maybe
others too) to determine which produces best. With certain models, Feature
Importance could be implemented to further refine the model.

**Model Summary Feedback:**
Model Evaluation Feedback: The provided model summary is significantly lacking in detail and rigor. Simply stating "This is a very good and fair model because it is very accurate and predicts very well" is not only vague, but it lacks the necessary metrics to substantiate such a claim. A comprehensive summary should include metrics such as f1-score, precision, and recall, preferably from a detailed classification report. Additionally, the summary fails to specify which model was employed, its training process, or any information about feature engineering. Critically, the omission of any validation procedure casts doubt on the model's robustness and its ability to generalize to new data. It would be beneficial to provide context on the problem domain and elucidate the rationale behind certain decisions made during the model's development. Furthermore, acknowledging the model's limitations and suggesting areas or methodologies for improvement would enhance the summary's credibility and usefulness.

#### Feature Importance

The dominant feature turned out to be `decile_score` (0.4591). The process
behind its determination is proprietary, rendering it non-transparent. It
raises the question: how can any other feature be deemed valuable when the most
crucial feature's origin remains enigmatic?

### Attempted Model Improvement

After curating and training various models including SVC, Logistic Regression, Random Forest Classifier, XGBoost, and a Keras RNN, with unique feature engineering, I managed to reach an average accuracy of 68% for each model. The models' performances, in terms of f1-score, recall, etc., were also comparable. This outcome may stem from an insufficient dataset, hinting at the benefits a more extensive dataset could offer.

The [fairlearn library](https://fairlearn.org/) was employed to diminish model bias, but the results were less than optimal.

#### Initial Model Results:

| Race             | Accuracy | Precision | Recall   | F1_score |
|------------------|----------|-----------|----------|----------|
| African-American | 0.628517 | 0.637609  | 0.754420 | 0.691114 |
| Asian            | 0.812500 | 0.692308  | 0.818182 | 0.750000 |
| Caucasian        | 0.626732 | 0.546462  | 0.625366 | 0.583258 |
| Hispanic         | 0.612245 | 0.496503  | 0.579592 | 0.534840 |
| Native American  | 0.666667 | 0.692308  | 0.818182 | 0.750000 |
| Other            | 0.628647 | 0.509434  | 0.566434 | 0.536424 |

#### Mitigated Model Results:

| Race             | Accuracy | Precision | Recall   | F1_score |
|------------------|----------|-----------|----------|----------|
| African-American | 0.457792 | 0.603896  | 0.045678 | 0.084932 |
| Asian            | 0.656250 | 0.000000  | 0.000000 | 0.000000 |
| Caucasian        | 0.583537 | 0.521739  | 0.035122 | 0.065814 |
| Hispanic         | 0.612245 | 0.454545  | 0.040816 | 0.074906 |
| Native American  | 0.444444 | 1.000000  | 0.090909 | 0.166667 |
| Other            | 0.628647 | 0.714286  | 0.034965 | 0.066667 |

### Conclusion

From the given results, it's evident that the initial model performed considerably better than the mitigated one in terms of the fundamental metrics: accuracy, precision, recall, and F1 score. The accuracy across various racial groups was more balanced in the initial model, with the highest accuracy observed for the Asian race group at 81.25%. However, post-mitigation, a significant drop in accuracy was noticed, especially for the African-American group.

It's worth noting that precision and recall took a major hit in the mitigated model. For example, the Asian group's recall plummeted to zero, suggesting the model failed to correctly identify any true positives for that particular group. Similarly, for many groups, recall values in the mitigated model became abysmally low, indicating a reduced ability of the model to detect positive cases accurately.

The F1 score, which combines precision and recall, also showcased a noticeable decline for all groups in the mitigated model, further supporting the assertion that while our intentions to reduce bias were well-placed, the implementation may have rendered the model less effective in its predictive capacities.

Given these observations, it underscores the inherent challenges in model fairness optimization. While the objective is to create an equitable model, there's an evident trade-off between fairness and performance, as seen in this case. It's essential to strike a balance, ensuring that efforts to mitigate bias don't overly compromise a model's predictive power.

Future endeavors in this domain would benefit from exploring different mitigation techniques, possibly involving domain expertise and more advanced methodologies, to ensure both fairness and model performance are upheld to the highest standards.

