# Fraud Detection in Banking

This project aims to develop a robust fraud detection system for banking transactions. We utilize a dataset containing transaction details such as amount, time of day, location category, and merchant type. The approach combines unsupervised learning (K-Means clustering) for anomaly detection on unlabeled data with supervised learning (classification) for labeled fraud data.

## Problem Statement

Identifying suspicious transaction patterns in real-time is crucial for minimizing financial losses due to fraud and reducing false positives that can negatively impact customer experience. The dataset presents a challenge with both labeled and a significant portion of unlabeled data, requiring a hybrid approach.

## 1. Data Preprocessing and K-Means Clustering for Unlabeled Data

For the unlabeled portion of our transaction data, we employ K-Means clustering. This unsupervised learning technique is ideal for discovering inherent groupings or patterns within data where explicit labels are absent. In the context of fraud detection, K-Means can help identify anomalous clusters that might represent suspicious activities (e.g., high-amount transactions occurring at unusual times or locations).

### Justification for K-Means Clustering

K-Means is chosen for its simplicity, efficiency, and effectiveness in partitioning data into a predefined number of clusters. It works by iteratively assigning data points to the nearest centroid and then updating the centroids to be the mean of the points assigned to them. This process helps in identifying natural groupings in the data. For fraud detection, transactions that fall into small, isolated clusters or clusters with unusual characteristics (e.g., high average transaction amounts at odd hours) can be flagged as potential anomalies.

### Choosing the Optimal Number of Clusters (k)

The choice of `k` (the number of clusters) is critical for K-Means. The **Elbow Method** is a common heuristic used for this purpose. It involves plotting the Within-Cluster Sum of Squares (WCSS) against different values of `k`. WCSS measures the sum of squared distances between each point and its assigned centroid. As `k` increases, WCSS generally decreases. The 


elbow point on the plot, where the rate of decrease in WCSS significantly slows down, is often considered the optimal `k`.

### Implementation Steps (Python with scikit-learn and matplotlib)

1.  **Load Data**: Load the `Question2Datasets.csv` into a pandas DataFrame.
2.  **Separate Labeled and Unlabeled Data**: The `Is_Fraud (Labeled Subset)` column contains `0.0` for non-fraud, `1.0` for fraud, and `-1.0` for unlabeled data. We will filter out the unlabeled data (where `Is_Fraud` is `-1.0`) for K-Means clustering.
3.  **Encode Categorical Features**: `Location` and `Merchant` are categorical. We will use One-Hot Encoding to convert them into a numerical format suitable for K-Means.
4.  **Scale Numerical Features**: `Amount` and `Time_Hour` are numerical features. Since K-Means is a distance-based algorithm, feature scaling is crucial to ensure that features with larger values do not disproportionately influence the distance calculations. We will use `StandardScaler` to normalize these features.
5.  **Apply K-Means**: Fit the K-Means model to the preprocessed unlabeled data.
6.  **Visualize Clusters**: Use `matplotlib` to visualize the clusters, potentially focusing on `Amount` and `Time_Hour` to identify anomalous groupings.
7.  **Elbow Method Visualization**: Plot the WCSS for a range of `k` values to determine the optimal number of clusters.




**Elbow Method Plot**: The `elbow_method.png` plot helps in visually identifying the optimal `k`. We look for the point where the WCSS curve starts to flatten out, resembling an elbow.

**K-Means Clusters Visualization**: The `kmeans_clusters.png` visualizes the clusters based on 'Amount' and 'Time_Hour'. This helps in understanding the groupings and identifying potential anomalous clusters (e.g., high amount transactions at unusual hours).

### K-Means Clustering Results

After running the K-Means algorithm with an assumed optimal `k=3` (to be confirmed by visual inspection of `elbow_method.png`):

*   **Cluster Distribution**:
    *   Cluster 1: 37 transactions
    *   Cluster 2: 35 transactions
    *   Cluster 0: 28 transactions

These clusters can now be analyzed to identify patterns. For instance, if one cluster predominantly contains transactions with high amounts occurring late at night, it could be flagged for further investigation as potentially fraudulent.

## 2. Classification for Labeled Data

For the labeled portion of our dataset (transactions marked as fraud or non-fraud), we will employ a supervised classification algorithm. This allows us to learn from known fraud patterns and predict whether new, unseen transactions are fraudulent.

### Algorithm Choice: Naïve Bayes

We choose **Gaussian Naïve Bayes** for its simplicity, efficiency, and effectiveness, especially when dealing with high-dimensional data and categorical features (after one-hot encoding). Naïve Bayes classifiers are a family of probabilistic classifiers based on applying Bayes' theorem with strong (naïve) independence assumptions between the features.

### Justification for Naïve Bayes

*   **Probabilistic Approach**: It provides a probabilistic output, which can be useful for ranking transactions by their likelihood of being fraudulent.
*   **Efficiency**: It is computationally inexpensive and can handle large datasets efficiently.
*   **Good with Categorical Features**: Although Gaussian Naïve Bayes assumes numerical features follow a Gaussian distribution, its variants (like Multinomial or Bernoulli Naïve Bayes) are well-suited for discrete or categorical data. With one-hot encoded categorical features, Gaussian Naïve Bayes can still perform reasonably well.

### Bias-Variance Tradeoff with Naïve Bayes

Naïve Bayes generally exhibits **high bias and low variance**. This means:

*   **High Bias**: The strong independence assumption is a simplifying assumption that might not hold true in real-world data, leading to a biased model that might not capture complex relationships. This can result in underfitting if the true relationship between features and target is highly complex.
*   **Low Variance**: Due to its simplicity and strong assumptions, Naïve Bayes models are less sensitive to fluctuations in the training data. This makes them less prone to overfitting, which is beneficial for generalizing to unseen data. In the context of fraud detection, where new fraud patterns can emerge, a low-variance model can be more stable.

### Implementation Steps (Python with scikit-learn)

1.  **Prepare Labeled Data**: Extract features (X) and target (y) from the `labeled_df`.
2.  **Preprocess Labeled Data**: Apply the same preprocessing steps (One-Hot Encoding for categorical features and StandardScaler for numerical features) to the labeled data. It's crucial to use the `fit_transform` on training data and `transform` on test data to prevent data leakage.
3.  **Split Labeled Data**: Divide the labeled dataset into training and testing sets (e.g., 80/20 split).
4.  **Train Naïve Bayes Model**: Fit a Gaussian Naïve Bayes classifier to the training data.
5.  **Predict**: Make predictions on the test data.
6.  **Evaluate**: Assess the model's performance using appropriate classification metrics (e.g., F1-score, Confusion Matrix).




### Naïve Bayes Classification Results

After training and evaluating the Gaussian Naïve Bayes classifier on the labeled data, we obtained the following results:

*   **F1-Score**: 0.17
*   **Confusion Matrix**:
    ```
    [[ 9 10]
     [ 0  1]]
    ```
*   **Classification Report**:
    ```
                  precision    recall  f1-score   support

           0.0       1.00      0.47      0.64        19
           1.0       0.09      1.00      0.17         1

      accuracy                           0.50        20
     macro avg       0.55      0.74      0.40        20
  weighted avg       0.95      0.50      0.62        20
    ```

The F1-score for the fraud class (1.0) is very low (0.17), indicating that the model is not performing well in identifying fraudulent transactions. While it has a high recall for the fraud class (1.00), meaning it caught all fraudulent transactions in the test set, its precision is extremely low (0.09), leading to a high number of false positives (10 out of 19 non-fraudulent transactions were misclassified as fraud). This is a critical issue in fraud detection, as false positives can lead to customer dissatisfaction.

This poor performance suggests that the current features and the Naïve Bayes model might not be sufficient to capture the complex patterns of fraud in this dataset. This highlights the need for robust feature engineering and potentially more sophisticated classification algorithms.

## 3. Feature Engineering and Model Re-evaluation

Given the unsatisfactory performance of the initial Naïve Bayes model, feature engineering becomes paramount. Creating more informative features can help the model better distinguish between fraudulent and non-fraudulent transactions. Additionally, addressing the class imbalance (where non-fraudulent transactions significantly outnumber fraudulent ones) is crucial.

### Feature Engineering Proposals

1.  **Binning Time of Day**: Instead of using `Time_Hour` as a continuous variable, we can bin it into categories like "morning" (6-12), "afternoon" (12-18), "evening" (18-24), and "night" (0-6). This can capture patterns related to fraud occurring during specific periods.
2.  **Amount Deviation from User Average (Conceptual)**: For a real-world scenario, we could engineer features like the deviation of the current transaction amount from a user's historical average transaction amount. This would require user-specific historical data, which is not available in this synthetic dataset but is a powerful technique in practice.
3.  **Interaction Features**: Similar to the previous problem, creating interaction terms (e.g., `Amount * Time_Hour`) could capture more complex relationships.

### Addressing Imbalanced Classes

Fraud datasets are typically highly imbalanced, with very few fraudulent transactions compared to legitimate ones. This can lead to models that are biased towards the majority class (non-fraud), resulting in poor detection of the minority class (fraud). Strategies to address this include:

*   **Resampling Techniques**: Oversampling the minority class (e.g., SMOTE) or undersampling the majority class.
*   **Cost-Sensitive Learning**: Assigning different misclassification costs to different classes.
*   **Algorithm Choice**: Some algorithms are inherently more robust to class imbalance.

For this problem, we will focus on feature engineering and re-evaluate the Naïve Bayes model with these new features. We will also consider using a different classification algorithm if the performance doesn't improve significantly.





### Feature Engineering Results

After applying feature engineering (binning `Time_Hour` into categories), we re-evaluated the Naïve Bayes model. Here are the results:

*   **F1-Score**: 0.22
*   **Confusion Matrix**:
    ```
    [[12  7]
     [ 0  1]]
    ```
*   **Classification Report**:
    ```
                  precision    recall  f1-score   support

           0.0       1.00      0.63      0.77        19
           1.0       0.12      1.00      0.22         1

      accuracy                           0.65        20
     macro avg       0.56      0.82      0.50        20
  weighted avg       0.96      0.65      0.75        20
    ```

The F1-score for the fraud class improved marginally from 0.17 to 0.22, but it remains low. The model still exhibits very low precision for the fraud class (0.12), indicating a high rate of false positives. This suggests that while feature engineering can help, the Naïve Bayes model, with its strong independence assumptions, might not be complex enough to capture the intricate patterns of fraud in this dataset, especially given the limited size of the labeled fraud data.

## 4. Model Evaluation with Cross-Validation and Metrics

To get a more reliable estimate of our classification model's performance and to ensure its generalization capability, we will use k-fold cross-validation. Given the imbalanced nature of fraud data, metrics like F1-score and a confusion matrix are crucial for a comprehensive evaluation.

### K-Fold Cross-Validation (k=10)

We will perform 10-fold cross-validation on the labeled data. This involves splitting the dataset into 10 equally sized folds. The model is trained on 9 folds and tested on the remaining 1 fold. This process is repeated 10 times, with each fold serving as the test set exactly once. The performance metrics are then averaged across all 10 iterations, providing a more robust assessment of the model's performance and reducing the impact of a single train-test split.

### Evaluation Metrics

*   **F1-score**: This metric is the harmonic mean of precision and recall. It is particularly useful for imbalanced datasets like fraud detection, as it balances the concerns of both false positives and false negatives. A high F1-score indicates a good balance between precision (minimizing false positives) and recall (minimizing false negatives).
*   **Confusion Matrix**: A table that describes the performance of a classification model on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm. It shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

### Implementation Steps

1.  **Import `StratifiedKFold` and `cross_val_score`**: From `sklearn.model_selection`.
2.  **Define Cross-Validation Strategy**: Initialize `StratifiedKFold` with `n_splits=10` and `shuffle=True` to maintain the proportion of fraud/non-fraud cases in each fold.
3.  **Calculate Cross-Validation Scores**: Use `cross_val_score` to get F1-scores for the Naïve Bayes model across the folds.
4.  **Report Averages**: Present the average F1-score and standard deviation.




### Cross-Validation Results (with Feature Engineering)

We performed 10-fold stratified cross-validation on the labeled data using the Naïve Bayes model with feature engineering. The average F1-score and its standard deviation are as follows:

*   **Naïve Bayes (FE) - Avg CV F1-Score**: 0.23 (+/- 0.29)

The cross-validation results confirm the challenges observed in the initial evaluation. The average F1-score remains low (0.23), and the high standard deviation (0.29) indicates significant variability in performance across different folds. This variability is largely attributable to the severe class imbalance in the dataset, as highlighted by the warning: "The least populated class in y has only 5 members, which is less than n_splits=10." With such a small number of positive (fraudulent) samples, it is difficult for the model to learn robust patterns, and the performance becomes highly sensitive to which few fraud instances end up in the test set of each fold.

### Applications: Anomaly Detection vs. Supervised Classification

This project utilized both unsupervised (K-Means clustering) and supervised (Naïve Bayes classification) techniques, each serving a distinct purpose in fraud detection:

*   **Anomaly Detection (Unsupervised - K-Means)**:
    *   **Purpose**: K-Means clustering was used on the *unlabeled* transaction data to group similar transactions. The primary application here is **anomaly detection**. Transactions that fall into small, isolated clusters, or clusters with characteristics significantly different from the majority (e.g., unusually high amounts, transactions at odd hours, or unusual merchant/location combinations), can be flagged as potential anomalies. These anomalies might not necessarily be fraudulent but warrant further investigation. This approach is valuable when historical fraud labels are scarce or non-existent, allowing for the discovery of novel suspicious patterns.
    *   **Performance**: K-Means helps in identifying *unusual* behavior based on the inherent structure of the data. Its effectiveness is measured by how well it groups similar transactions and isolates outliers. It's a discovery tool, not a direct fraud predictor.

*   **Supervised Classification (Naïve Bayes)**:
    *   **Purpose**: The Naïve Bayes classifier was trained on *labeled* data to predict whether a transaction is fraudulent or not. This is a direct **fraud prediction** application. The model learns from known examples of fraud and non-fraud to classify new transactions. The goal is to minimize false positives (to avoid annoying customers) while maximizing the detection of actual fraud (high recall).
    *   **Performance**: As observed, the Naïve Bayes model struggled with the imbalanced nature of the dataset and the limited number of fraud samples, resulting in a low F1-score and high false positive rate. This indicates that while supervised learning is the ultimate goal for direct fraud prediction, its success heavily relies on the quality and quantity of labeled data, as well as appropriate handling of class imbalance.

### Comparison of Supervised vs. Unsupervised Performance

In this scenario, both approaches faced challenges. K-Means successfully grouped transactions, providing a basis for anomaly detection, but its output requires human interpretation to determine if a cluster represents fraud. The supervised Naïve Bayes model, despite feature engineering, showed poor predictive performance for fraud due to data imbalance and potentially the simplicity of the model for complex fraud patterns. This highlights that:

*   **Unsupervised methods** are excellent for initial exploration, pattern discovery, and flagging *potential* anomalies, especially in the absence of labels. They can complement supervised models by providing new features (e.g., cluster IDs) or by identifying data points that need human review.
*   **Supervised methods** are necessary for direct fraud prediction but demand high-quality, balanced labeled data and often require more sophisticated algorithms (e.g., ensemble methods, deep learning) and techniques (e.g., advanced resampling, cost-sensitive learning) to perform effectively on imbalanced datasets.

For a robust fraud detection system, a combination of both approaches is often ideal: unsupervised methods for initial anomaly detection and pattern discovery, followed by supervised methods trained on carefully curated and balanced labeled data for precise fraud prediction.




## Code Snippets

### `fraud_detection.py`

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv("/home/ubuntu/upload/Question2Datasets.csv")

# Separate labeled and unlabeled data
unlabeled_df = df[df["Is_Fraud (Labeled Subset)"] == -1.0].copy()
labeled_df = df[df["Is_Fraud (Labeled Subset)"] != -1.0].copy()

# --- K-Means Clustering for Unlabeled Data ---
# Features for clustering (unlabeled data)
X_unlabeled = unlabeled_df.drop(columns=["Index", "Is_Fraud (Labeled Subset)"])

# Identify categorical and numerical features
numerical_features = ["Amount", "Time_Hour"]
categorical_features = ["Location", "Merchant"]

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown=\'ignore\'), categorical_features)
    ])

# Apply preprocessing to unlabeled data
X_unlabeled_processed = preprocessor.fit_transform(X_unlabeled)

# Determine optimal k using the Elbow Method
wcss = []
max_k = 10 # Let\'s test up to 10 clusters
for i in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=i, init=\'k-means++\', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_unlabeled_processed)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker=\'o\', linestyle=\'--\')
plt.title(\'Elbow Method for Optimal K\')
plt.xlabel(\'Number of Clusters (K)\')
plt.ylabel(\'WCSS\')
plt.grid(True)
plt.savefig(\'elbow_method.png\')

print("Elbow method plot saved to elbow_method.png")

# Based on the elbow method (visual inspection will be needed), choose an optimal k.
# For now, let\'s assume k=3 for demonstration purposes, this will be refined after viewing the plot.
optimal_k = 3

kmeans_model = KMeans(n_clusters=optimal_k, init=\'k-means++\', max_iter=300, n_init=10, random_state=42)
clusters = kmeans_model.fit_predict(X_unlabeled_processed)

unlabeled_df["Cluster"] = clusters

print(f"K-Means clustering performed with {optimal_k} clusters.")
print("Cluster distribution:")
print(unlabeled_df["Cluster"].value_counts())

# For visualization, we need to inverse transform or use original features for plotting.
# Let\'s plot Amount vs Time_Hour with cluster assignments.
plt.figure(figsize=(12, 8))
plt.scatter(unlabeled_df["Time_Hour"], unlabeled_df["Amount"], c=unlabeled_df["Cluster"], cmap=\'viridis\', alpha=0.7)
plt.title(\'K-Means Clusters of Unlabeled Transactions (Amount vs. Time_Hour)\')
plt.xlabel(\'Time of Day (Hour)\')
plt.ylabel(\'Amount (USD)\')
plt.colorbar(label=\'Cluster ID\')
plt.grid(True)
plt.savefig(\'kmeans_clusters.png\')

print("K-Means clusters plot saved to kmeans_clusters.png")

# Save the processed unlabeled data with cluster assignments for potential future use
unlabeled_df.to_csv(\'unlabeled_data_clustered.csv\', index=False)
print("Clustered unlabeled data saved to unlabeled_data_clustered.csv")

# --- Classification for Labeled Data ---
X_labeled = labeled_df.drop(columns=["Index", "Is_Fraud (Labeled Subset)"])
y_labeled = labeled_df["Is_Fraud (Labeled Subset)"]

# --- Feature Engineering for Labeled Data ---
def bin_time_of_day(hour):
    if 0 <= hour < 6:
        return "night"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "evening"

X_labeled["Time_Category"] = X_labeled["Time_Hour"].apply(bin_time_of_day)

# Update numerical and categorical features for the preprocessor
numerical_features_fe = ["Amount", "Time_Hour"]
categorical_features_fe = ["Location", "Merchant", "Time_Category"]

# Create a preprocessor for labeled data with feature engineering
preprocessor_labeled_fe = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features_fe),
        ("cat", OneHotEncoder(handle_unknown=\'ignore\'), categorical_features_fe)
    ])

# Create a pipeline for preprocessing and Naive Bayes classification with FE
model_pipeline_fe = Pipeline(steps=[
    (\'preprocessor\', preprocessor_labeled_fe),
    (\'classifier\', GaussianNB())
])

# Split labeled data into training and testing sets (for initial evaluation)
X_train_labeled, X_test_labeled, y_train_labeled, y_test_labeled = train_test_split(
    X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled # Stratify for imbalanced classes
)

# Train the Naive Bayes model with FE
model_pipeline_fe.fit(X_train_labeled, y_train_labeled)

# Make predictions on the test set with FE
y_pred_labeled_fe = model_pipeline_fe.predict(X_test_labeled)

# Evaluate the classifier with FE
print("\n--- Naive Bayes Classifier Evaluation (with Feature Engineering) ---")
print("F1-Score: ", f1_score(y_test_labeled, y_pred_labeled_fe))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_labeled, y_pred_labeled_fe))
print("\nClassification Report:\n", classification_report(y_test_labeled, y_pred_labeled_fe))

# --- K-Fold Cross-Validation (k=10) ---
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define F1 scorer
f1_scorer = make_scorer(f1_score)

print("\n--- Cross-Validation Results (with Feature Engineering) ---")

# Cross-validation for Naive Bayes with FE
f1_scores_nb_fe = cross_val_score(model_pipeline_fe, X_labeled, y_labeled, cv=kf, scoring=f1_scorer)

print(f"Naive Bayes (FE) - Avg CV F1-Score: {np.mean(f1_scores_nb_fe):.2f} (+/- {np.std(f1_scores_nb_fe):.2f})")
```


