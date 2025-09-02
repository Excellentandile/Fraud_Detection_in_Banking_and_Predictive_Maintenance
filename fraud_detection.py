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


