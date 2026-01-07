import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# NOTE: Removed joblib persistence as it can lead to data leakage or file corruption
# in certain deployment environments, causing metrics to incorrectly show 1.00.
# We will use st.cache_resource for reliable single training/split per session.
# MODEL_PATH = "spam_model.joblib"

st.set_page_config(page_title="Email / SMS Spam Classifier", layout="centered")
st.title("📧 Spam Classifier (using a real dataset)")

st.write(
    """
    Enter the message (email or SMS) below and click **Predict** to check if it's spam or not.
    The model uses a Naive Bayes classifier with TF-IDF features.
    """
)

# -------------------------
# LOAD & PREPARE DATA
# -------------------------

@st.cache_data(show_spinner="Loading and preparing data...")
def load_data(csv_path="C:/Users/acer/Downloads/archive/spam_ham_dataset.csv"):
    """
    Load the spam dataset from a CSV and perform basic cleaning.
    """
    # The SMS spam dataset often uses 'latin-1' encoding
    # Removed the first column selection [df.columns[0], df.columns[1]] for robustness
    # The dataset typically has 'Unnamed: 0', 'label', 'text' columns
    df = pd.read_csv(csv_path, encoding="latin-1")
    
    # Identify the correct label and text columns
    # Assuming 'label' is the second or third column, and 'text' is the other main column
    # Let's try to map the known columns for the spam_ham_dataset
    if 'label' in df.columns and 'text' in df.columns:
        df = df[['label', 'text']]
    elif df.shape[1] >= 2:
        # Fallback for common spam dataset structures
        df = df[[df.columns[1], df.columns[2]]] if df.columns[2] in ['text', 'message'] else df[[df.columns[1], df.columns[1]]] # Attempt to get the right columns
        df.columns = ["label", "text"]
    else:
        # Handle case where expected columns are missing (not strictly necessary for this fix)
        st.error("Dataset columns not in expected format.")
        return pd.DataFrame({'label':[], 'text':[]})

    # Drop rows with missing values
    df = df.dropna()
    
    # Convert text to lowercase for consistency
    df["text"] = df["text"].str.lower()
    
    return df

# Load dataset once
df = load_data()

# -------------------------
# TRAIN / LOAD MODEL (Using st.cache_resource for reliable, single run)
# -------------------------

@st.cache_resource(show_spinner="Training model and splitting data...")
def train_and_split(df):
    """
    Splits the data, trains the model pipeline, and returns the trained
    model and the test set for evaluation.
    This function runs only once per app session/deployment.
    """
    # Check for classes with only 1 member and remove them for safe stratification
    # First, find the count of each label
    label_counts = df["label"].value_counts()
    
    # Identify labels with less than 2 members
    single_member_labels = label_counts[label_counts < 2].index.tolist()
    
    if single_member_labels:
        # If any are found, remove those rows from the DataFrame
        df_filtered = df[~df["label"].isin(single_member_labels)].copy()
        st.warning(f"Removed {len(df) - len(df_filtered)} samples belonging to single-member classes: {single_member_labels}. This is necessary for stratified splitting.")
        df = df_filtered
        
    # Check if we still have data after filtering
    if df.empty:
        raise ValueError("The filtered dataset is empty or does not contain enough unique labels for splitting.")

    # Determine if stratification is possible/necessary
    # If the standard spam/ham dataset is used, stratification is good.
    # If there are only two classes and both have >1 member, we can stratify.
    if len(df["label"].unique()) >= 2 and all(df["label"].value_counts() >= 2):
        st.info("Performing **stratified** train-test split.")
        stratify_param = df["label"]
    else:
        st.info("Performing **non-stratified** train-test split (due to limited class samples).")
        stratify_param = None # Do not stratify
        
    # Split data (80% train, 20% test)
    # The fix is to ensure the 'stratify' condition is met OR not used.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Pipeline: TF-IDF Vectorizer followed by Multinomial Naive Bayes
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True)),
            ("nb", MultinomialNB()),
        ]
    )
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

# Get the trained model and the consistent test set
model, X_test, y_test = train_and_split(df)

# -------------------------
# UI INPUT
# -------------------------

message = st.text_area("✉️ Enter message here", height=200, placeholder="Example: Win a FREE iPhone! Click here now.")

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict", type="primary"):
    if not message.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Perform prediction and probability calculation
        pred = model.predict([message])[0]
        prob = model.predict_proba([message])[0]
        
        # Get the probability for the 'spam' class
        # Ensure 'spam' is in the list of known classes before trying to index
        model_classes = list(model.classes_)
        if "spam" in model_classes:
            spam_prob = prob[model_classes.index("spam")]
        else:
            # Fallback if 'spam' class is not in the trained model (highly unlikely for this dataset)
            spam_prob = 0.0 

        st.subheader("Prediction Result")
        
        if pred == "spam":
            st.error(f"🚨 This is **SPAM!** (Confidence: {spam_prob*100:.2f}%)")
        else:
            st.success(f"✔ This is **NOT spam** (Spam likelihood: {spam_prob*100:.2f}%)")

        st.divider()

        # -------------------------
        # Evaluate on test data
        # -------------------------
        
        st.subheader("📊 Model Performance on Unseen Test Set")
        
        # Check the size of the test set to confirm it's not the full dataset
        st.info(f"Test set size: {len(X_test)} samples (20% of total data).")

        y_pred = model.predict(X_test)
        
        # Calculate all required metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Only calculate precision, recall, f1 if 'spam' is in the test set
        if "spam" in y_test.unique():
            # Precision, Recall, F1 are calculated specifically for the 'spam' class
            precision = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0
            st.warning("The test set contains only 'ham' samples. Precision, Recall, and F1 for 'spam' are set to 0.0.")


        # Displaying the exact values (to 4 decimal places)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision (Spam)", f"{precision:.4f}")
        col3.metric("Recall (Spam)", f"{recall:.4f}")
        col4.metric("F1-Score (Spam)", f"{f1:.4f}")

        # -------------------------
        # Visualize metrics
        # -------------------------
        st.subheader("📈 Metrics Visualization")
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [accuracy, precision, recall, f1]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metrics, values, color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Metrics")
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.02,
                f"{yval:.4f}", # Displaying the exact value on the bar
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        st.pyplot(fig)

        # -------------------------
        # Confusion Matrix
        # -------------------------
        st.subheader("🧩 Confusion Matrix (Percentage View)")
        # Ensure labels are consistent for matrix
        labels = ["ham", "spam"]
        
        # Filter for labels actually present in the test set to avoid errors in unique_labels check
        present_labels = [l for l in labels if l in y_test.unique()]
        
        if len(present_labels) < 2:
             st.warning("Cannot generate Confusion Matrix: Not enough unique classes in the test set.")
        else:
            cm = confusion_matrix(y_test, y_pred, labels=present_labels)
            
            # Normalize by true row totals to get percentages
            # Use a small epsilon to prevent division by zero in case a row sum is zero
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1 # Avoids division by zero if all predictions for a true class are wrong
            cm_percent = cm.astype("float") / row_sums * 100
            
            cm_df = pd.DataFrame(
                cm_percent,
                index=[f"Actual: {l}" for l in present_labels],
                columns=[f"Predicted: {l}" for l in present_labels],
            )

            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm_df, 
                annot=True, 
                fmt=".2f", # Displaying percentage to two decimal places
                cmap="Blues", 
                cbar=True, 
                ax=ax_cm,
                linewidths=.5,
                linecolor='lightgray',
                annot_kws={"fontsize": 12, "fontweight": "bold"}
            )
            ax_cm.set_title("Confusion Matrix (%)")
            st.pyplot(fig_cm)

# -------------------------
# (Optional) Show raw data
# -------------------------
with st.expander("📚 Dataset Example (First 10 Rows)"):
    st.dataframe(df.head(10))