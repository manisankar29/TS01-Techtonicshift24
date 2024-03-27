import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
@st.cache_data
def load_data():
    train_df = pd.read_csv('D:\Suspicious_activity_detection\Train.csv')
    test_df = pd.read_csv('D:\Suspicious_activity_detection\Test.csv')
    return train_df, test_df

train_df, test_df = load_data()

# Sidebar options
st.sidebar.title("Options")
show_train_data = st.sidebar.checkbox("Show Train Data", value=False)
show_test_data = st.sidebar.checkbox("Show Test Data", value=False)

if show_train_data:
    st.subheader("Train Data")
    st.write(train_df)

if show_test_data:
    st.subheader("Test Data")
    st.write(test_df)

# Data preprocessing
st.subheader("Data Preprocessing")

X_train = train_df.drop('Activity', axis=1)
Y_train = train_df['Activity']
X_test = test_df.drop('Activity', axis=1)
Y_test = test_df['Activity']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
def build_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    return model

def train_lstm_model(X_train, Y_train, X_val, Y_val, batch_size=32, epochs=50):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), len(np.unique(Y_train)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=1)
    return model, history

def evaluate_model(model, X_test, Y_test):
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    testing_accuracy = accuracy_score(Y_test, Y_pred)
    print("Testing Accuracy:", testing_accuracy)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    #print("Confusion Matrix:")
    #print(conf_matrix)
    
    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Display confusion matrix as heatmap with percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()
    
    return testing_accuracy, conf_matrix

# Train LSTM model
st.subheader("Training LSTM Model")
with st.spinner('Training the LSTM model...'):
    lstm_model, lstm_history = train_lstm_model(X_train_reshaped, Y_train_encoded, 
                                                X_test_reshaped, Y_test_encoded)
    st.success("Training completed successfully!")

# Plot training and validation curves
st.subheader("Training and Validation Curves")
st.pyplot(plt)

# Evaluate LSTM model
st.subheader("Evaluation")
testing_accuracy, conf_matrix = evaluate_model(lstm_model, X_test_reshaped, Y_test_encoded)
st.write("Testing Accuracy:", testing_accuracy)

# Display confusion matrix
st.subheader("Confusion Matrix")
st.write("Confusion Matrix:")
st.write(conf_matrix)
