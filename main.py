#main.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("D:\Sanjana\defaulter\DefaulterList - Sheet1 (1).csv")

# Encode the target variable
le = LabelEncoder()
df['Defaulter'] = le.fit_transform(df['Defaulter'])

# Selecting only the specified columns for features
selected_columns = ['IoT', 'SE', 'HS-OB', 'OE1- SC', 'OEI-ICCF', 'IOTL']
X = df[selected_columns]

# Target variable
y = df['Defaulter']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
outlier_indices = []
for column in selected_columns:
    outlier_index = df[df[column] > 30].index
    outlier_indices.extend(outlier_index)

# Remove outliers from the dataset
df.drop(outlier_indices, inplace=True)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('interface.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extracting values from the form
    iot = int(request.form['iot'])
    se = int(request.form['se'])
    hs_ob = int(request.form['hs-ob'])
    oe1_sc = int(request.form['oe1-sc'])
    oei_iccf = int(request.form['oei-iccf'])
    iotl = int(request.form['iotl'])

    # Making prediction
    prediction = rf_classifier.predict([[iot, se, hs_ob, oe1_sc, oei_iccf, iotl]])

    # Convert prediction back to its original label
    prediction = le.inverse_transform(prediction)[0]

    return render_template('interface.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

