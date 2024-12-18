import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

class AVFPredictor:
    def __init__(self, data_path):
        # Load the data
        self.data = pd.read_csv(data_path)

        # Separate features and targets
        self.features = self.data.drop(['AVF Group', 'Post Op Complications', 'Post Op Thrill', 'Post Op Bruit'], axis=1)

        # Identify columns that will be used for preprocessing
        self.categorical_columns = list(self.features.select_dtypes(include=['object']).columns) + ['AVF Group']
        self.numerical_columns = self.features.select_dtypes(include=['float64']).columns

        # Preprocessing for subsequent models
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_columns)
            ])

        # Label encoders for target variables
        self.avf_group_encoder = LabelEncoder()
        self.complications_encoder = LabelEncoder()
        self.thrill_encoder = LabelEncoder()
        self.bruit_encoder = LabelEncoder()

    def train_avf_group_model(self):
        # Encode AVF Group
        y_avf = self.avf_group_encoder.fit_transform(self.data['AVF Group'])

        # Preprocessing for AVF Group model (excluding AVF Group as a feature)
        avf_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'),
                 self.features.select_dtypes(include=['object']).columns)
            ])

        # Create and train AVF Group model
        self.avf_group_model = Pipeline([
            ('preprocessor', avf_preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.avf_group_model.fit(self.features, y_avf)

        # Save the model and encoders
        joblib.dump(self.avf_group_model, 'avf_group_model.joblib')
        joblib.dump(self.avf_group_encoder, 'avf_group_encoder.joblib')

    def train_subsequent_models(self):
        # Prepare data with features and AVF Group
        X_with_avf = self.data[self.features.columns.tolist() + ['AVF Group']]

        # Train models for each target
        targets = [
            ('Complications', self.complications_encoder, self.data['Post Op Complications']),
            ('Thrill', self.thrill_encoder, self.data['Post Op Thrill']),
            ('Bruit', self.bruit_encoder, self.data['Post Op Bruit'])
        ]

        self.subsequent_models = {}

        for name, encoder, target in targets:
            # Encode target
            y_encoded = encoder.fit_transform(target)

            model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])

            model.fit(X_with_avf, y_encoded)

            # Save model and encoder
            joblib.dump(model, f'{name.lower()}_model.joblib')
            joblib.dump(encoder, f'{name.lower()}_encoder.joblib')

            self.subsequent_models[name] = model

    def train(self):
        self.train_avf_group_model()
        self.train_subsequent_models()

# Usage
predictor = AVFPredictor('AVF_data.csv')
predictor.train()