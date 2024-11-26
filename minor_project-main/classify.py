def model():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Load the dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('animalData.csv')

    # Preprocess the data
    def preprocess_data(df):
        df = df.dropna().reset_index(drop=True)
        label_encoder = LabelEncoder()
        df['Dangerous'] = label_encoder.fit_transform(df['Dangerous'])

        X = df.drop(['AnimalName', 'Dangerous'], axis=1)
        y = df['Dangerous']

        X_encoded = pd.get_dummies(X)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42), label_encoder

    # Train the Neural Network model
    @st.cache_resource
    def train_neural_model(X_train, y_train):
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)
        return model

    # Train KNN model
    def train_knn(X_train, y_train):
        knn_model = KNeighborsClassifier(n_neighbors=21)
        knn_model.fit(X_train, y_train)
        return knn_model

    # Train SVM model
    def train_svm(X_train, y_train):
        svm_model = SVC(kernel='linear', probability=True, random_state=0)
        svm_model.fit(X_train, y_train)
        return svm_model

    # Train Random Forest with GridSearchCV
    def train_random_forest(X_train, y_train):
        forest_clf = RandomForestClassifier()
        params = {'n_estimators': [4, 6, 8, 10], 'max_depth': [2, 4, 6], 'min_samples_split': [2, 3, 4]}
        grid_forest_clf = GridSearchCV(forest_clf, params, cv=6, n_jobs=-1)
        grid_forest_clf.fit(X_train, y_train)
        return grid_forest_clf.best_estimator_

    # Streamlit Layout for Binary Classification
    options = st.radio("Choose an option:", ['Home', 'Data Overview', 'Model Training', 'Predictions'])

    st.title("Animal Health Prediction System")

    # Load the dataset
    data = load_data()

    if options == 'Home':
        st.image("animal.jpg", use_column_width=True)
        st.markdown("""
            *Welcome!* This app allows you to explore animal health data and predict dangerous conditions.
            """)

    elif options == 'Data Overview':
        st.subheader("Dataset Overview")
        st.write(data.head(10))
        st.markdown("*Summary*")
        st.write(data.describe())

        st.markdown("### Data Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(x='Dangerous', data=data, ax=ax[0])
        ax[0].set_title("Dangerous Distribution")
        sns.countplot(x='AnimalName', data=data, order=data['AnimalName'].value_counts().index[:5], ax=ax[1])
        ax[1].set_title("Top 5 Animals")
        st.pyplot(fig)

    elif options == 'Model Training':
        st.subheader("Train and Evaluate Models")
        (X_train, X_test, y_train, y_test), label_encoder = preprocess_data(data)

        st.markdown("### Neural Network Training")
        nn_model = train_neural_model(X_train, y_train)
        accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]
        st.write(f"*Neural Network Accuracy:* {accuracy:.2f}")

        st.markdown("### KNN Model")
        knn_model = train_knn(X_train, y_train)
        knn_accuracy = knn_model.score(X_test, y_test)
        st.write(f"KNN Accuracy: {knn_accuracy:.2f}")

        st.markdown("### SVM Model")
        svm_model = train_svm(X_train, y_train)
        svm_accuracy = svm_model.score(X_test, y_test)
        st.write(f"SVM Accuracy: {svm_accuracy:.2f}")

        st.markdown("### Random Forest Model")
        rf_model = train_random_forest(X_train, y_train)
        rf_accuracy = rf_model.score(X_test, y_test)
        st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

    elif options == 'Predictions':
        st.subheader("Animal Danger Check")

        st.header("Input Animal Details")
        animal_name = st.text_input("Enter the animal name:")
        symptoms = [st.text_input(f"Symptom {i + 1}:") for i in range(5)]

        def find_dangerous_status(animal_name, symptoms):
            symptoms_sorted = sorted(symptoms)
            for _, row in data.iterrows():
                row_symptoms = sorted(
                    [row['symptoms1'], row['symptoms2'], row['symptoms3'], row['symptoms4'], row['symptoms5']]
                )
                if row['AnimalName'].lower() == animal_name.lower() and row_symptoms == symptoms_sorted:
                    return row['Dangerous']
            return "YES"

        if st.button("Check Danger Status"):
            if all(symptoms):
                result = find_dangerous_status(animal_name, symptoms)
                st.subheader(f"Result: {result}")
            else:
                st.warning("Please enter all 5 symptoms.")
