import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Cleaning.clean_datasets import DataClener
from Cleaning.feature_engineering import FeatureEngineering
from Model.data_preprocess import DataPreprocess
from Model.model_develop import ModelDevelop

def main():

    data_train = pd.read_csv('Data/train_set.csv')
    data_test = pd.read_csv('Data/test_set.csv')

      # Limpieza de datos
    cleaner = DataClener()
    print("\n[Realizando limpieza de datasets]")
    cleaned_train_data = cleaner.cleaning_data(data_train)
    cleaned_test_data = cleaner.cleaning_data(data_test)
    cleaned_train_data = cleaned_train_data.dropna(subset=['passholder_type'])

    data_train_cleaned = cleaned_train_data.copy()
    data_test_cleaned = cleaned_test_data.copy()
    print("\n[Datasets limpios]")

    feature_eng = FeatureEngineering()
    print("\n[Con feature engineering]")
    data_train_cleaned_pre = feature_eng.feature_engineering(data_train_cleaned)
    data_test_cleaned_pre = feature_eng.feature_engineering(data_test_cleaned)

    pre_process = DataPreprocess()
    print("\n[Handling outliers]")
    data_train_cleaned = pre_process.handle_outliers(data_train_cleaned_pre)
    print("\n[Preprocesando features]")
    data_train_cleaned = pre_process.pre_process_df(data_train_cleaned, Train=True)[0]
    data_test_cleaned_pre = pre_process.pre_process_df(data_test_cleaned_pre)[0]
    encoded_columns = pre_process.pre_process_df(data_train_cleaned, Train=True)[1]

    features = ['start_station', 'trip_duration_calculated', 'start_hour', 'distance', 'is_weekend'] + list(encoded_columns)

    X_train = data_train_cleaned[features]
    y_train = data_train_cleaned['passholder_type_encoded']
    X_test = data_test_cleaned_pre[features]

    model_develop = ModelDevelop()
    print("\n[Train Test Split]")
    X_train_internal, X_val, y_train_internal, y_val = model_develop.train_test_split_df(X_train, y_train)
    X_train_balanced, y_train_balanced = model_develop.balance_data(X_train_internal, y_train_internal)
    X_train_balanced = feature_eng.add_features(X_train_balanced)
    X_val = feature_eng.add_features(X_val)
    X_test = feature_eng.add_features(X_test)

    print("\n[Entrenando modelos]")
    le = LabelEncoder()
    le.fit(data_train_cleaned['passholder_type']) 
    rf_model, xgb_model = model_develop.train_and_evaluate_models(X_train_balanced, y_train_balanced, X_val, y_val, le)

    y_test_rf = model_develop.predict_on_test_set(rf_model, X_test)
    y_test_xgb = model_develop.predict_on_test_set(xgb_model, X_test)

    print("\n[Guardando predicciones]")
    predictions_df = pd.DataFrame({
    'trip_id': data_test_cleaned['trip_id'],  # Identificador único
    'predicted_rf': y_test_rf,
    'predicted_xgb': y_test_xgb})

    predictions_df.to_csv("predicciones_finales_opp.csv", index=False)

if __name__ == "__main__":
    main()