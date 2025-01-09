import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel, RFECV

class ModelDevelop:
    
    def __init__(self):
        pass

    def balance_data(self, X, y):
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        under_sampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = under_sampler.fit_resample(X_smote, y_smote)
        return X_balanced, y_balanced

    def train_test_split_df(self, X_train, y_train):
        X_train_internal, X_val, y_train_internal, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        return X_train_internal, X_val, y_train_internal, y_val
    
    def select_features_with_model(self, X, y):
        model_temp = RandomForestClassifier(random_state=42)
        model_temp.fit(X, y)
        selector = SelectFromModel(model_temp, prefit=True, threshold="mean")
        selected_features = X.columns[selector.get_support()]
        X_selected = selector.transform(X)
        return X_selected, selected_features
    
    def grid_search_rf(self, X, y):
        print("\n[Optimized Search - Random Forest]")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=2,  # Reducir número de particiones
            scoring='accuracy',
            n_jobs=-1  # Paralelizar
        )
        grid_search.fit(X, y)
        print("\n[Random Forest] Mejores parámetros:", grid_search.best_params_)
        return grid_search.best_estimator_

    def grid_search_xgb(self, X, y):
        print("\n[Optimized Search - XGBoost]")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [6, 10],
            'learning_rate': [0.1],
            'scale_pos_weight': [1, 5]
        }
        grid_search = GridSearchCV(
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            param_grid=param_grid,
            cv=3,  # Reducir validación cruzada
            scoring='accuracy',
            n_jobs=-1  # Usar todos los núcleos
        )
        grid_search.fit(X, y)
        print("\n[XGBoost] Mejores parámetros:", grid_search.best_params_)
        return grid_search.best_estimator_
    
    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val, le):
        # 3.1 Validación cruzada para Random Forest
        #le = LabelEncoder()
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)

        print("\n[Random Forest - Validación Cruzada]")
        rf_model = self.grid_search_rf(X_train_balanced, y_train_balanced)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')
        print(f"[Random Forest] Accuracy promedio CV: {np.mean(rf_cv_scores):.4f} ± {np.std(rf_cv_scores):.4f}")

        # Validación interna para Random Forest
        rf_model.fit(X_train_balanced, y_train_balanced)
        y_val_pred_rf = rf_model.predict(X_val)
        print("\n[Evaluación en Validación Interna - Random Forest]")
        print("Accuracy:", accuracy_score(y_val, y_val_pred_rf))
        print("Classification Report:\n", classification_report(y_val, y_val_pred_rf, target_names=le.classes_))

        conf_matrix_rf = confusion_matrix(y_val, y_val_pred_rf)
        disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=le.classes_)
        disp_rf.plot(cmap='viridis')
        plt.title("Random Forest - Matriz de Confusión")
        plt.show()

        # 3.2 Validación cruzada para XGBoost
        print("\n[XGBoost - Validación Cruzada]")
        xgb_model = self.grid_search_xgb(X_train_balanced, y_train_balanced)
        xgb_cv_scores = cross_val_score(xgb_model, X_train_balanced, y_train_balanced, cv=kf, scoring='accuracy')
        print(f"[XGBoost] Accuracy promedio CV: {np.mean(xgb_cv_scores):.4f} ± {np.std(xgb_cv_scores):.4f}")

        # Validación interna para XGBoost
        xgb_model.fit(X_train_balanced, y_train_balanced)
        y_val_pred_xgb = xgb_model.predict(X_val)
        print("\n[Evaluación en Validación Interna - XGBoost]")
        print("Accuracy:", accuracy_score(y_val, y_val_pred_xgb))
        print("Classification Report:\n", classification_report(y_val, y_val_pred_xgb, target_names=le.classes_))

        conf_matrix_xgb = confusion_matrix(y_val, y_val_pred_xgb)
        disp_xgb = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_xgb, display_labels=le.classes_)
        disp_xgb.plot(cmap='viridis')
        plt.title("XGBoost - Matriz de Confusión")
        plt.show()

        return rf_model, xgb_model
    
    def plot_feature_importance(self, model, feature_names, title):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title(title)
        plt.show()
