import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, \
    roc_curve, average_precision_score
import matplotlib.pyplot as plt

# Пункт 1
data = pd.read_csv('data.csv')

# Пункт 2
target_column = 'y'
task_type = 'binary_classification' if data[target_column].nunique() == 2 else 'multiclass_classification'

# Пункт 3
X = data.drop(target_column, axis=1)
y = data[target_column]

# D Кодирование категориальных признаков на всей выборке перед разделением
categorical_cols = X.select_dtypes(include=[object]).columns
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

# Создание нового DataFrame с кодированными данными
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

# Объединение кодированных категориальных признаков с численными признаками
X_combined = pd.concat([X.drop(categorical_cols, axis=1), X_encoded_df], axis=1)

# A
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# B
numerical_cols = X_train.select_dtypes(include=[np.number]).columns

numerical_imputer = SimpleImputer(strategy='mean')
X_train[numerical_cols] = numerical_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = numerical_imputer.transform(X_test[numerical_cols])

# C
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Пункт 4
models = [
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42)),
    ('KNeighborsClassifier', KNeighborsClassifier()),
    ('RandomForestClassifier', RandomForestClassifier(random_state=42))
]

for name, model in models:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Пункт 5
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Графики ROC-AUC и PR-кривые
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train, tpr_train, label=f'Train ROC-AUC = {train_roc_auc:.2f}')
    plt.plot(fpr_test, tpr_test, label=f'Test ROC-AUC = {test_roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_train, precision_train, label=f'Train PR-AUC = {average_precision_score(y_train, y_train_proba):.2f}')
    plt.plot(recall_test, precision_test, label=f'Test PR-AUC = {average_precision_score(y_test, y_test_proba):.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    print(f"{name} - Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
    print(f"{name} - Train ROC-AUC: {train_roc_auc}, Test ROC-AUC: {test_roc_auc}")
    print(f"{name} - Train F1-score: {train_f1}, Test F1-score: {test_f1}")

# Шаг 6: Сравнить метрики и ответить на вопросы
# Здесь вы должны проанализировать полученные результаты и ответить на вопросы