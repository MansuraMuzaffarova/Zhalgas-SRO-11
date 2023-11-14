# Zhalgas-SRO-11
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
# Загрузка данных
try:
    data = pd.read_csv("C:/Users/XE/Downloads/populationkz.csv")
except FileNotFoundError:
    print("Файл не найден.")
    exit(1)

# Подготовка данных
X = data[['latitude', 'longitude']]
y = data['population_2020'] - data['population_2015']

# Разбиение на обучающий, валидационный и тестовый наборы
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Определение параметров для каждой модели в ансамбле
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
param_grid_gb = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 8]}
param_grid_xgb = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [4, 6, 8]}

# Обучение моделей и поиск лучших параметров
grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=3, scoring='neg_mean_squared_error')
grid_search_gb = GridSearchCV(GradientBoostingRegressor(), param_grid_gb, cv=3, scoring='neg_mean_squared_error')
grid_search_xgb = GridSearchCV(XGBRegressor(), param_grid_xgb, cv=3, scoring='neg_mean_squared_error')

grid_search_rf.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)

# Создание ансамбля
ensemble = VotingRegressor(estimators=[
    ('rf', grid_search_rf.best_estimator_),
    ('gb', grid_search_gb.best_estimator_),
    ('xgb', grid_search_xgb.best_estimator_)
])

# Обучение ансамбля
ensemble.fit(X_train, y_train)

# Прогнозирование и оценка качества
predictions = ensemble.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
# Визуализация результатов
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual Values vs Predicted Values')
plt.show()
