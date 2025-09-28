import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
df = pd.read_csv("/Users/anamaksimova/Documents/healthcare_dataset.csv")

# Просмотр первых строк
print(df.head())

# Размер и столбцы
print(df.shape)
print(df.columns)

# Проверка пропусков
print(df.isnull().sum())

# Обзор статистики
print(df.describe())

# Удаляем строки с пустыми значениями
df = df.dropna()

# Или заполняем числовые колонки средним значением
df['Age'] = df['Age'].fillna(df['Age'].mean())

# ---- ВИЗУАЛИЗАЦИИ ----
# Гистограмма возраста пациентов
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Распределение возраста пациентов")
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.show()

# Boxplot: возраст по полу
plt.figure(figsize=(8,5))
sns.boxplot(x='Gender', y='Age', data=df)
plt.title("Возраст пациентов по полу")
plt.show()

# Boxplot: сумма счета по типу госпитализации
plt.figure(figsize=(8,5))
sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
plt.title("Сумма счета по типу госпитализации")
plt.show()

# ---- ДОПОЛНИТЕЛЬНЫЕ АНАЛИЗЫ ----
avg_age_by_gender = df.groupby('Gender')['Age'].mean()
avg_bill_by_admission = df.groupby('Admission Type')['Billing Amount'].mean()
top_conditions = df['Medical Condition'].value_counts().head(3)

# ---- ВЫВОДЫ ----
print("Выводы:")
print(f"1. Средний возраст пациентов по полу:\n{avg_age_by_gender}\n")
print(f"2. Наиболее распространённые медицинские состояния:\n{top_conditions}\n")
print(f"3. Средняя сумма счета по типу госпитализации:\n{avg_bill_by_admission}\n")

most_expensive = avg_bill_by_admission.idxmax()
cheapest = avg_bill_by_admission.idxmin()
print(f"4. Самая дорогая госпитализация: {most_expensive}, самая дешёвая: {cheapest}.")
print(f"5. На гистограмме видно, что большинство пациентов — в возрастной группе "
      f"{int(df['Age'].quantile(0.25))}-{int(df['Age'].quantile(0.75))} лет.")


