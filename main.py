import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df_iris = pd.read_csv("./Iris.csv")

print("--- Primeras Filas ---")
print(df_iris.head(), end="\n\n")

print("--- Información General ---")
print(df_iris.info(), end="\n\n")

print("--- Estadísticas Descriptivas ---")
print(df_iris.describe(), end="\n\n")

print("--- Valores Nulos ---")
print(df_iris.isnull().sum(), end="\n\n")

duplicados = df_iris.duplicated().sum()
print(f"Total de filas duplicadas: {duplicados}")

if duplicados > 0:
    print("Filas duplicadas detectadas:")
    print(df_iris[df_iris.duplicated()])
    df_iris = df_iris.drop_duplicates().reset_index(drop=True)
    print("Duplicados eliminados.", end="\n\n")


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribución de las Variables Numéricas')

sns.histplot(x=df_iris['SepalLengthCm'], kde=True, ax=axes[0, 0], color='skyblue')
sns.histplot(x=df_iris['SepalWidthCm'], kde=True, ax=axes[0, 1], color='salmon')
sns.histplot(x=df_iris['PetalLengthCm'], kde=True, ax=axes[1, 0], color='green')
sns.histplot(x=df_iris['PetalWidthCm'], kde=True, ax=axes[1, 1], color='purple')
plt.show()



plt.figure(figsize=(8, 6))

corr = df_iris.drop(['Id', 'Species'], axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_iris.drop(columns=['Id']), palette="Set2")
plt.title("Análisis de Outliers (Valores Atípicos)")
plt.show()

print("Análisis completado.")