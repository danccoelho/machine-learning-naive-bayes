# %%
# Naive Bayes aplicado a Risco de Crédito, Dados de Empréstimos e Censo

# %%
# ========== Imports Gerais ==========
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from yellowbrick.classifier import ConfusionMatrix

# %%
# ========== Parte 1: Risco de Crédito ========== 

# %%
base_risco = pd.read_csv('risco_credito.csv')
base_risco.describe()

# %%
X_risco_credito = base_risco.iloc[:, 0:4].values
y_risco_credito = base_risco.iloc[:,4].values
X_risco_credito

# %%
# Codificação de variáveis categóricas
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

# %%
X_risco_credito

# %%
# Salvando o dataset processado
import pickle
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

# %%
# Treinamento
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

# %%
# Previsão exemplo
previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
print("Previsão risco crédito:", previsao)

# %%
# Informacoes do modelo
print(naive_risco_credito.classes_)
print(naive_risco_credito.class_count_)
print(naive_risco_credito.class_prior_)

# %%
# ========== Parte 2: Dados de Crédito ==========
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_teste, y_teste = pickle.load(f)

# %%
naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

# %%
previsoes = naive_credit_data.predict(X_teste)

# %%
previsoes

# %%
y_teste

# %%
# Visualização da matriz de confusão
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_teste, y_teste)

# %%
# Visualização da matriz de confusão com seaborn
sns.heatmap(confusion_matrix(y_teste, previsoes), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Crédito")
plt.show()

# %%
# Avaliação
print("Acurácia (crédito):", accuracy_score(y_teste, previsoes))
print("Matrix:")
print(confusion_matrix(y_teste, previsoes))
print()
print(classification_report(y_teste, previsoes))

# %%
# ========== Parte 3: Censo ==========
with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

# %%
naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, y_census_treinamento)


# %%
previsoes = naive_census.predict(X_census_teste)
previsoes

# %%
y_census_teste

# %%
cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste,y_census_teste)

# %%
# Avaliação
print("Acurácia (censo):", accuracy_score(y_census_teste, previsoes))
print("Matrix:")
print(confusion_matrix(y_census_teste, previsoes))
print()
print(classification_report(y_census_teste, previsoes))



