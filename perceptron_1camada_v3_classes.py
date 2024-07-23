# Exemplo 3: PERCEPTRON DE 1 CAMADA (usando NUMPY)
# para Classificação entre grupos

import numpy as np
entradas = np.array([4,3])
pesos = np.array([0.21, 0.22])

# Definindo a função para o somatório
def soma(e,p):
    return e.dot(p)

s = soma(entradas, pesos)
print('Valor do somatório:', s)

# Definindo a função para a função de ativação
def stepFunction(somatorio):
    if (somatorio >=  2):
        return "Classe B"
    return "Classe A"

y = stepFunction(s)
print("O resultado da classificação é:", y)
