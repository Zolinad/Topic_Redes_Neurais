# Exemplo 2: PERCEPTRON DE 1 CAMADA (com PRODUTO ESCALAR)
# Obs.: usando a bibliotecca numpy

import numpy as np

entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma(e, p):
    return e.dot(p)

s = soma(entradas, pesos)
print('Resultado do SOMATÓRIO:', s)

# Função Degrau para para ativação do Neurônio
def stepFunction(somatorio):
    if(somatorio >= 0):
        return 1
    return 0

y = stepFunction(s)
print("Resultado da função de ativação", y)