# Exemplo 4: perceptron de 1 camada (para uma porta lógica AND)

import numpy as np

# Definindo as entradas e saídas para a porta lógica AND
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas = np.array([0, 0, 0, 1])

# Inicializando os pesos e a taxa de aprendizagem
pesos = np.array([0.1, 0.1])
taxaAprendizagem = 0.1

# Função de ativação (step function)
def stepFunction(soma):
    # Retorna 1 se a soma for maior ou igual a 1, caso contrário, retorna 0
    if soma >= 1:
        return 1
    return 0

# Função para calcular a saída do perceptron
def calculaSaida(registro):
    # Calcula a soma ponderada das entradas e pesos, e aplica a função de ativação (chamando a função stepFunction)
    s = registro.dot(pesos)
    return stepFunction(s)

# Função de treinamento do perceptron
def treinar():
    erroTotal = 1
    while erroTotal != 0:
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(entradas[i])
            # Calcula o erro absoluto entre a saída calculada e a saída desejada
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                # Atualiza os pesos usando a regra de aprendizado do perceptron
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print("Peso atualizado: ", pesos[j])
                
        print("Total de Erros: ", erroTotal)
        print('\n')

# Chamando a função de treinamento
treinar()

# Chamando a função para calcular a saída do perceptron após o treinamento.
print('Rede Neural Treinada' + '\n' + 'Saídas calculadas para a tabela verdade AND')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
