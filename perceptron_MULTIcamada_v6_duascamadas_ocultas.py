import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])

# Definindo a quantidade de neurônios em cada camada
neuronios_entrada = entradas.shape[1]
neuronios_oculta1 = 4
neuronios_oculta2 = 4
neuronios_saida = saidas.shape[1]

# Inicialização dos pesos
pesos0 = 2*np.random.random((neuronios_entrada, neuronios_oculta1)) - 1
pesos1 = ...
pesos2 = ....

# Parâmetros de treinamento
epocas = 1000000
taxa_aprendizagem = 0.2
momento = 1

for j in range(epocas):
    # Feedforward
    camada_entrada = entradas
    soma_sinapse0 = np.dot(camada_entrada, pesos0)
    camada_oculta1 = sigmoid(soma_sinapse0)
    
    soma_sinapse1 = np.dot(camada_oculta1, pesos1)
    camada_oculta2 = sigmoid(soma_sinapse1)

    soma_sinapse2 = np.dot(camada_oculta2, pesos2)
    camada_saida = sigmoid(soma_sinapse2)
    
    # Cálculo do erro
    erro_camada_saida = saidas - camada_saida
    media_absoluta = np.mean(np.abs(erro_camada_saida))
    if j % 10000 == 0:
        print("Erro: " + str(media_absoluta))
    
    # Retropropagação
    derivada_saida = sigmoidDerivada(camada_saida)
    delta_saida = erro_camada_saida * derivada_saida
    
    pesos2_transposta = pesos2.T
    delta_saida_x_peso2 = delta_saida.dot(pesos2_transposta)
    delta_camada_oculta2 = delta_saida_x_peso2 * sigmoidDerivada(camada_oculta2)

    pesos1_transposta = pesos1.T
    delta_oculta2_x_peso1 = delta_camada_oculta2.dot(pesos1_transposta)
    delta_camada_oculta1 = delta_oculta2_x_peso1 * sigmoidDerivada(camada_oculta1)
    
    # Atualização dos pesos
    camada_oculta2_transposta = camada_oculta2.T
    pesos_novo2 = camada_oculta2_transposta.dot(delta_saida)
    pesos2 = (pesos2 * momento) + (pesos_novo2 * taxa_aprendizagem)
    
    camada_oculta1_transposta = camada_oculta1.T
    pesos_novo1 = camada_oculta1_transposta.dot(delta_camada_oculta2)
    pesos1 = (pesos1 * momento) + (pesos_novo1 * taxa_aprendizagem)

    camada_entrada_transposta = camada_entrada.T
    pesos_novo0 = camada_entrada_transposta.dot(delta_camada_oculta1)
    pesos0 = (pesos0 * momento) + (pesos_novo0 * taxa_aprendizagem)