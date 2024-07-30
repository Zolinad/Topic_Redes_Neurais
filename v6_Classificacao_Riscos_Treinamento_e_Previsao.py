# Importando as bibliotecas
import numpy as np

#%%
# Definindo as funções utilizadas ao longo do código

# Função de Ativação de cada neurônio (neste caso é utilizada a função sigmóide)
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

# Derivada da função sigmóide (para o cálculo do delta)
def sigmoidDerivada(sig):
    return sig * (1 - sig)

# Após o treinamento, vamos fazer previsões de cada registro na base e associar as saídas binárias ao texto (Alto, Moderado, Baixo)
def previsao_saida(saida):
    limiar = 0.5  # Limiar para decidir se a saída está próxima de 1 ou de 0
    if np.allclose(saida, [1, 0, 0], atol=limiar):
        return "Alto"
    elif np.allclose(saida, [0, 1, 0], atol=limiar):
        return "Moderado"
    elif np.allclose(saida, [0, 0, 1], atol=limiar):
        return "Baixo"
    else:
        return "Saída não reconhecida"

# Função para fazer previsões de saídas para novas entradas, com base nos pesos treinados
def fazer_previsao(entrada):
    soma_sinapse0 = np.dot(entrada, pesos0)
    camada_oculta1 = sigmoid(soma_sinapse0)
    
    soma_sinapse1 = np.dot(camada_oculta1, pesos1)
    camada_oculta2 = sigmoid(soma_sinapse1)

    soma_sinapse2 = np.dot(camada_oculta2, pesos2)
    camada_saida = sigmoid(soma_sinapse2)
    
    return previsao_saida(camada_saida)


#%%
entradas = np.array([[3,1,1,1],
                     [2,1,1,2],
                     [2,2,1,2],
                     [2,2,1,3],
                     [2,2,2,3],
                     [3,2,1,3],
                     [3,2,2,3],
                     [1,2,1,3],
                     [1,2,1,3],
                     [1,1,2,3],
                     [1,1,1,1],
                     [1,1,1,2],
                     [1,1,1,3],
                     [1,1,1,3]])

saidas = np.array([[1,0,0],
                   [1,0,0],
                   [0,1,0],
                   [1,0,0],
                   [0,0,1],
                   [0,0,1],
                   [1,0,0],
                   [0,1,0],
                   [0,0,1],
                   [0,0,1],
                   [1,0,0],
                   [0,1,0],
                   [0,0,1],
                   [1,0,0]])

# Definindo a quantidade de neurônios em cada camada
quanti_entrada = entradas.shape[1] # quantidade de entradas
neuronios_oculta1 = 4 # calcular com base na fórmula (entradas + saídas) / 2
neuronios_oculta2 = 4 # repetir a quantidade de neurônios da camada oculta 1
neuronios_saida = saidas.shape[1] # quantidade de saídas

# Inicialização dos pesos
pesos0 = 2*np.random.random((quanti_entrada, neuronios_oculta1)) - 1
pesos1 = 2*np.random.random((neuronios_oculta1, neuronios_oculta2)) - 1
pesos2 = 2*np.random.random((neuronios_oculta2, neuronios_saida)) - 1

# Parâmetros de treinamento
epocas = 1000000
taxa_aprendizagem = 0.2
momento = 1

#%%
# TREINAMENTO DA REDE NEURAL (algoritmo Backpropagation, com pesos aleatórios e atualização dos pesos a cada época) 
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

#%%
k=0
# Realizando previsões para todas as entradas (registros da base)
for entrada, saida_esperada in zip(entradas, saidas):
    k+=1 
    
    soma_sinapse0 = np.dot(entrada, pesos0)
    camada_oculta1 = sigmoid(soma_sinapse0)
    
    soma_sinapse1 = np.dot(camada_oculta1, pesos1)
    camada_oculta2 = sigmoid(soma_sinapse1)

    soma_sinapse2 = np.dot(camada_oculta2, pesos2)
    camada_saida = sigmoid(soma_sinapse2)
    
    print("Entrada ", k, ":", entrada)
    print("Saída Esperada:", previsao_saida(saida_esperada))
    print("Saída Prevista:", previsao_saida(camada_saida))
    print()

#%%

# INSERINDO NOVASS ENTRADAS PARA OBTENÇÃO DE PREVISÕES (algoritmo Feedfoward com pesos treinados)
novas_entradas = []
num_entradas = int(input("Para quantas novas entradas você deseja fazer previsões? "))

for i in range(num_entradas):
    entrada = [float(x) for x in input(f"Insira os valores da entrada {i+1} separados por vírgula: ").split(",")]
    novas_entradas.append(entrada) 

novas_entradas = np.array(novas_entradas)
1
# Realizando previsões para as novas entradas
for i, entrada in enumerate(novas_entradas):
    print()
    print("Nova Entrada ", i+1, ":", entrada)
    print("Saída Prevista:", fazer_previsao(entrada))
# %%
