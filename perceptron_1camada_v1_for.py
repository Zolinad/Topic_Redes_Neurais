
# Exemplo 1: PERCEPTRON DE 1 CAMADA (com laço for)

entradas = [1, 7, 5]
pesos = [0.8, 0.1, 0]

def soma(e,p):
    s = 0
    for i in range(3):
        s_i = e[i] * p[i]
        s += s_i
        print(e[i],'*', p[i], '=', s_i)
    return s
       
s = soma(entradas, pesos)
print('Resultado da soma:',s) #resultado da soma dos produtos


#Função Degrau para ativação do Neurônio
def stepFunction(soma):
    if (soma >= 0):
        return 1
    return 0

y = stepFunction(s)
print("Resultado da função de ativação:",y) #resultado da função de ativação