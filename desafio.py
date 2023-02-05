#bibliotecas
#essas são as bibliotecas que utilizei para o projeto, estou fazendo o projeto no visual studio code e usando o python 3.9.12 do anaconda
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

#puxando o arquivo xlsx
Tabela = pd.read_excel('Dados.xlsx')

# Converte a coluna "Data" para o formato datetime
Tabela["Data"] = pd.to_datetime(Tabela["Data"])

# Cria a coluna "Dia da Semana" com o número do dia da semana (0-6)
Tabela["Dia da Semana"] = Tabela["Data"].dt.dayofweek

# Muda o valor para começar pela terça 
Tabela["Dia da Semana"] = np.where(Tabela["Dia da Semana"] == 0, 6, Tabela["Dia da Semana"] - 1)

print(Tabela)

y = Tabela["Vendas"] # é de consenso que o "y" é aquilo que vc deseja prever 
x = Tabela[["Dia da Semana"]]# e o "x" é o que vc quer usar para prever o y

from sklearn.model_selection import train_test_split

#vai treinar a ia para que ela tente prever as vendas futuras 
# estou separando os dados em dados de treino e dados de teste para garantir a funcionalidade da ia, cerca de 80% em treino e 20% em teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.8) 

# esses são dois modelos de ia que usarei para o teste, o objetivo é uusar aquela que for mais precisa, então eu irei testar as duas 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
modelo_regressaolinear = LinearRegression()
modelo_arvore = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino) # aqui estou entregando os dados para os modelos
modelo_arvore.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvore = modelo_arvore.predict(x_teste)

from sklearn.metrics import r2_score

#isso da uma nota especie de nota a ia, para ter ideia do quão preciso ou viciado ela esta a nota vai de -1 a 1
print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvore))

#com isso chegamos a conclusão que a ia arvore de decisao é mais eficiente para esse caso, então vamos ver a predicao dela
tabela_a = pd.DataFrame()
tabela_a["y_teste"] = y_teste
tabela_a["previsao arvore decisao"] = previsao_arvore
tabela_a["previsao regressao linear"] = previsao_regressaolinear
print(tabela_a)
sns.lineplot(data=tabela_a)
plt.show()

#estou usando isso para fazer a previsão dos proximos 5 dias solicitados no teste da vaga, 
dias_5 = pd.date_range(start='20230120', periods=5, freq='D')
dias_5df = pd.DataFrame(dias_5, columns=['Dia da Semana'])
dias_5_predicao = modelo_arvore.predict(dias_5df)
print(dias_5_predicao)

#criando uma segunda tabela para não alterar a original
Tabela2 = Tabela
Tabela2['nome_dia'] = Tabela2['Data'].dt.strftime("%A")

# Tradução dos nomes dos dias da semana
taducao = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
Tabela2['nome_dia'] = Tabela2['nome_dia'].map(taducao)

# Reordena as linhas da tabela para começar na terça
Tabela2 = Tabela2.sort_values(by='nome_dia', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
Tabela2 = Tabela2.reset_index(drop=True)

# Agrupa as linhas da tabela pelo dia da semana
grouped = Tabela2.groupby('nome_dia').sum()

# Reordena as linhas da tabela para começar na terça
grouped = grouped.reindex(['Terca', 'Quarta', 'Quinta', 'Sexta', 'Sabado', 'Domingo', 'Segunda'])

grouped.drop(columns=['Dia da Semana'], inplace=True)

#estou usando isso para criar uma tabela onde aparecem o total de vendas pra cada dia, acredito que isso seja util para pensar
#em estrategias de promoção ou coisas do tipo, tentei fazer o maximo de analises possivel
print(grouped) 
