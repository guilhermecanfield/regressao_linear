
# In[EXEMPLO 6]:
#############################################################################
#         DIAGNÓSTICO DE MULTICOLINEARIDADE EM MODELOS DE REGRESSÃO         #
#                EXEMPLO 6 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_salarios = pd.read_csv('salarios.csv', delimiter=',')
df_salarios

# Características das variáveis do dataset
df_salarios.info()

# Estatísticas univariadas
df_salarios.describe()

# In[6.1]: Matriz de correlações

correlation_matrix = df_salarios.iloc[:,1:6].corr()
correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".3f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 20}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.2]: CORRELAÇÃO BAIXA (variáveis 'rh1' e 'econometria1'):

# Correlação entre 'rh1' e 'econometria1', com p-value
corr1, p_value1 = pearsonr(df_salarios['rh1'], df_salarios['econometria1'])
"{:.4f}".format(corr1), "{:.4f}".format(p_value1)

# Matriz de correlação (maneira simples) pela função 'corr'
corr1 = df_salarios[['rh1','econometria1']].corr()
corr1

# Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

corr1b = pg.rcorr(df_salarios[['rh1','econometria1']], method='pearson',
                  upper='pval', decimals=6,
                  pval_stars={0.01: '***',
                              0.05: '**',
                              0.10: '*'})
corr1b

# Mapa de calor com a correlação entre 'rh1' e 'econometria1'
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(corr1, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 30}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.3]: Grafo com a inter-relação entre as variáveis do dataframe 'df1'

df1 = df_salarios[['salario','rh1','econometria1']]
cormat1 = df1.corr()
cormat1

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in cormat1.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(cormat1.columns):
    for j, variable2 in enumerate(cormat1.columns):
        if i != j:
            correlation = cormat1.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('viridis_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.9)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Definição dos ticks da colorbar
cbar.set_ticks(np.arange(round(min(correlations),1),
                         max(correlations), 0.1))

# Exibição do gráfico
plt.show()

# In[6.4]: Modelo 1

modelo1 = sm.OLS.from_formula('salario ~ rh1 + econometria1', df_salarios).fit()

modelo1.summary()

# In[6.5]: Diagnóstico de multicolinearidade (Variance Inflation Factor
#e Tolerance)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculando os valores de VIF
X1 = sm.add_constant(df_salarios[['rh1', 'econometria1']])
VIF = pd.DataFrame()
VIF["Variável"] = X1.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X1.values, i+1)
              for i in range(X1.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF

# In[6.6]: CORRELAÇÃO MUITO ALTA (variáveis 'rh2' e 'econometria2'):

# Correlação entre 'rh2' e 'econometria2', com p-value
corr2, p_value2 = pearsonr(df_salarios['rh2'], df_salarios['econometria2'])
"{:.4f}".format(corr2), "{:.4f}".format(p_value2)

# Matriz de correlação (maneira simples) pela função 'corr'
corr2 = df_salarios[['rh2','econometria2']].corr()
corr2

# Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg

corr2b = pg.rcorr(df_salarios[['rh2','econometria2']], method='pearson',
                  upper='pval', decimals=6,
                  pval_stars={0.01: '***',
                              0.05: '**',
                              0.10: '*'})
corr2b

# Mapa de calor com a correlação entre 'rh2' e 'econometria2'
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(corr2, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 30}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.7]: Grafo com a inter-relação entre as variáveis do dataframe 'df2'

df2 = df_salarios[['salario','rh2','econometria2']]
cormat2 = df2.corr()
cormat2

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in cormat2.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(cormat2.columns):
    for j, variable2 in enumerate(cormat2.columns):
        if i != j:
            correlation = cormat2.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('viridis_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.9)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Definição dos ticks da colorbar
cbar.set_ticks(np.arange(round(min(correlations) - 0.01,2),
                         max(correlations) + 0.01, 0.01))

# Exibição do gráfico
plt.show()

# In[6.8]: Modelo 2

modelo2 = sm.OLS.from_formula('salario ~ rh2 + econometria2', df_salarios).fit()

modelo2.summary()

# In[6.9]: Diagnóstico de multicolinearidade (Variance Inflation Factor
#e Tolerance)

# Calculando os valores de VIF
X2 = sm.add_constant(df_salarios[['rh2', 'econometria2']])
VIF = pd.DataFrame()
VIF["Variável"] = X2.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X2.values, i+1)
              for i in range(X2.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF


# In[EXEMPLO 7]:
#############################################################################
#        DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO         #
#               EXEMPLO 7 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################
    
df_saeb_rend = pd.read_csv('saeb_rend.csv', delimiter=',')
df_saeb_rend

# Características das variáveis do dataset
df_saeb_rend.info()

# Estatísticas univariadas
df_saeb_rend.describe()

# In[7.1]: Tabela de frequências absolutas das variáveis 'uf' e rede'

df_saeb_rend['uf'].value_counts().sort_index()
df_saeb_rend['rede'].value_counts().sort_index()

# In[7.2]: Plotando a variável 'saeb' em função de 'rendimento', com fit linear
# Gráfico pela função 'regplot' do 'seaborn'

plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df_saeb_rend, marker='o',
            color='royalblue', ci=False,
            scatter_kws={'color':'lightsalmon', 'alpha':0.5, 's':150},
            line_kws={'linewidth': 4})
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=22)
plt.xlabel('rendimento', fontsize=20)
plt.ylabel('saeb', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.3]: Plotando a variável 'saeb' em função de 'rendimento', com destaque
#para a 'rede' escolar e linear fits -> Gráfico pela função 'regplot' do
#pacote 'seaborn'

# Definição de dataframes com subgrupos por 'rede'
df1 = df_saeb_rend[df_saeb_rend['rede'] == 'Municipal']
df2 = df_saeb_rend[df_saeb_rend['rede'] == 'Estadual']
df3 = df_saeb_rend[df_saeb_rend['rede'] == 'Federal']

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df1, marker='o', ci=False,
            scatter_kws={'color':'darkorange', 'alpha':0.3, 's':150},
            line_kws={'color':'darkorange', 'linewidth': 4}, label='Municipal')
sns.regplot(x='rendimento', y='saeb', data=df2, marker='o', ci=False,
            scatter_kws={'color':'darkviolet', 'alpha':0.3, 's':150},
            line_kws={'color':'darkviolet', 'linewidth': 4}, label='Estadual')
sns.regplot(x='rendimento', y='saeb', data=df3, marker='o', ci=False,
            scatter_kws={'color':'darkgreen', 'alpha':0.8, 's':150},
            line_kws={'color':'darkgreen', 'linewidth': 4}, label='Federal')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=22)
plt.xlabel('rendimento', fontsize=20)
plt.ylabel('saeb', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# In[7.4]: Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do 'modelo_saeb'
modelo_saeb.summary()

# In[7.5]: Adicionando fitted values e resíduos do 'modelo_saeb' no
# dataset 'df_saeb_rend'

df_saeb_rend['fitted'] = modelo_saeb.fittedvalues
df_saeb_rend['residuos'] = modelo_saeb.resid
df_saeb_rend

# In[7.6]: Gráfico que relaciona resíduos e fitted values do 'modelo_saeb'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted', y='residuos', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo', fontsize=20)
plt.ylabel('Resíduos do Modelo', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.7]: Histograma dos resíduos do 'modelo_saeb' com curva normal teórica
#para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_saeb.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_saeb.resid, bins=20, kde=True, stat="density",
             color='red', alpha=0.4)
plt.xlim(-4, 4)
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[7.8]: Função para o teste de Breusch-Pagan para a elaboração de diagnóstico
#de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value

# In[7.9]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[7.10]: Procedimento n-1 dummies para as unidades federativas
    
# Dummização da variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      dtype=int,
                                      drop_first=True)

df_saeb_rend_dummies

# In[7.11]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb',
                                                        'fitted',
                                                        'residuos']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo

# Estimação
modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

# Parâmetros do modelo 'modelo_saeb_dummies_uf'
modelo_saeb_dummies_uf.summary()

# In[7.12]: Estimação do modelo por meio do procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

modelo_saeb_dummies_uf_step = stepwise(modelo_saeb_dummies_uf, pvalue_limit=0.05)

# In[7.13]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf_step'

breusch_pagan_test(modelo_saeb_dummies_uf_step)

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb_dummies_uf_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[7.14]: Adicionando fitted values e resíduos do 'modelo_saeb_dummies_uf_step'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted_step'] = modelo_saeb_dummies_uf_step.fittedvalues
df_saeb_rend['residuos_step'] = modelo_saeb_dummies_uf_step.resid
df_saeb_rend

# In[7.15]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb_dummies_uf_step'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'dodgerblue', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Dummies', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Dummies', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.16]: Histograma dos resíduos do 'modelo_saeb_dummies_uf_step' com curva
#normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_saeb_dummies_uf_step.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_saeb_dummies_uf_step.resid, bins=20, kde=True,
             stat="density", color='dodgerblue', alpha=0.4)
plt.xlim(-4, 4)
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise com Dummies', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[7.17]: Plotando a variável 'saeb' em função de 'rendimento', com destaque
#para as unidades federativas e fits lineares - Gráfico pela função 'lmplot' do
#pacote 'seaborn', com estratificação de 'uf' pelo argumento 'hue'

uf_order = df_saeb_rend['uf'].value_counts().index.sort_values()

plt.figure(figsize=(15, 10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='uf', ci=None, palette='viridis', legend=False,
           scatter_kws={'alpha': 0.5},
           hue_order=uf_order)
plt.title('Gráfico de Dispersão com Ajuste Linear por UF', fontsize=13)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, ncol=3, bbox_to_anchor=(1, 0.75))
plt.show()


# In[EXEMPLO 8]:
#############################################################################
#                 REGRESSÃO NÃO LINEAR MÚLTIPLA COM DUMMIES                 #
#                 EXEMPLO 8 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_planosaude = pd.read_csv('planosaude.csv', delimiter=',')
df_planosaude

# Características das variáveis do dataset
df_planosaude.info()

# Estatísticas univariadas
df_planosaude.describe()

# In[8.1]: Tabela de frequências absolutas da variável 'plano'

df_planosaude['plano'].value_counts().sort_index()

# In[8.2]: Distribuições das variáveis, scatters, valores das correlações e
#suas respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.3f}".format(r),
                xy=(.30, .9), xycoords=ax.transAxes, fontsize=16)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.30, .8), xycoords=ax.transAxes, fontsize=16)

# Configuração do gráfico
sns.set(style="whitegrid", palette="viridis")

plt.figure(figsize=(20,10))
graph = sns.pairplot(df_planosaude.loc[:,'despmed':'renda'], diag_kind="kde",
                     plot_kws={"color": "darkorange"},
                     height=2.5, aspect=1.7)
graph.map(corrfunc)
for ax in graph.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=17)
    ax.set_ylabel(ax.get_ylabel(), fontsize=17)
plt.show()

# In[8.3]: Dummizando a variável 'plano' (n-1 dummies)

df_planosaude_dummies = pd.get_dummies(df_planosaude, columns=['plano'],
                                       dtype=int,
                                       drop_first=True)

df_planosaude_dummies

# In[8.4]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_planosaude_dummies.drop(columns=['id',
                                                         'despmed']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "despmed ~ " + formula_dummies_modelo

# Estimação
modelo_planosaude = sm.OLS.from_formula(formula_dummies_modelo,
                                        df_planosaude_dummies).fit()

# Parâmetros do modelo
modelo_planosaude.summary()

# In[8.5]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_planosaude = stepwise(modelo_planosaude, pvalue_limit=0.05)

# In[8.6]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.tests import shapiro_francia

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_planosaude.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[8.7]: Histograma dos resíduos do 'modelo_step_planosaude' com curva normal
#teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_planosaude.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_planosaude.resid, bins=15, kde=True, stat="density",
             color='red', alpha=0.4)
plt.xlim(-60, 70)
x = np.linspace(-60, 70, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[8.8]: Função para o teste de Breusch-Pagan para a elaboração de diagnóstico
#de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value

# In[8.9]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_step_planosaude)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_planosaude) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[8.10]: Adicionando fitted values e resíduos do 'modelo_step_planosaude'
#no dataframe 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step'] = modelo_step_planosaude.fittedvalues
df_planosaude_dummies['residuos_step'] = modelo_step_planosaude.resid
df_planosaude_dummies

# In[8.11]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_planosaude'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-50, 71, 20), fontsize=17)
x_min = df_planosaude_dummies['fitted_step'].min()-1
x_max = df_planosaude_dummies['fitted_step'].max()+1
plt.xlim(x_min, x_max)
plt.show()

# In[8.12]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_planosaude', com boundaries

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-50, 71, 20), fontsize=17)
x_min = df_planosaude_dummies['fitted_step'].min()-1
x_max = df_planosaude_dummies['fitted_step'].max()+1
plt.xlim(x_min, x_max)

sns.kdeplot(data=df_planosaude_dummies, x='fitted_step', y='residuos_step',
            levels=2, color='red', linewidths=3)

plt.show()

# In[8.13]: Transformação de Box-Cox

# Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(df_planosaude_dummies['despmed'])

print("Lambda: ",lmbda)

# In[8.14]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_planosaude_dummies['bc_despmed'] = yast
df_planosaude_dummies

# Verificação do cálculo, apenas para fins didáticos
df_planosaude_dummies['bc_despmed2'] = ((df_planosaude_dummies['despmed'])**\
                                        (lmbda) - 1) / (lmbda)
df_planosaude_dummies

del df_planosaude_dummies['bc_despmed2']

# In[8.15]: Estimando um novo modelo com todas as variáveis e a variável
#dependente transformada
modelo_bc_planosaude = sm.OLS.from_formula('bc_despmed ~ idade + dcron +\
                                           renda + plano_esmeralda +\
                                               plano_ouro',
                                               df_planosaude_dummies).fit()

# Parâmetros do modelo
modelo_bc_planosaude.summary()

# In[8.16]: Procedimento Stepwise no 'modelo_bc_planosaude'

modelo_step_bc_planosaude = stepwise(modelo_bc_planosaude, pvalue_limit=0.05)

# In[8.17]: Teste de verificação da aderência à normalidade dos resíduos do novo
#'modelo_step_bc_planosaude'

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_bc_planosaude.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[8.18]: Histograma dos resíduos do 'modelo_step_bc_planosaude' com curva
#normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_bc_planosaude.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_bc_planosaude.resid, bins=15, kde=True, stat="density",
             color='limegreen', alpha=0.4)
plt.xlim(-0.15, 0.15)
x = np.linspace(-0.15, 0.15, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[8.19]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_step_bc_planosaude'

breusch_pagan_test(modelo_step_bc_planosaude)

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_bc_planosaude) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[8.20]: Adicionando fitted values e resíduos do 'modelo_step_bc_planosaude'
#no dataframe 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step_bc'] = modelo_step_bc_planosaude.fittedvalues
df_planosaude_dummies['residuos_step_bc'] = modelo_step_bc_planosaude.resid
df_planosaude_dummies

# In[8.21]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_bc_planosaude'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step_bc', y='residuos_step_bc', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'limegreen', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-.15, .16, .05), fontsize=17)
x_min = df_planosaude_dummies['fitted_step_bc'].min()-0.01
x_max = df_planosaude_dummies['fitted_step_bc'].max()+0.01
plt.xlim(x_min, x_max)
plt.show()

# In[8.22]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_bc_planosaude', com boundaries

plt.figure(figsize=(15, 10))
sns.regplot(x='fitted_step_bc', y='residuos_step_bc', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color": 'limegreen', 'alpha': 0.5, 's': 200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-.15, .16, .05), fontsize=17)
x_min = df_planosaude_dummies['fitted_step_bc'].min()-0.01
x_max = df_planosaude_dummies['fitted_step_bc'].max()+0.01
plt.xlim(x_min, x_max)

sns.kdeplot(data=df_planosaude_dummies, x='fitted_step_bc', y='residuos_step_bc',
            levels=2, color='green', linewidths=3)

################################## FIM ######################################