# Regressão Linear

*Importância da remoção de variáveis sem significância estatística:*

Excluir variáveis sem significância estatística em um modelo de regressão linear, mesmo que o \( R^2 \) diminua, é importante por algumas razões principais:

### 1. **Multicolinearidade e Estabilidade do Modelo:**
   Variáveis não significativas podem estar correlacionadas com outras variáveis explicativas (multicolinearidade), o que pode gerar instabilidade no modelo. Isso significa que pequenos desvios nos dados podem causar grandes mudanças nos coeficientes estimados. Ao remover essas variáveis, o modelo tende a ser mais robusto e os coeficientes se tornam mais confiáveis.

### 2. **Simplicidade e Interpretabilidade:**
   Modelos mais simples e com menos variáveis são mais fáceis de interpretar e entender. Se uma variável não é estatisticamente significativa, ela não contribui para explicar a variação da variável dependente de forma confiável. Mantê-la pode criar a ilusão de que ela tem um impacto relevante, dificultando a compreensão dos resultados.

### 3. **Evitar Overfitting:**
   Incluir variáveis irrelevantes aumenta o risco de overfitting, o que significa que o modelo pode se ajustar muito bem aos dados de treinamento, mas generalizar mal para novos dados. Remover variáveis sem significância ajuda a construir um modelo que tenha melhor capacidade de generalização.

### 4. **Significância Estatística e Inferências:**
   Variáveis sem significância estatística têm coeficientes que não são significativamente diferentes de zero, indicando que não há evidência suficiente de que elas afetam a variável dependente. Se a meta é fazer inferências sobre quais fatores influenciam o resultado, manter variáveis irrelevantes pode levar a conclusões erradas.

### 5. **Redução de Ruído:**
   Manter variáveis sem significância adiciona ruído ao modelo. Essas variáveis não contribuem de forma significativa para prever a variável dependente, mas podem introduzir variações que dificultam a identificação das relações reais.

---

Apesar de a remoção de variáveis não significativas às vezes resultar em uma queda no \( R^2 \), isso ocorre porque o \( R^2 \) reflete apenas o ajuste global do modelo aos dados, incluindo variáveis irrelevantes que podem aumentar artificialmente o valor. No entanto, o objetivo é ter um modelo parsimonioso, onde as variáveis incluídas tenham um impacto real, e não maximizar o \( R^2 \) a todo custo.
