# Como o sistema funciona

## Resumo

O sistema lê avaliações de um produto, classifica cada uma como **negativa** (1 ponto), **neutra** (3) ou
**positiva** (5) e dá ao produto uma nota em estrelas igual à média dos pontos. Usa bag of words com TF-IDF e o
classificador KNN.

**0. Dados.** Usamos o B2W-Reviews01, com cerca de 130 mil avaliações reais da Americanas, cada uma com a nota do
cliente. A nota vira o rótulo: 1–2 estrelas = negativa, 3 = neutra, 4–5 = positiva. Sorteamos 5.000 avaliações por
classe (15.000 no total), para o modelo não favorecer a classe mais comum. As 40 avaliações de uma panela de pressão
ficaram de fora, para testar o sistema.

**1. Entrada.** O usuário cola as avaliações, uma por linha.

**2. Pré-processamento.** Padroniza o texto para que palavras equivalentes contem como o mesmo termo: minúsculas,
sem pontuação e números, sem stopwords, com stemização e sem acentos. Mantivemos palavras que mudam o sentimento,
como "não", "muito", "mas" e "ou". Ex.: "não pega pressão, péssima" → `nao peg press pess`.

**3. BOW + TF-IDF.** O BOW monta um léxico de 114 mil termos (palavras e pares de palavras) e conta quantas vezes
cada um aparece. O TF-IDF dá mais peso aos termos raros: "nao" aparece em 6 mil avaliações e pesa pouco; "pega
pressão" aparece em 2 e pesa muito. Cada avaliação vira um vetor numérico.

**4. Treinamento.** Feito uma vez: o BOW aprende o léxico, o TF-IDF aprende o peso de cada termo e o KNN guarda os
15.000 vetores com seus rótulos. O modelo fica salvo em um arquivo de 8 MB.

**5. Classificação.** A avaliação nova vira um vetor e é comparada, pela similaridade de cosseno, com as 15.000
guardadas. As 25 mais parecidas votam no próprio rótulo, e quanto mais parecida, mais vale o voto. Vence a classe
com a maior soma. Ex.: 83,8% dos pesos foram para "negativa".

**6. Pontuação.** Cada classe vira pontos (1, 3 ou 5) e a média vira estrelas, arredondada para meia estrela.
Ex.: a panela tem média 3,95, então ganha 4 estrelas (a média real dos clientes é 3,6).

**7. Publicação.** A página web mostra a nota em estrelas e cada avaliação com seu sentimento.

**Resultado.** 69% de acerto na validação cruzada e 29 de 40 acertos na panela. As principais dificuldades são a
classe neutra, que mistura elogios e críticas, e o fato de o bag of words ignorar o contexto: "não decepciona"
parece uma reclamação.

---

# Cada etapa em detalhe

Para cada etapa: por quê, como e resultado. Exemplo usado em todas as etapas:
**"A panela não pega pressão, péssima, não recomendo"**.

---

## Etapa 0 — Dados (`etapa0_dados.py`)

**Por quê:** o KNN classifica comparando com exemplos já rotulados; sem avaliações reais com rótulo, não há com
o que comparar.

**Como:**
- Base **B2W-Reviews01** (Americanas, CC BY-NC-SA 4.0): 132.373 avaliações reais, cada uma com a nota de 1 a 5
  estrelas do cliente; 129.098 têm texto.
- A nota vira o rótulo: **1–2 estrelas = negativa, 3 = neutra, 4–5 = positiva**.
- Sorteio balanceado de **5.000 avaliações por classe** (a base original tem 2× mais positivas que negativas e
  5× mais que neutras; sem balancear, o modelo tenderia a dizer "positiva").
- As 40 avaliações de um produto (Panela de Pressão Rochedo) ficam **fora do treino** para testar o sistema.

**Resultado:** `treino.csv` com 15.000 avaliações rotuladas e `produto_demo.csv` com 40 avaliações inéditas
(média real dos clientes: 3,6 estrelas).

---

## Etapa 1 — Entrada (`etapa1_entrada.py`)

**Por quê:** o sistema precisa de uma série de opiniões sobre um produto.

**Como:** o usuário cola as avaliações na página, uma por linha (linhas vazias são ignoradas). O nome do produto
é opcional e serve só de título.

**Resultado:** uma lista de textos, e.g. `["A panela não pega pressão, péssima, não recomendo", ...]`.

---

## Etapa 2 — Pré-processamento (`etapa2_preprocessamento.py`)

**Por quê:** reduzir variações que não mudam o sentido ("Péssima", "péssimo", "pessimo"), para que palavras
equivalentes contem como o mesmo termo, e descartar o que não indica sentimento.

**Como:** minúsculas → remove URLs, pontuação e números → separa em palavras → remove stopwords → stemização
(RSLP, stemmer para o português) → remove acentos.
A lista de stopwords do NLTK foi ajustada: **"não", "nem", "nunca", "muito", "mais", "sem", "mas", "ou"... são
mantidas**, porque mudam o sentimento ("não gostei", "muito ruim", "bom, mas demorou", "mais ou menos").

**Resultado:**
```
original        A panela não pega pressão, péssima, não recomendo
tokens          ['a', 'panela', 'não', 'pega', 'pressão', 'péssima', 'não', 'recomendo']
sem stopwords   ['panela', 'não', 'pega', 'pressão', 'péssima', 'não', 'recomendo']
stemização      ['panel', 'não', 'peg', 'press', 'péss', 'não', 'recom']
sem acentos     ['panel', 'nao', 'peg', 'press', 'pess', 'nao', 'recom']
```

---

## Etapa 3 — Bag of words com TF-IDF (`etapa3_vetorizacao.py`)

**Por quê:** o modelo só compara números; cada texto precisa virar um vetor. São duas transformações em sequência:
o BOW diz **quais termos aparecem e quantas vezes**; o TF-IDF diz **quanto cada termo importa**.

### 3a. Bag of words (`CountVectorizer`)

**Como:** monta o **léxico** com todos os termos das 15.000 avaliações de treino — palavras (unigramas) e pares de
palavras (bigramas, como "nao recom") — e representa cada avaliação pela contagem de cada termo. A ordem das
palavras é descartada (por isso "saco de palavras"), exceto o pouco que os bigramas preservam.

**Resultado:** léxico de **114.138 termos**; cada avaliação é um vetor de 114.138 posições, quase todas zero
(em média 28 termos presentes). O exemplo tem 11 termos, todos com contagem 1, exceto **"nao", que aparece 2 vezes**.

### 3b. TF-IDF (`TfidfTransformer`)

**Por quê:** na contagem pura, "nao" (2) vale o dobro de "pess" (1), mas "nao" aparece em milhares de avaliações
e diz pouco sozinho. Termos raros são mais informativos.

**Como:**
- **TF** = contagem do termo na avaliação (do BOW);
- **IDF** = ln((1 + N) / (1 + nº de avaliações com o termo)) + 1, com N = 15.000, calculado no treino — alto para
  termos raros, baixo para termos comuns;
- peso = TF × IDF, e o vetor é dividido pelo seu comprimento (norma = 1), para que avaliações longas e curtas
  sejam comparáveis.

**Resultado** para o exemplo (`python comandos.py explicar "..."` mostra esta tabela):

| termo | contagem (BOW) | IDF | TF-IDF |
|---|---|---|---|
| peg press | 1 | 9,52 | **0,460** |
| panel nao | 1 | 8,05 | 0,389 |
| nao peg | 1 | 7,18 | 0,347 |
| pess nao | 1 | 6,93 | 0,335 |
| press | 1 | 6,57 | 0,318 |
| panel | 1 | 5,91 | 0,285 |
| peg | 1 | 5,67 | 0,274 |
| pess | 1 | 4,49 | 0,217 |
| nao recom | 1 | 4,47 | 0,216 |
| nao | **2** | 1,90 | 0,184 |
| recom | 1 | 2,94 | 0,142 |

"nao" tem a maior contagem, mas aparece em 6.093 das 15.000 avaliações de treino (IDF 1,90) e fica com um dos
menores pesos; o bigrama "peg press" ("pega pressão") aparece em só 2 (IDF 9,52) e fica com o maior.
Cálculo de "peg press": 1 (contagem) × 9,52 (IDF) / 20,7 (norma do vetor) = 0,460.

---

## Etapa 4 — Treinamento (`etapa4_treinamento.py`)

**Por quê:** preparar o modelo uma única vez, antes de classificar qualquer avaliação.

**Como** (`python comandos.py treinar`, ~5 s), com as 15.000 avaliações de treino:
- o **BOW aprende o léxico** (os 114.138 termos);
- o **TF-IDF aprende o IDF** de cada termo;
- o **KNN guarda** os 15.000 vetores TF-IDF com seus rótulos.

```
 dados/treino.csv — 15.000 avaliações rotuladas
 ┌────────────────────────────────────────────────────────────┬──────────┐
 │ "Pessimo, não pega e ligar ele já desliga não recomendo"   │ negativa │
 │ "Chegou rápido, recomendo"                                 │ positiva │
 │ "É mais ou menos. Nada demais. ..."                        │ neutra   │
 │ ... mais 14.997                                            │          │
 └────────────────────────────────────────────────────────────┴──────────┘
                │  pré-processamento (etapa 2)
                ▼
 ① BOW (CountVectorizer)  ──── APRENDE o léxico
      114.138 termos:  "nao", "recom", "pess", "peg", "rapid", "mais ou", ...
                │
                ▼
 ② TF-IDF (TfidfTransformer)  ──── APRENDE o IDF (1 número por termo)
      nao 1,90 · recom 2,94 · rapid 3,52 · pess 4,49 · peg 5,67 · mais ou 7,67
      (comum = IDF baixo, raro = IDF alto)
                │
                ▼
 ③ KNN (KNeighborsClassifier)  ──── só GUARDA os vetores e os rótulos
                             nao   recom  pess   peg   rapid  mais ou  … 114.138 colunas   rótulo
      "Pessimo, não pega…"   0,17  0,13   0,20   0,26  0      0        …                   negativa
      "Chegou rápido…"       0     0,27   0      0     0,33   0        …                   positiva
      "É mais ou menos…"     0,07  0      0      0     0      0,27     …                   neutra
      …  15.000 linhas (uma por avaliação de treino)
                │
                ▼
 dados/modelo.joblib (8 MB) = léxico + IDF + matriz de vetores + rótulos
```

A matriz tem 15.000 × 114.138 ≈ 1,7 bilhão de posições, mas só 423.506 (0,02%) não são zero: cada avaliação usa
poucas dezenas de termos. Por isso ela é guardada de forma compacta (só os valores diferentes de zero) e cabe em 8 MB.

O KNN não aprende pesos. Modelos como a regressão logística aprendem um peso por termo (e.g., um peso negativo para
"pess") e descartam os dados; o KNN é um classificador "preguiçoso": só guarda os exemplos e deixa todo o trabalho para a
classificação. Por isso, quanto mais dados de treino, maior o modelo e mais lenta a classificação.

**Resultado:** o modelo treinado, salvo em `dados/modelo.joblib` (8 MB). A aplicação web só carrega este arquivo;
ela nunca treina.

---

## Etapa 5 — Classificação (`etapa5_classificacao.py`)

**Por quê:** decidir o sentimento de uma avaliação nova a partir das avaliações rotuladas mais parecidas com ela.

**Como:**
1. A avaliação nova passa pelas etapas 2 e 3 (com o léxico e o IDF do treino) e vira um vetor.
2. Calcula a **similaridade de cosseno** entre esse vetor e os 15.000 vetores guardados (1 = iguais, 0 = nenhum
   termo em comum) e seleciona os **25 mais parecidos** (k = 25).
3. **Votação:** cada vizinho vota no seu rótulo, mas os votos não valem o mesmo: **peso = 1 / distância**, com
   distância = 1 − similaridade. Um vizinho muito parecido vale mais que um pouco parecido. Os pesos são somados
   por classe e **vence a classe com a maior soma** — não necessariamente a com mais vizinhos.

```
 Avaliação nova: "A panela não pega pressão, péssima, não recomendo"
                │  etapas 2 e 3, com o léxico e o IDF aprendidos no treino
                ▼
 ① Vetor da avaliação nova (11 termos diferentes de zero)
                 nao    recom  pess   peg    rapid  mais ou  …
                 0,18   0,14   0,22   0,27   0      0        …
                │
                ▼
 ② Similaridade de cosseno com CADA um dos 15.000 vetores guardados na etapa 4
      "Pessimo, não pega…"   (negativa)  →  0,427
      "Chegou rápido…"       (positiva)  →  0,039
      "É mais ou menos…"     (neutra)    →  0,012
      …  15.000 comparações (7.243 dão 0: nenhum termo em comum)
                │
                ▼
 ③ Ordena e fica com os 25 mais parecidos; peso do voto = 1 / (1 − similaridade)
      #   similaridade  peso   rótulo     avaliação de treino
      1   0,427         1,75   negativa   "Pessimo, não pega e ligar ele já desliga não recomendo"
      2   0,340         1,52   negativa   "não chegou produto, pessimo, não recomendo"
      3   0,336         1,51   positiva   "gostei muito, pega pressão rapidinho e de fácil limpeza"
      4   0,270         1,37   negativa   "A panela não ligou. Fiz tudo conforme as instruções…"
      …
      25  0,162         1,19   negativa
                │
                ▼
 ④ Apuração: soma dos pesos por rótulo
      negativa  ██████████████████████████  26,90  (21 votos)  83,8%   ◄ vence
      positiva  ████                         3,91  ( 3 votos)  12,2%
      neutra    █                            1,28  ( 1 voto )   4,0%
```

**Como a similaridade é calculada** (vizinho nº 1): os vetores já têm comprimento 1, então o cosseno é a soma dos
produtos dos termos que as duas avaliações têm em comum; os outros termos são zero em um dos dois e não contam.

| termo em comum | avaliação nova | "Pessimo, não pega…" | produto |
|---|---|---|---|
| nao peg | 0,347 | 0,326 | 0,113 |
| pess nao | 0,335 | 0,314 | 0,105 |
| peg | 0,274 | 0,257 | 0,070 |
| pess | 0,217 | 0,204 | 0,044 |
| nao recom | 0,216 | 0,203 | 0,044 |
| nao | 0,184 | 0,172 | 0,032 |
| recom | 0,142 | 0,134 | 0,019 |
| **soma = similaridade** | | | **0,427** |

O vizinho nº 3 é positivo porque também fala em "pega pressão" — mas é minoria: os votos negativos somam muito mais.

→ **negativa**. Num caso apertado, como "mais ou menos", o resultado é neutra com 56,3% dos pesos contra 43,7% de
negativa (14 votos contra 11).

---

## Etapa 6 — Pontuação e estrelas (`etapa6_pontuacao.py`)

**Por quê:** o enunciado pede uma nota do produto em estrelas a partir das opiniões classificadas.

**Como:** negativa = 1 ponto, neutra = 3, positiva = 5; nota = média dos pontos, arredondada para a meia estrela
mais próxima.

**Resultado (Panela de Pressão, 40 avaliações):** 27 positivas, 5 neutras, 8 negativas →
(27×5 + 5×3 + 8×1) / 40 = 158 / 40 = **3,95 → 4,0 estrelas**. Média real dos clientes: 3,6.

---

## Etapa 7 — Publicação (`etapa7_publicacao.py`, `app.py`)

**Por quê:** o enunciado pede que o sistema publique as opiniões classificadas com a nota.

**Como:** uma página web (Flask) recebe as avaliações e mostra a nota em estrelas, a contagem por classe e cada
opinião com seu sentimento e pontos (`python comandos.py servidor`). `python comandos.py publicar` gera a mesma
página como arquivo HTML.

**Resultado:** a página com o produto, as estrelas e a lista de opiniões coloridas por sentimento.

---

## Avaliação (`python comandos.py avaliar`)

**Por quê:** medir se o modelo acerta em avaliações que ele não viu.

**Como:**
- **Validação cruzada (5 folds):** divide o treino em 5 partes, treina com 4 e testa na outra, 5 vezes.
- **Produto da demonstração:** compara a classe prevista com a indicada pelas estrelas de cada cliente.

**Resultado:**

| | Precisão | Recall | F1 |
|---|---|---|---|
| negativa | 0,76 | 0,81 | 0,79 |
| neutra | 0,62 | 0,43 | 0,51 |
| positiva | 0,66 | 0,82 | 0,73 |
| **acurácia** | | | **69%** |

No produto da demonstração: **29 de 40 acertos**; nota do sistema 4,0 × média real 3,6.

---

## Limitações

- **Neutra é a classe mais difícil:** avaliações de 3 estrelas misturam elogios e críticas.
- **Bag of words ignora o contexto:** "não decepciona" é elogio, mas "não" + palavra negativa parece reclamação.
- **Rótulos ruidosos:** os clientes nem sempre são coerentes ("chegou rápido, mas veio com defeito" com 5 estrelas).
- **Avaliações muito curtas** têm poucos termos para comparar, então os vizinhos são pouco parecidos.
