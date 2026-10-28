# PC.11 — Analisador de sentimentos de avaliações de produtos

Aplicação web (Flask) que faz a raspagem das avaliações de uma página de produto, classifica cada
opinião como **negativa** (1 ponto), **neutra** (3 pontos) ou **positiva** (5 pontos) e publica as
opiniões classificadas junto com a nota média em estrelas.

## Pipeline

1. **Raspagem** (`raspagem.py`): `requests` + `BeautifulSoup`. Por padrão procura os seletores
   `.review-text`, `[itemprop=reviewBody]`, `.review-body`, `.comment-text` e `.review p`. Para outras
   páginas, informe o seletor CSS das avaliações no formulário.
2. **Pré-processamento** (`sentimento.preprocessar`): minúsculas, remoção de URLs, pontuação e dígitos,
   remoção de stopwords (NLTK, **mantendo negações** como "não" e "nem"), stemização RSLP e remoção de acentos.
3. **Representação**: bag of words com unigramas e bigramas e transformação TF-IDF (`TfidfVectorizer`).
   Os bigramas capturam construções como "não recomendo" e "nao gost".
4. **Classificação**: `KNeighborsClassifier` (k = 5, distância de cosseno). A classe é decidida por
   votação entre os 5 vizinhos mais próximos, ponderada pela distância.
5. **Nota**: média dos pontos (1/3/5), arredondada para a meia estrela mais próxima.

O corpus de treino está em `dados/treino.csv` (147 avaliações rotuladas, 49 por classe).
Validação cruzada estratificada (5 folds): **acurácia de 87%** (`python sentimento.py` imprime o relatório).

## Execução

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Abra <http://127.0.0.1:5000> e clique em **Raspar e analisar**. A URL padrão aponta para a página de
demonstração `static/avaliacoes_demo.html`, servida pela própria aplicação. Essa página também pode ser
publicada no GitHub Pages e raspada a partir da URL pública.

Os recursos `stopwords` e `rslp` do NLTK são baixados automaticamente na primeira execução.
