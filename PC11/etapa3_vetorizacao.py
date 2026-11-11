"""Etapa 3 — Representação bag of words (BOW) com transformação TF-IDF.

São duas operações separadas:
1. BOW (CountVectorizer): monta o léxico e conta quantas vezes cada termo aparece no texto;
2. TF-IDF (TfidfTransformer): multiplica cada contagem pelo IDF do termo (maior para termos
   raros no treino) e normaliza o vetor para comprimento 1.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from etapa2_preprocessamento import preprocessar


def criar_bow() -> CountVectorizer:
    """BOW: aplica o pré-processamento da etapa 2 e conta unigramas e bigramas ("nao recom").
    O léxico (lista de termos) é montado no treino (etapa 4) a partir das avaliações de treino."""
    return CountVectorizer(preprocessor=preprocessar, ngram_range=(1, 2))


def criar_tfidf() -> TfidfTransformer:
    """TF-IDF: pondera as contagens do BOW. TF = contagem do termo no texto; IDF = ln((1 + N) /
    (1 + nº de avaliações com o termo)) + 1, calculado no treino. O vetor final tem comprimento 1,
    para que textos longos e curtos sejam comparáveis."""
    return TfidfTransformer()


def detalhar_vetor(modelo: Pipeline, texto: str) -> pd.DataFrame:
    """Mostra a transformação de um texto, termo a termo: contagem no BOW, IDF aprendido no treino
    e peso TF-IDF final, do maior peso para o menor (comando `explicar`)."""
    bow, tfidf = modelo.named_steps["bow"], modelo.named_steps["tfidf"]
    contagens = bow.transform([texto])
    pesos = tfidf.transform(contagens).toarray()[0]
    indices = contagens.nonzero()[1]
    tabela = pd.DataFrame({
        "termo": bow.get_feature_names_out()[indices],
        "contagem (BOW)": contagens.toarray()[0][indices],
        "IDF": tfidf.idf_[indices].round(2),
        "TF-IDF": pesos[indices].round(3),
    })
    return tabela.sort_values("TF-IDF", ascending=False).reset_index(drop=True)
