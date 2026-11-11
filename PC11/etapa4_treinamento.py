"""Etapa 4 — Treinamento do KNeighborsClassifier, gravação do modelo e avaliação."""

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from etapa0_dados import CAMINHO_MODELO, CAMINHO_TREINO
from etapa1_entrada import carregar_produto_demo
from etapa3_vetorizacao import criar_bow, criar_tfidf
from etapa6_pontuacao import calcular_estrelas, calcular_media, pontuar, sentimento_pelas_estrelas

K_VIZINHOS = 25


def carregar_dados_treino() -> pd.DataFrame:
    """Lê as 15.000 avaliações rotuladas de dados/treino.csv (colunas: texto, sentimento)."""
    return pd.read_csv(CAMINHO_TREINO)


def criar_modelo(k: int = K_VIZINHOS) -> Pipeline:
    """Monta o modelo, ainda sem treino: BOW e TF-IDF (etapa 3) seguidos do KNN. O KNN mede a
    distância de cosseno entre vetores, e o voto de cada vizinho pesa mais quanto mais próximo ele estiver."""
    return Pipeline([
        ("bow", criar_bow()),
        ("tfidf", criar_tfidf()),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")),
    ])


def treinar(k: int = K_VIZINHOS) -> Pipeline:
    """Treina o modelo com as avaliações de treino: o BOW aprende o léxico, o TF-IDF aprende o IDF
    de cada termo e o KNN guarda os vetores e rótulos. O KNN não aprende pesos: a comparação com os
    vetores guardados acontece só na classificação."""
    dados = carregar_dados_treino()
    return criar_modelo(k).fit(dados["texto"], dados["sentimento"])


def salvar_modelo(modelo: Pipeline):
    """Grava o modelo treinado (léxico, IDF e vetores do KNN) em dados/modelo.joblib."""
    joblib.dump(modelo, CAMINHO_MODELO)
    return CAMINHO_MODELO


def carregar_modelo() -> Pipeline:
    """Lê dados/modelo.joblib, gravado pelo comando `treinar`. Não treina sozinho: se o
    arquivo não existir, avisa qual comando executar."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"{CAMINHO_MODELO} não existe. Execute: python comandos.py treinar")
    return joblib.load(CAMINHO_MODELO)


def avaliar_validacao_cruzada(k: int = K_VIZINHOS) -> str:
    """Divide o treino em 5 partes; treina com 4 e testa na restante, 5 vezes, para que cada
    avaliação seja prevista por um modelo que não a viu. Devolve precisão, recall e F1 por classe."""
    dados = carregar_dados_treino()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    previsto = cross_val_predict(criar_modelo(k), dados["texto"], dados["sentimento"], cv=folds)
    return classification_report(dados["sentimento"], previsto, zero_division=0)


def avaliar_produto_demo(modelo: Pipeline) -> dict:
    """Classifica as avaliações do produto da demonstração (fora do treino) e compara com o
    sentimento indicado pelas estrelas dos clientes: acertos, nota do sistema e média real."""
    _, textos, estrelas = carregar_produto_demo()
    previstos = modelo.predict(textos)
    reais = [sentimento_pelas_estrelas(e) for e in estrelas]
    media = calcular_media([pontuar(p) for p in previstos])
    return {
        "acertos": sum(p == r for p, r in zip(previstos, reais)),
        "total": len(textos),
        "media_sistema": media,
        "estrelas_sistema": calcular_estrelas(media),
        "media_clientes": calcular_media(estrelas),
    }
