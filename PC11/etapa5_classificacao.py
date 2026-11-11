"""Etapa 5 — Classificação: votação ponderada entre os k vizinhos mais próximos."""

import pandas as pd
from sklearn.pipeline import Pipeline

from etapa4_treinamento import carregar_dados_treino


def classificar(modelo: Pipeline, textos: list[str]) -> list[str]:
    """Classifica cada texto: pré-processa, vetoriza com o léxico do treino e devolve a classe
    mais votada entre os k vizinhos mais próximos (negativa, neutra ou positiva)."""
    return list(modelo.predict(textos))


def explicar_votacao(modelo: Pipeline, texto: str) -> pd.DataFrame:
    """Os k vizinhos mais próximos do texto no treino, do mais parecido ao menos parecido, com
    o sentimento, a similaridade de cosseno e o peso do voto de cada um (comando `explicar`).
    Peso = 1 / distância, com distância = 1 - similaridade: quanto mais parecido, maior o peso."""
    vetor = modelo[:-1].transform([texto])  # BOW + TF-IDF, sem o KNN
    distancias, indices = modelo.named_steps["knn"].kneighbors(vetor)
    vizinhos = carregar_dados_treino().iloc[indices[0]].copy()
    vizinhos["similaridade"] = (1 - distancias[0]).round(3)
    vizinhos["peso do voto"] = (1 / distancias[0]).round(3)
    return vizinhos.reset_index(drop=True)


def somar_votos(vizinhos: pd.DataFrame) -> pd.DataFrame:
    """Apuração da votação: por sentimento, quantos vizinhos votaram nele e a soma dos pesos
    desses votos. Vence a classe com a maior soma (não necessariamente a com mais votos)."""
    apuracao = vizinhos.groupby("sentimento").agg(votos=("sentimento", "size"), soma_dos_pesos=("peso do voto", "sum"))
    apuracao["% dos pesos"] = (100 * apuracao["soma_dos_pesos"] / apuracao["soma_dos_pesos"].sum()).round(1)
    return apuracao.round(2).sort_values("soma_dos_pesos", ascending=False)
