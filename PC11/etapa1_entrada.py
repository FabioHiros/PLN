"""Etapa 1 — Entrada: as avaliações que serão analisadas."""

import pandas as pd

from etapa0_dados import CAMINHO_DEMO


def carregar_produto_demo() -> tuple[str, list[str], list[int]]:
    """Lê dados/produto_demo.csv e devolve o nome do produto, os textos das avaliações e as
    estrelas dos clientes (estas só servem para conferir o resultado, o modelo não as usa)."""
    dados = pd.read_csv(CAMINHO_DEMO)
    return dados["produto"].iloc[0], dados["texto"].tolist(), dados["estrelas"].tolist()


def ler_texto_colado(texto: str) -> list[str]:
    """Converte o texto colado no formulário em uma lista de avaliações:
    uma por linha, ignorando linhas em branco."""
    return [linha.strip() for linha in texto.splitlines() if linha.strip()]
