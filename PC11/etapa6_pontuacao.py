"""Etapa 6 — Pontuação: converte sentimentos em pontos e a média em estrelas."""

import math

PONTOS = {"negativa": 1, "neutra": 3, "positiva": 5}


def pontuar(sentimento: str) -> int:
    """Negativa = 1 ponto, neutra = 3 pontos, positiva = 5 pontos."""
    return PONTOS[sentimento]


def calcular_media(pontos: list[int]) -> float:
    """Média aritmética dos pontos (ou das estrelas) de uma lista de avaliações."""
    return sum(pontos) / len(pontos)


def sentimento_pelas_estrelas(estrelas: int) -> str:
    """Converte as estrelas de um cliente em sentimento (1-2 = negativa, 3 = neutra,
    4-5 = positiva). Usada para rotular o treino e para conferir o modelo."""
    return "negativa" if estrelas <= 2 else "neutra" if estrelas == 3 else "positiva"


def calcular_estrelas(media: float) -> float:
    """Converte a nota média em estrelas, arredondando para a meia estrela mais próxima
    (e.g., 3.7 -> 3.5). Empates sobem (3.25 -> 3.5); o round() do Python arredondaria
    para o par mais próximo (3.25 -> 3.0)."""
    return math.floor(media * 2 + 0.5) / 2
