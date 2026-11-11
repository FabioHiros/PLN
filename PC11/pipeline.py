"""Fluxo completo: classificação (etapa 5) -> pontuação (6) -> resultado publicado (7)."""

from sklearn.pipeline import Pipeline

from etapa5_classificacao import classificar
from etapa7_publicacao import montar_resultado


def analisar_opinioes(modelo: Pipeline, produto: str | None, opinioes: list[str]) -> dict | None:
    """Fluxo usado pela aplicação web: classifica as opiniões e monta o resultado com pontos,
    média e estrelas. Devolve None se a lista estiver vazia."""
    if not opinioes:
        return None
    return montar_resultado(produto, opinioes, classificar(modelo, opinioes))
