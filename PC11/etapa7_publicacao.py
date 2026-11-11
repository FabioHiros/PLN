"""Etapa 7 — Publicação: reúne as opiniões classificadas e gera a página HTML."""

from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from etapa6_pontuacao import calcular_estrelas, calcular_media, pontuar

PASTA_TEMPLATES = Path(__file__).parent / "templates"
templates = Environment(loader=FileSystemLoader(PASTA_TEMPLATES), autoescape=True)


def montar_resultado(produto: str | None, opinioes: list[str], sentimentos: list[str]) -> dict:
    """Junta cada opinião com seu sentimento e pontos, conta quantas há de cada classe e
    calcula a nota média e as estrelas do produto."""
    classificadas = [
        {"texto": texto, "sentimento": s, "pontos": pontuar(s)}
        for texto, s in zip(opinioes, sentimentos)
    ]
    media = calcular_media([o["pontos"] for o in classificadas])
    return {
        "produto": produto,
        "opinioes": classificadas,
        "contagem": Counter(sentimentos),
        "media": media,
        "estrelas": calcular_estrelas(media),
    }


def gerar_pagina(resultado: dict) -> str:
    """Gera a página HTML com o resultado, usando os mesmos templates da aplicação web."""
    return templates.get_template("publicado.html").render(resultado=resultado)


def salvar_pagina(resultado: dict, caminho: str | Path) -> Path:
    """Grava a página gerada em um arquivo HTML (comando `publicar`)."""
    caminho = Path(caminho)
    caminho.write_text(gerar_pagina(resultado), encoding="utf-8")
    return caminho
