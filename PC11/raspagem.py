"""Raspagem das avaliações de produtos de uma página web."""

import requests
from bs4 import BeautifulSoup

# Seletores comuns de avaliações; o primeiro que encontrar elementos é usado.
SELETORES_PADRAO = [
    ".review-text",
    "[itemprop=reviewBody]",
    ".review-body",
    ".comment-text",
    ".review p",
]


def obter_html(url: str, timeout: int = 10) -> str:
    resposta = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (PLN PC.11)"})
    resposta.raise_for_status()
    resposta.encoding = resposta.apparent_encoding
    return resposta.text


def extrair_avaliacoes(html: str, seletor: str | None = None) -> tuple[str | None, list[str]]:
    """Retorna o nome do produto (quando houver) e a lista de textos das avaliações."""
    soup = BeautifulSoup(html, "html.parser")
    seletores = [seletor] if seletor else SELETORES_PADRAO
    avaliacoes = []
    for s in seletores:
        avaliacoes = [e.get_text(" ", strip=True) for e in soup.select(s)]
        avaliacoes = [a for a in avaliacoes if a]
        if avaliacoes:
            break

    titulo = soup.select_one("[itemprop=name], .product-name, h1")
    return (titulo.get_text(strip=True) if titulo else None), avaliacoes


def raspar(url: str, seletor: str | None = None) -> tuple[str | None, list[str]]:
    return extrair_avaliacoes(obter_html(url), seletor)
