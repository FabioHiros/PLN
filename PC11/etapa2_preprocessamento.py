"""Etapa 2 — Pré-processamento: normaliza o texto antes da vetorização."""

import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

# A lista de stopwords do NLTK é genérica e remove palavras que carregam sentimento.
# Negações invertem o sentido ("não gostei", "nem bom nem ruim"); "muito" e "mais" dão
# intensidade ("muito ruim", "esperava mais"); "sem" e "mas" mudam a avaliação
# ("sem defeito", "bom, mas demorou"); "ou" preserva a expressão "mais ou menos" (sem ele,
# o bigrama "mais menos" se confunde com "mais com menos"). Por isso essas palavras são mantidas.
PALAVRAS_DE_SENTIMENTO = {"não", "nem", "nunca", "jamais", "nada", "ninguém", "muito", "mais", "sem", "mas", "ou"}
STOPWORDS = set(stopwords.words("portuguese")) - PALAVRAS_DE_SENTIMENTO
PONTUACAO = string.punctuation + "“”‘’…–—"

stemmer = RSLPStemmer()


def converter_minusculas(texto: str) -> str:
    """Iguala "Ótimo" e "ótimo", para que contem como o mesmo termo."""
    return texto.lower()


def remover_urls(texto: str) -> str:
    """Troca links (http://..., https://...) por espaço; não indicam sentimento."""
    return re.sub(r"https?://\S+", " ", texto)


def remover_pontuacao(texto: str) -> str:
    """Troca cada sinal de pontuação por espaço, separando "ótimo!" em "ótimo"."""
    return texto.translate(str.maketrans(PONTUACAO, " " * len(PONTUACAO)))


def remover_numeros(texto: str) -> str:
    """Troca sequências de dígitos por espaço (preços, datas, modelos)."""
    return re.sub(r"\d+", " ", texto)


def tokenizar(texto: str) -> list[str]:
    """Divide o texto em palavras (tokens) pelos espaços."""
    return texto.split()


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Descarta palavras muito comuns e sem sentimento ("o", "de", "que") e letras soltas."""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stemizar(tokens: list[str]) -> list[str]:
    """Reduz cada palavra ao radical com o RSLP, stemmer para o português:
    "recomendo" e "recomendado" viram "recom" e contam como o mesmo termo."""
    return [stemmer.stem(t) for t in tokens]


def remover_acentos(tokens: list[str]) -> list[str]:
    """Remove acentos depois da stemização, unificando "péss" e "pess"
    (muitas avaliações são escritas sem acento)."""
    sem_acento = lambda t: "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return [sem_acento(t) for t in tokens]


def aplicar_etapas(texto: str) -> dict:
    """Aplica as etapas em ordem e devolve o resultado de cada uma,
    para mostrar a transformação passo a passo (comando `explicar`)."""
    etapas = {"original": texto}
    etapas["minúsculas"] = converter_minusculas(texto)
    etapas["sem URLs"] = remover_urls(etapas["minúsculas"])
    etapas["sem pontuação"] = remover_pontuacao(etapas["sem URLs"])
    etapas["sem números"] = remover_numeros(etapas["sem pontuação"])
    etapas["tokens"] = tokenizar(etapas["sem números"])
    etapas["sem stopwords"] = remover_stopwords(etapas["tokens"])
    etapas["stemização"] = stemizar(etapas["sem stopwords"])
    etapas["sem acentos"] = remover_acentos(etapas["stemização"])
    return etapas


def preprocessar(texto: str) -> str:
    """Pipeline completa; devolve os tokens normalizados unidos por espaço.
    É chamada pelo TF-IDF (etapa 3) para cada avaliação, no treino e na classificação."""
    return " ".join(aplicar_etapas(texto)["sem acentos"])
