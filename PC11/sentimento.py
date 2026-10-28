"""Analisador de sentimentos: pré-processamento, BOW com TF-IDF e KNN."""

import re
import string
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

for recurso in ("stopwords", "rslp"):
    nltk.download(recurso, quiet=True)

CAMINHO_TREINO = Path(__file__).parent / "dados" / "treino.csv"

PONTOS = {"negativa": 1, "neutra": 3, "positiva": 5}

# Negações invertem o sentido da frase ("não gostei", "nem bom nem ruim"),
# por isso são retiradas da lista de stopwords.
NEGACOES = {"não", "nem", "nunca", "jamais", "nada", "ninguém"}
STOPWORDS = set(stopwords.words("portuguese")) - NEGACOES

stemmer = RSLPStemmer()


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def preprocessar(texto: str) -> str:
    """Normaliza o texto: minúsculas, sem pontuação/dígitos, sem stopwords,
    com stemização (RSLP) e sem acentos."""
    texto = texto.lower()
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = texto.translate(str.maketrans(string.punctuation + "“”‘’…–—", " " * (len(string.punctuation) + 7)))
    texto = re.sub(r"\d+", " ", texto)
    tokens = [t for t in texto.split() if t not in STOPWORDS and len(t) > 1]
    tokens = [remover_acentos(stemmer.stem(t)) for t in tokens]
    return " ".join(tokens)


def criar_modelo(k: int = 5) -> Pipeline:
    """BOW (uni + bigramas) com TF-IDF alimentando um KNN por similaridade de cosseno.
    A classe é decidida por votação entre os k vizinhos, ponderada pela distância."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocessar, ngram_range=(1, 2))),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")),
    ])


def carregar_treino() -> pd.DataFrame:
    return pd.read_csv(CAMINHO_TREINO)


def treinar(k: int = 5) -> Pipeline:
    df = carregar_treino()
    modelo = criar_modelo(k)
    modelo.fit(df["texto"], df["sentimento"])
    return modelo


def avaliar(k: int = 5) -> str:
    """Validação cruzada estratificada (5 folds) sobre o corpus de treino."""
    df = carregar_treino()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    previsto = cross_val_predict(criar_modelo(k), df["texto"], df["sentimento"], cv=cv)
    return classification_report(df["sentimento"], previsto, zero_division=0)


def classificar(modelo: Pipeline, opinioes: list[str]) -> dict:
    """Classifica as opiniões e calcula a nota média (1 a 5) e as estrelas."""
    if not opinioes:
        return {"opinioes": [], "media": None, "estrelas": None}
    sentimentos = modelo.predict(opinioes)
    resultado = [
        {"texto": texto, "sentimento": s, "pontos": PONTOS[s]}
        for texto, s in zip(opinioes, sentimentos)
    ]
    media = sum(r["pontos"] for r in resultado) / len(resultado)
    return {"opinioes": resultado, "media": media, "estrelas": estrelas(media)}


def estrelas(media: float) -> float:
    """Arredonda a nota média para a meia estrela mais próxima (e.g., 3.7 -> 3.5)."""
    return round(media * 2) / 2


if __name__ == "__main__":
    print(avaliar())
