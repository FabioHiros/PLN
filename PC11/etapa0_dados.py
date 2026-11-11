"""Etapa 0 — Dados: prepara os dados a partir do B2W-Reviews01 (avaliações reais da Americanas).

Gera:
- dados/treino.csv: amostra balanceada de avaliações rotuladas pelas estrelas do cliente;
- dados/produto_demo.csv: as avaliações reais de um produto (fora do treino), usadas
  como entrada do analisador.

Fonte: B2W Digital, B2W-Reviews01, https://github.com/americanas-tech/b2w-reviews01
Licença: CC BY-NC-SA 4.0. Uso: python comandos.py dados
"""

from pathlib import Path

import pandas as pd
import requests

from etapa6_pontuacao import sentimento_pelas_estrelas

PASTA = Path(__file__).parent
URL_B2W = "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv"
CAMINHO_B2W = PASTA / "dados" / "b2w" / "B2W-Reviews01.csv"
CAMINHO_TREINO = PASTA / "dados" / "treino.csv"
CAMINHO_DEMO = PASTA / "dados" / "produto_demo.csv"
CAMINHO_MODELO = PASTA / "dados" / "modelo.joblib"
CAMINHO_PUBLICADO = PASTA / "resultado_publicado.html"

PRODUTO_DEMO = "7261312"  # Panela de Pressão Rochedo: 40 avaliações, média 3,6
AVALIACOES_POR_CLASSE = 5000


def baixar_b2w() -> Path:
    """Baixa o CSV do GitHub (~50 MB) para dados/b2w/, só se ainda não estiver lá."""
    if not CAMINHO_B2W.exists():
        CAMINHO_B2W.parent.mkdir(parents=True, exist_ok=True)
        CAMINHO_B2W.write_bytes(requests.get(URL_B2W, timeout=120).content)
    return CAMINHO_B2W


def carregar_b2w() -> pd.DataFrame:
    """Lê o CSV com as colunas usadas (produto, nota de 1 a 5, texto), renomeadas para
    português, e descarta as avaliações sem texto."""
    colunas = {"product_id": "id_produto", "product_name": "produto", "overall_rating": "estrelas", "review_text": "texto"}
    dados = pd.read_csv(baixar_b2w(), usecols=list(colunas), dtype=str, low_memory=False)
    dados = dados.rename(columns=colunas).dropna(subset=["texto", "estrelas"])
    dados["estrelas"] = dados["estrelas"].astype(int)
    return dados


def rotular(dados: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna `sentimento` a partir das estrelas do cliente:
    1-2 estrelas = negativa, 3 = neutra, 4-5 = positiva."""
    return dados.assign(sentimento=dados["estrelas"].map(sentimento_pelas_estrelas))


def amostrar_treino(dados: pd.DataFrame) -> pd.DataFrame:
    """Sorteia 5.000 avaliações de cada classe (balanceado, para o modelo não favorecer a
    classe mais comum) e exclui o produto da demonstração, para testá-lo com dados inéditos."""
    sem_demo = dados[dados["id_produto"] != PRODUTO_DEMO]
    amostra = sem_demo.groupby("sentimento").sample(AVALIACOES_POR_CLASSE, random_state=42)
    return amostra[["texto", "sentimento"]].sample(frac=1, random_state=42)


def separar_produto_demo(dados: pd.DataFrame) -> pd.DataFrame:
    """Separa as avaliações do produto da demonstração (nome, texto e estrelas do cliente)."""
    return dados.loc[dados["id_produto"] == PRODUTO_DEMO, ["produto", "texto", "estrelas"]]


def preparar_dados():
    """Executa a etapa 0 inteira: baixa, rotula e grava dados/treino.csv e dados/produto_demo.csv."""
    dados = rotular(carregar_b2w())
    amostrar_treino(dados).to_csv(CAMINHO_TREINO, index=False)
    separar_produto_demo(dados).to_csv(CAMINHO_DEMO, index=False)
