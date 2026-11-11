"""Aplicação web (Flask): avaliações coladas -> opiniões classificadas e nota em estrelas."""

from flask import Flask, render_template, request

from etapa1_entrada import ler_texto_colado
from etapa4_treinamento import carregar_modelo
from pipeline import analisar_opinioes

app = Flask(__name__)
modelo = carregar_modelo()  # treinado antes com: python comandos.py treinar


@app.route("/", methods=["GET", "POST"])
def index():
    """GET: mostra o formulário vazio. POST: classifica as avaliações enviadas e mostra o
    resultado abaixo do formulário. O nome do produto é só o título; não afeta a classificação."""
    if request.method == "GET":
        return render_template("index.html", produto="", avaliacoes="", resultado=None, erro=None)

    produto = request.form.get("produto", "").strip()
    avaliacoes = request.form.get("avaliacoes", "")
    resultado = analisar_opinioes(modelo, produto or None, ler_texto_colado(avaliacoes))
    erro = "Informe ao menos uma avaliação." if resultado is None else None
    return render_template("index.html", produto=produto, avaliacoes=avaliacoes, resultado=resultado, erro=erro)
