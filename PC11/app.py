"""PC.11 — Analisador de sentimentos de avaliações de produtos (Flask)."""

from collections import Counter

import requests
from flask import Flask, render_template, request, url_for

import raspagem
import sentimento

app = Flask(__name__)
modelo = sentimento.treinar()


@app.route("/", methods=["GET", "POST"])
def index():
    url_demo = url_for("static", filename="avaliacoes_demo.html", _external=True)
    url = request.form.get("url", "").strip() or url_demo
    seletor = request.form.get("seletor", "").strip() or None
    contexto = {"url": url, "seletor": seletor or "", "resultado": None, "erro": None}

    if request.method == "POST":
        try:
            produto, opinioes = raspagem.raspar(url, seletor)
        except requests.RequestException as e:
            contexto["erro"] = f"Não foi possível acessar a página: {e}"
            return render_template("index.html", **contexto)

        if not opinioes:
            contexto["erro"] = "Nenhuma avaliação encontrada. Informe o seletor CSS das avaliações."
            return render_template("index.html", **contexto)

        resultado = sentimento.classificar(modelo, opinioes)
        resultado["produto"] = produto
        resultado["contagem"] = Counter(o["sentimento"] for o in resultado["opinioes"])
        contexto["resultado"] = resultado

    return render_template("index.html", **contexto)


if __name__ == "__main__":
    app.run(debug=True)
