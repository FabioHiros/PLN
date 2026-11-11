"""Comandos do projeto. Uso: python comandos.py <comando>

  dados                  baixa o B2W-Reviews01 e gera dados/treino.csv e dados/produto_demo.csv
  treinar [--k 25]       treina o modelo e salva em dados/modelo.joblib
  avaliar [--k 25]       relatório de acurácia: validação cruzada + produto da demonstração
  classificar "texto"    classifica uma avaliação com o modelo salvo
  explicar "texto"       mostra cada etapa para uma avaliação: pré-processamento, TF-IDF e votação
  publicar               classifica o produto da demonstração e gera resultado_publicado.html
  servidor               inicia a aplicação web em http://127.0.0.1:5000 (só carrega o modelo)
"""

import argparse

from etapa4_treinamento import K_VIZINHOS


def comando_dados(args):
    """Etapa 0: regenera os arquivos de dados a partir do B2W-Reviews01."""
    from etapa0_dados import CAMINHO_DEMO, CAMINHO_TREINO, preparar_dados
    preparar_dados()
    print(f"Gerados: {CAMINHO_TREINO} e {CAMINHO_DEMO}")


def comando_treinar(args):
    """Treina o modelo com o k escolhido e grava em dados/modelo.joblib."""
    from etapa4_treinamento import salvar_modelo, treinar
    print(f"Treinando com k = {args.k}...")
    print("Modelo salvo em:", salvar_modelo(treinar(args.k)))


def comando_avaliar(args):
    """Validação cruzada com o k escolhido e acertos do modelo salvo no produto da demonstração."""
    from etapa4_treinamento import avaliar_produto_demo, avaliar_validacao_cruzada, carregar_modelo
    print(f"Validação cruzada (5 folds, k = {args.k}):")
    print(avaliar_validacao_cruzada(args.k))
    demo = avaliar_produto_demo(carregar_modelo())
    print("Produto da demonstração (modelo salvo):")
    print(f"  acertos: {demo['acertos']} de {demo['total']}")
    print(f"  nota do sistema: {demo['media_sistema']:.2f} -> {demo['estrelas_sistema']} estrelas"
          f" | média real dos clientes: {demo['media_clientes']:.2f}")


def comando_classificar(args):
    """Imprime o sentimento de uma única avaliação."""
    from etapa4_treinamento import carregar_modelo
    from etapa5_classificacao import classificar
    print(classificar(carregar_modelo(), [args.texto])[0])


def comando_explicar(args):
    """Mostra o caminho de uma avaliação pelas etapas 2, 3, 5 e 6 (o treino, etapa 4, já foi feito)."""
    from etapa2_preprocessamento import aplicar_etapas
    from etapa3_vetorizacao import detalhar_vetor
    from etapa4_treinamento import carregar_modelo
    from etapa5_classificacao import classificar, explicar_votacao, somar_votos
    from etapa6_pontuacao import pontuar
    modelo = carregar_modelo()

    print("== Etapa 2 — Pré-processamento")
    for etapa, valor in aplicar_etapas(args.texto).items():
        print(f"  {etapa:<15} {valor}")

    print("\n== Etapa 3 — BOW e TF-IDF (termos do léxico presentes no texto)")
    print(detalhar_vetor(modelo, args.texto).to_string(index=False))

    vizinhos = explicar_votacao(modelo, args.texto)
    print(f"\n== Etapa 5 — Classificação: votação dos {len(vizinhos)} vizinhos mais próximos")
    print("  similaridade  peso  sentimento  avaliação de treino")
    for _, v in vizinhos.head(args.vizinhos).iterrows():
        print(f"  {v['similaridade']:>11.3f}  {v['peso do voto']:.2f}  {v['sentimento']:<10}  {v['texto'][:80]}")
    print("\n  Apuração (vence a maior soma dos pesos):")
    print("  " + somar_votos(vizinhos).to_string().replace("\n", "\n  "))

    sentimento = classificar(modelo, [args.texto])[0]
    pontos = pontuar(sentimento)
    print(f"\n== Etapa 6 — Pontuação: {sentimento} = {pontos} ponto{'s' if pontos > 1 else ''}")


def comando_publicar(args):
    """Classifica o produto da demonstração e grava a página publicada."""
    from etapa0_dados import CAMINHO_PUBLICADO
    from etapa1_entrada import carregar_produto_demo
    from etapa4_treinamento import carregar_modelo
    from etapa7_publicacao import salvar_pagina
    from pipeline import analisar_opinioes
    produto, textos, _ = carregar_produto_demo()
    resultado = analisar_opinioes(carregar_modelo(), produto, textos)
    print(f"{produto}: {resultado['media']:.2f} -> {resultado['estrelas']} estrelas | {dict(resultado['contagem'])}")
    print("Página publicada em:", salvar_pagina(resultado, CAMINHO_PUBLICADO))


def comando_servidor(args):
    """Inicia a aplicação web; o modelo precisa ter sido treinado antes."""
    from app import app
    app.run()


def criar_parser() -> argparse.ArgumentParser:
    """Define os comandos e seus argumentos; cada um aponta para a função que o executa."""
    parser = argparse.ArgumentParser(description="PC.11 — analisador de sentimentos")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("dados", help="gera os dados a partir do B2W-Reviews01").set_defaults(executar=comando_dados)
    for nome, funcao, ajuda in [("treinar", comando_treinar, "treina e salva o modelo"),
                                ("avaliar", comando_avaliar, "relatório de acurácia")]:
        p = sub.add_parser(nome, help=ajuda)
        p.add_argument("--k", type=int, default=K_VIZINHOS, help="número de vizinhos do KNN")
        p.set_defaults(executar=funcao)
    p = sub.add_parser("classificar", help="classifica uma avaliação")
    p.add_argument("texto")
    p.set_defaults(executar=comando_classificar)
    p = sub.add_parser("explicar", help="mostra cada etapa para uma avaliação")
    p.add_argument("texto")
    p.add_argument("--vizinhos", type=int, default=10, help="quantos vizinhos listar")
    p.set_defaults(executar=comando_explicar)
    sub.add_parser("publicar", help="gera resultado_publicado.html").set_defaults(executar=comando_publicar)
    sub.add_parser("servidor", help="inicia a aplicação web").set_defaults(executar=comando_servidor)
    return parser


if __name__ == "__main__":
    args = criar_parser().parse_args()
    args.executar(args)
