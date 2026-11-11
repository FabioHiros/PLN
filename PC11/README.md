# PC.11 — Analisador de sentimentos de avaliações de produtos

O sistema classifica cada avaliação de um produto como **negativa** (1 ponto), **neutra** (3 pontos) ou
**positiva** (5 pontos) e publica as opiniões classificadas com a nota média em estrelas. Usa bag of words
com TF-IDF e `KNeighborsClassifier` (scikit-learn).

Para a apresentação: [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md) — um resumo curto do sistema e, depois, cada etapa
em detalhe (por quê, como e resultado), com diagramas do treinamento e da classificação.

## Onde está cada etapa

| Etapa | Arquivo | O que faz |
|---|---|---|
| 0. Dados | `etapa0_dados.py` | Baixa o B2W-Reviews01, monta o treino balanceado e separa o produto da demonstração |
| 1. Entrada | `etapa1_entrada.py` | Lê as avaliações do produto da demonstração ou as coladas pelo usuário |
| 2. Pré-processamento | `etapa2_preprocessamento.py` | Minúsculas, remoção de URLs/pontuação/números, tokenização, stopwords (mantendo negações e "muito", "mais", "sem", "mas", "ou"), stemização RSLP, remoção de acentos |
| 3. BOW + TF-IDF | `etapa3_vetorizacao.py` | BOW (`CountVectorizer`, unigramas e bigramas) seguido de TF-IDF (`TfidfTransformer`) |
| 4. Treinamento | `etapa4_treinamento.py` | Treina (léxico, IDF e vetores do KNN, k = 25, cosseno), salva/carrega o modelo e avalia por validação cruzada |
| 5. Classificação | `etapa5_classificacao.py` | Os 25 vizinhos mais próximos votam; cada voto pesa 1 / distância e vence a maior soma |
| 6. Pontuação | `etapa6_pontuacao.py` | Pontos por sentimento, média e estrelas (meia estrela mais próxima); estrelas do cliente → sentimento, para comparação |
| 7. Publicação | `etapa7_publicacao.py` + `templates/` | Monta o resultado e gera a página HTML |
| Fluxo completo | `pipeline.py` | Junta as etapas 5 → 7 (`analisar_opinioes`) |
| Aplicação web | `app.py` | Formulário Flask: cola as avaliações e vê o resultado publicado (só carrega o modelo salvo) |
| Comandos | `comandos.py` | Ponto de entrada: `dados`, `treinar`, `avaliar`, `classificar`, `explicar`, `publicar`, `servidor` |

## Dados

**B2W-Reviews01** (B2W Digital, CC BY-NC-SA 4.0, <https://github.com/americanas-tech/b2w-reviews01>): cerca de
130 mil avaliações reais de produtos da Americanas, cada uma com a nota de 1 a 5 estrelas do cliente. A nota
define o rótulo: 1–2 estrelas = negativa, 3 = neutra, 4–5 = positiva.

- `dados/treino.csv` — amostra balanceada de 15.000 avaliações (5.000 por classe). Validação cruzada (5 folds):
  **acurácia de 69%** (F1: negativa 0,79 · neutra 0,51 · positiva 0,73).
- `dados/produto_demo.csv` — as 40 avaliações reais da Panela de Pressão Rochedo (produto fora do treino;
  média real 3,6). O sistema acerta 29 de 40 e dá 4,0 estrelas.
- `dados/modelo.joblib` — o modelo treinado (~8 MB), gerado pelo comando `treinar`: léxico do BOW, IDF do TF-IDF
  e os 15.000 vetores com seus rótulos (sem os textos). A aplicação web apenas carrega este arquivo (0,1 s); ela
  nunca treina sozinha. O arquivo depende da versão do scikit-learn fixada em `requirements.txt`.
- `treino.csv` e `produto_demo.csv` são gerados pelo comando `dados`. O CSV original do B2W (~50 MB) fica em
  `dados/b2w/` e não é versionado.

**Quem baixar o projeto não precisa do B2W:** os arquivos acima já estão no repositório. O CSV original só é
usado pelo comando `dados` (que o baixa automaticamente). O que cada comando usa:

| Comando | Arquivos necessários |
|---|---|
| `servidor`, `classificar` | `modelo.joblib` |
| `explicar` | `modelo.joblib` + `treino.csv` (o modelo não guarda os textos; o CSV mostra o que cada vizinho escreveu) |
| `publicar` | `modelo.joblib` + `produto_demo.csv` |
| `treinar`, `avaliar` | `treino.csv` (+ `produto_demo.csv` e `modelo.joblib` no `avaliar`) |
| `dados` | B2W-Reviews01 (baixado automaticamente) |

## Como executar

Requisitos: Python 3.10 ou mais recente e internet na primeira execução (o NLTK baixa as stopwords e o
stemmer RSLP automaticamente).

### 1. Instalar

Todos os comandos são executados **dentro da pasta `PC11`** (os arquivos importam uns aos outros).

```bash
cd PC11
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Treinar o modelo

Os dados (`dados/treino.csv` e `dados/produto_demo.csv`) e o modelo treinado (`dados/modelo.joblib`) já estão no
projeto, então este passo só é necessário se o modelo não existir ou se os dados forem regenerados.

```bash
python comandos.py dados      # opcional: baixa o B2W-Reviews01 (~50 MB) e regenera os dados
python comandos.py treinar    # treina (~5 s) e salva dados/modelo.joblib
```

### 3. Usar

```bash
python comandos.py servidor   # aplicação web em http://127.0.0.1:5000 (Ctrl+C para parar)
```

Cole as avaliações (uma por linha) e clique em **Analisar** para ver as opiniões classificadas e a nota em
estrelas. O nome do produto é opcional e só aparece como título do resultado; não influencia a classificação.
Para ver o produto da demonstração sem digitar nada, use `python comandos.py publicar`.

## Comandos

Tudo é feito pelo `comandos.py` (`python comandos.py --help` lista os comandos). Nenhum comando treina ou avalia
sozinho; o servidor apenas carrega o modelo salvo e, se ele não existir, avisa para executar `treinar`.

| Comando | O que faz |
|---|---|
| `python comandos.py dados` | Baixa o B2W-Reviews01 e gera `dados/treino.csv` e `dados/produto_demo.csv` |
| `python comandos.py treinar [--k 25]` | Treina o modelo e salva em `dados/modelo.joblib` |
| `python comandos.py avaliar [--k 25]` | Relatório de acurácia: validação cruzada (~25 s) e acertos do modelo salvo no produto da demonstração |
| `python comandos.py classificar "texto"` | Classifica uma avaliação com o modelo salvo |
| `python comandos.py explicar "texto" [--vizinhos 10]` | Mostra cada etapa para uma avaliação: pré-processamento, tabela BOW → IDF → TF-IDF, os vizinhos mais próximos com o peso de cada voto, a apuração e os pontos |
| `python comandos.py publicar` | Classifica o produto da demonstração e gera `resultado_publicado.html` |
| `python comandos.py servidor` | Inicia a aplicação web em <http://127.0.0.1:5000> |

Exemplos:

```bash
python comandos.py classificar "É bom, mas o acabamento deixa a desejar"   # neutra
python comandos.py explicar "muito ruim, não recomendo" --vizinhos 5
python comandos.py treinar --k 15 && python comandos.py avaliar --k 15   # testar outro número de vizinhos
```

### Roteiro para a apresentação

1. `python comandos.py avaliar` — acurácia do modelo (69%) e comparação com as estrelas reais (29 de 40).
2. `python comandos.py explicar "..."` — o caminho de uma avaliação por todas as etapas.
3. `python comandos.py servidor` — o sistema funcionando: opiniões classificadas e nota em estrelas.

## Limitações

- A classe neutra é a mais difícil: avaliações de 3 estrelas misturam elogios e críticas.
- Bag of words ignora o contexto ("não decepciona" é elogio, mas parece reclamação).
- Os rótulos vêm das estrelas, e os clientes nem sempre são coerentes: "Chegou rápido, mas veio com defeito" é
  classificada como positiva porque as avaliações mais parecidas no treino ("O aparelho chegou rápido! Mas veio
  com defeito...") receberam 4 ou 5 estrelas (veja com `python comandos.py explicar`).
- O modelo foi treinado com avaliações de produtos; textos de outros domínios tendem a ser classificados com
  menos precisão.
