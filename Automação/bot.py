import json
import random
import re
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def carregar_json(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
    conteudo = re.sub(r"//.*", "", conteudo)
    return json.loads(conteudo)

def salvar_json(dados, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

def limpar(texto):
    texto = texto.lower()
    texto = re.sub(r"<.*?>", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def palavras_importantes(texto):
    stopwords = {
        "o", "a", "os", "as", "de", "do", "da", "e",
        "é", "to", "ta", "tá", "um", "uma",
        "o", "no", "na", "lá", "li", "meu", "minha",
        "pra", "pro", "que", "como", "isso", "ai"
    }

    palavras = limpar(texto).split()
    return [p for p in palavras if p not in stopwords]

def treinar_modelo(base):
    frases = []
    tags = []

    for item in base.get("intencoes", []):
        if "perguntas" in item:
            for pergunta in item["perguntas"]:
                frases.append(pergunta)
                tags.append(item["tag"])
    
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,2))),
        ("clf", MultinomialNB())
    ])

    modelo.fit(frases, tags)
    return modelo

def similaridade(a, b):
    p1 = set(palavras_importantes(a))
    p2 = set(palavras_importantes(b))

    if not p1 or not p2:
        return 0

    return len(p1.intersection(p2)) / len(p1.union(p2))

        #Função em que faz o meu py guarda memoria ao conversar com alguém

def carregar_memoria(caminho="memoria.json"):
    if not os.path.exists(caminho):
        memoria = {"inicio_conversa":[], "fim_conversa":[], "topico atual": None , "Estado": "inicio"}
        salvar_json(memoria, caminho)
        return memoria
    else:
        memoria = carregar_json(caminho)
        # Garantir que as chaves existam
        if "inicio_conversa" not in memoria:
             memoria["inicio_conversa"] = []
        if "fim_conversa" not in memoria:
             memoria["fim_conversa"] = []
        if "estado" not in memoria:
            memoria["estado"] = "inicio"
        return memoria
def set_estado(memoria, estado):
    memoria["estado"] = estado
    salvar_memoria(memoria)

def get_estado(memoria):
    return memoria.get("estado", "inicio")
def salvar_memoria(memoria, caminho="memoria.json"):
    salvar_json(memoria, caminho)

def adicionar_inicio(memoria, msg):
    memoria["inicio_conversa"].append(msg)
    salvar_memoria(memoria)

def adicionar_fim(memoria, msg):
    memoria["fim_conversa"].append(msg)
    salvar_memoria(memoria)
    
def gerenciar_memoria(memoria, topico = None, acao="get"):
    if acao == "set" and topico:
        memoria["topico atual"] = topico

        return None
    elif acao == "get":
        return memoria.get("topico atual", None)
    else:
        return None
    


def responder(msg, base, memoria, modelo):
    msg = limpar(msg)
    estado = get_estado(memoria)

    melhor = None
    maior_score = 0

    if get_estado(memoria) == "fim":
        # reseta contexto
        gerenciar_memoria(memoria, topico=None, acao="set")
        set_estado(memoria, "inicio")

    msg = limpar(msg)
    topico_atual = gerenciar_memoria(memoria, acao="get")
    
    # Busca combinando similaridade, palavras-chave e contexto
    for item in base.get("intencoes", []):
        score_sim = 0
        for pergunta in item["perguntas"]:
            s = similaridade(msg, pergunta)
            if s > score_sim:
                score_sim = s
        
        score_kw = 0
        if "keywords" in item:
            qt_kw = 0
            for kw in item["keywords"]:
                if re.search(r'\b' + re.escape(kw) + r'\b', msg.lower()):
                    qt_kw += 1
            if qt_kw > 0:
                score_kw = 0.5 + (0.1 * qt_kw) # Dá peso alto se achou kw
                
        # Reforço se pertencer ao mesmo tópico de conversa atual
        score_contexto = 0
        if topico_atual and item.get("topico") == topico_atual:
            score_contexto = 0.3

        score_total = score_sim + score_kw + score_contexto

        if score_total > maior_score:
            maior_score = score_total
            melhor = item

    if melhor and maior_score >= 0.5:
        tag = melhor["tag"]
        estado_atual = get_estado(memoria)

        # 🧠 Controle de estado
        if tag in ["despedida", "boa_noite_final"]:
            set_estado(memoria, "fim")
        elif tag in ["saudacao_inicial", "bom_dia", "boa_tarde", "boa_noite_inicial"]:
            if estado_atual != "meio":
                set_estado(memoria, "inicio")
        else:
            if estado_atual != "inicio":
                set_estado(memoria, "fim")

        gerenciar_memoria(memoria, topico=melhor.get("topico"), acao="set")
        salvar_memoria(memoria)
        return random.choice(melhor["respostas"])


    # --- Machine Learning ---
    tag_ml = modelo.predict([msg])[0]
    probs = modelo.predict_proba([msg])[0]
    confianca = max(probs)

    if confianca > 0.6 and maior_score < 0.5:
        for item in base["intencoes"]:
            if item["tag"] == tag_ml:
                if tag_ml in ["despedida", "boa_noite_final"]:
                    if estado == "inicio":
                        continue
                    set_estado(memoria, "fim")

                elif tag_ml in ["saudacao_inicial", "bom_dia", "boa_tarde", "boa_noite_inicial"]:
                    set_estado(memoria, "meio")

                else:
                    set_estado(memoria, "meio")
                gerenciar_memoria(memoria, topico=item.get("topico"), acao="set")
                salvar_memoria(memoria)
                return random.choice(item["respostas"])

    # Se não houver score decente, cai no não entendido
    nao_entendido = next((item for item in base["intencoes"] if item.get("tag")=="nao_entendido"), None)
    return random.choice(nao_entendido["respostas"])
  
    
# rodar
base = carregar_json("intencoes.jsonc")
memoria = carregar_memoria()
modelo = treinar_modelo(base) # Treina o modelo com a base atual

print("🤖 Bot iniciado (digite sair ou Ctrl+C)\n")

primeira_mensagem = True
ultima_mensagem = None

try:
    while True:
        msg = input("Você: ")

        if msg.lower() == "sair":
            if ultima_mensagem:
                adicionar_fim(memoria, ultima_mensagem)
            break

        if primeira_mensagem:
            adicionar_inicio(memoria, msg)
            primeira_mensagem = False

        resposta = responder(msg, base, memoria, modelo)
        print("Bot:", resposta)
        ultima_mensagem = msg  # Guarda para ser possivelmente a última antes de sair
except KeyboardInterrupt:
    print("\n👋 Bot encerrado. Até logo!")
except Exception as e:
    print(f"\n❌ Ocorreu um erro: {e}")