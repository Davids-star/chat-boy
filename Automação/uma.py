import json
import re

def carregar_json():
    with open("base.json", "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)

def salvar_json(data, nome="intencoes.json"):
    with open(nome, "w", encoding="utf-8") as arquivo:
        json.dump(data, arquivo, indent=4, ensure_ascii=False)

def limpar_texto(msg):
   
    lixo = ["kkk", "kkkk", "ok", "sim", "não", "ata", "aham"]
    msg =msg.strip()

    if len(msg) < 4:
        return None
    if msg.lower() in lixo:
        return None
    
    return msg

def chave_simples(texto):

    palavras = texto.lower().split()
    return " ".join(palavras[:2])

def similiaridade(a, b):
    palavras_a = set(a.lower().split())
    palavras_b = set(b.lower().split())

    if not palavras_a or not palavras_b:
        return 0
    return len(palavras_a.intersection(palavras_b)) / len(palavras_a.union(palavras_b))

def converter_para_intencao():
    with open("conversa.txt", "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

        padrao = r"\d{2}/\d{2}/\d{4},? \d{2}:\d{2} - (.*)"

        mensagem= []

        for linha in linhas:
            match = re.search(padrao, linha)
            if match:
                msg = match.group(1)
                mensagem.append(msg)
   
    pares =[]

    for i in range(0, len(mensagem)-1, 2):
        pergunta = limpar_texto(mensagem[i])
        resposta = limpar_texto(mensagem[i+1])

        if not pergunta or not resposta:
            continue   
        pares.append({
            "perguntas": pergunta,
            "resposta": resposta    
        })
    return pares


     
def agrupar_respostas(base):
    grupo = []

    for item in base:
        perguntas = item["perguntas"]
        resposta = item["resposta"]
        encontrado = False

        for g in grupo:
            score = similiaridade(resposta, g["resposta"][0])

            if score > 0.5:
                g["perguntas"].append(perguntas)
                g["resposta"].append(resposta)
                encontrado = True
                break
        if not encontrado:
            grupo.append({
               
                "perguntas": [perguntas],
                "resposta": [resposta]
            })
    return grupo

pares = converter_para_intencao()
intencoes = agrupar_respostas(pares)
with open("intencoes.json", "w", encoding="utf-8") as arquivo:
    json.dump(intencoes, arquivo, indent=4, ensure_ascii=False)
