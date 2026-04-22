import json
import random
import re
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


class Config:
    ARQUIVO_INTENCOES = "intencoes.jsonc"
    ARQUIVO_MEMORIA   = "memoria.json"
    ARQUIVO_MODELO    = "modelo_cache.pkl"


# ---------------------------------------------------------------------------
# MEMÓRIA
# ---------------------------------------------------------------------------

class Memoria:
    def __init__(self, caminho):
        self.caminho = caminho
        self.dados   = self._carregar()

    def _carregar(self):
        base = {
            "topico_atual":        None,
            "estado":              "inicio",
            "aguardando":          [],       # tags esperadas na próxima mensagem
            "intencao_ativa":      None,     # intenção que está "segurando" o contexto
            "voltas_no_topico":    0,        # quantas vezes ficou no mesmo tópico sem avançar
            "msgs_sem_encaixe":    0,        # mensagens que não encaixaram nas esperadas
            "historico_tags":      [],       # últimas 10 tags disparadas
        }
        if not os.path.exists(self.caminho):
            return base
        with open(self.caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        for k, v in base.items():
            dados.setdefault(k, v)
        return dados

    def salvar(self):
        with open(self.caminho, "w", encoding="utf-8") as f:
            json.dump(self.dados, f, indent=4, ensure_ascii=False)

    def set_estado(self, estado: str):
        self.dados["estado"] = estado
        self.salvar()

    def atualizar_apos_resposta(self, intencao: dict, veio_de_fallback: bool = False):
        """
        Atualiza o estado da memória após uma intenção ser escolhida.
        veio_de_fallback=True indica que a resposta veio do fallback_resposta,
        ou seja, o bot não avançou no fluxo — só repetiu o contexto.
        """
        novo_topico = intencao.get("topico")

        if veio_de_fallback:
            # não avançou: incrementa contadores de loop
            self.dados["msgs_sem_encaixe"]  += 1
            self.dados["voltas_no_topico"]  += 1
        else:
            # avançou normalmente: reseta contadores e atualiza tudo
            self.dados["msgs_sem_encaixe"]   = 0
            self.dados["voltas_no_topico"]   = 0 if novo_topico != self.dados["topico_atual"] else self.dados["voltas_no_topico"]
            self.dados["topico_atual"]       = novo_topico
            self.dados["aguardando"]         = intencao.get("proximo_esperado", [])
            self.dados["intencao_ativa"]     = intencao.get("tag")

        # histórico sempre cresce
        hist = self.dados["historico_tags"]
        hist.append(intencao["tag"])
        self.dados["historico_tags"] = hist[-10:]

        if intencao["tag"] in ("despedida", "boa_noite_final"):
            self.dados["estado"] = "fim"

        self.salvar()

    def expirou(self, intencao_ativa: dict | None) -> bool:
        """Verifica se o contexto atual expirou pelo limite de msgs_sem_encaixe."""
        if not intencao_ativa:
            return False
        limite = intencao_ativa.get("expira_em", 3)
        return self.dados["msgs_sem_encaixe"] >= limite

    def loop_estourou(self, intencao_ativa: dict | None) -> bool:
        """Verifica se o bot ficou em loop no mesmo tópico além do max_voltas."""
        if not intencao_ativa:
            return False
        limite = intencao_ativa.get("max_voltas", 3)
        return self.dados["voltas_no_topico"] >= limite

    def resetar_contexto(self):
        """Limpa o contexto quando expira ou estoura o loop."""
        self.dados["aguardando"]        = []
        self.dados["intencao_ativa"]    = None
        self.dados["msgs_sem_encaixe"]  = 0
        self.dados["voltas_no_topico"]  = 0
        self.dados["topico_atual"]      = None
        self.salvar()


# ---------------------------------------------------------------------------
# RESULTADO DA BUSCA — carrega informação extra para o ChatBot
# ---------------------------------------------------------------------------

class ResultadoBusca:
    def __init__(self, intencao: dict, veio_de_fallback: bool = False, resposta_override: str | None = None):
        self.intencao         = intencao
        self.veio_de_fallback = veio_de_fallback
        self.resposta_override = resposta_override  # fallback_resposta personalizada

    def escolher_resposta(self) -> str:
        if self.resposta_override:
            return self.resposta_override
        return random.choice(self.intencao.get("respostas", ["..."]))


# ---------------------------------------------------------------------------
# CÉREBRO
# ---------------------------------------------------------------------------

class Cerebro:
    BONUS_AGUARDANDO   = 0.6
    BONUS_TOPICO       = 0.3
    BONUS_KEYWORD      = 0.5
    THRESHOLD_SIM      = 0.3   # com contexto
    THRESHOLD_SIM_FRIO = 0.5   # sem contexto
    THRESHOLD_ML       = 0.55

    def __init__(self, caminho_json: str):
        self.caminho = caminho_json
        self.base    = self._carregar_json(caminho_json)
        self.modelo  = self._carregar_ou_treinar()
        # índice tag → intenção para lookup O(1)
        self._idx    = {i["tag"]: i for i in self.base["intencoes"]}

    # ── carregamento ──────────────────────────────────────────────────────

    def _carregar_json(self, caminho: str) -> dict:
        with open(caminho, "r", encoding="utf-8") as f:
            texto = re.sub(r"//.*", "", f.read())
            return json.loads(texto)

    def _carregar_ou_treinar(self):
        if os.path.exists(Config.ARQUIVO_MODELO):
            cache = joblib.load(Config.ARQUIVO_MODELO)
            if cache.get("mtime") == os.path.getmtime(self.caminho):
                return cache["modelo"]
        modelo = self._treinar()
        joblib.dump({"modelo": modelo, "mtime": os.path.getmtime(self.caminho)}, Config.ARQUIVO_MODELO)
        return modelo

    def _treinar(self):
        frases, tags = [], []
        for i in self.base["intencoes"]:
            for p in i.get("perguntas", []):
                frases.append(p)
                tags.append(i["tag"])
        m = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))), ("clf", MultinomialNB())])
        m.fit(frases, tags)
        return m

    # ── utilidades ────────────────────────────────────────────────────────

    def _limpar(self, texto: str) -> str:
        return re.sub(r"[^\w\s]", "", texto.lower()).strip()

    def _similaridade(self, a: str, b: str) -> float:
        s1 = set(self._limpar(a).split())
        s2 = set(self._limpar(b).split())
        return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0

    def _score_base(self, msg_limpa: str, item: dict) -> float:
        """Score puro de similaridade + keyword, sem bônus de contexto."""
        perguntas = item.get("perguntas", [])
        score = max((self._similaridade(msg_limpa, p) for p in perguntas), default=0)
        if any(k in msg_limpa for k in item.get("keywords", [])):
            score += self.BONUS_KEYWORD
        return score

    def _melhor_entre(self, msg_limpa: str, itens: list[dict], threshold: float) -> dict | None:
        """Retorna a melhor intenção de uma lista se passar do threshold, senão None."""
        melhor, maior = None, 0
        for item in itens:
            s = self._score_base(msg_limpa, item)
            if s > maior:
                maior, melhor = s, item
        return melhor if melhor and maior >= threshold else None

    # ── 4 camadas de busca ────────────────────────────────────────────────

    def _camada_esperadas(self, msg_limpa: str, memoria: Memoria) -> dict | None:
        """Camada 1: busca só entre as tags que o bot está aguardando."""
        aguardando = memoria.dados.get("aguardando", [])
        if not aguardando:
            return None
        itens = [self._idx[t] for t in aguardando if t in self._idx]
        return self._melhor_entre(msg_limpa, itens, self.THRESHOLD_SIM)

    def _camada_topico(self, msg_limpa: str, memoria: Memoria) -> dict | None:
        """Camada 2: busca qualquer intenção do mesmo tópico ativo."""
        topico = memoria.dados.get("topico_atual")
        if not topico:
            return None
        itens = [i for i in self.base["intencoes"] if i.get("topico") == topico]
        return self._melhor_entre(msg_limpa, itens, self.THRESHOLD_SIM)

    def _camada_livre(self, msg_limpa: str) -> dict | None:
        """Camada 3: busca livre em todo o JSON."""
        return self._melhor_entre(msg_limpa, self.base["intencoes"], self.THRESHOLD_SIM_FRIO)

    def _camada_ml(self, msg_limpa: str) -> dict | None:
        """Camada 3b: fallback com Naive Bayes se a busca livre falhou."""
        tag  = self.modelo.predict([msg_limpa])[0]
        prob = max(self.modelo.predict_proba([msg_limpa])[0])
        if prob > self.THRESHOLD_ML:
            return self._idx.get(tag)
        return None

    def _nao_entendido(self) -> dict:
        return self._idx.get("nao_entendido", {"tag": "nao_entendido", "respostas": ["Não entendi, pode repetir?"]})

    # ── entrada principal ─────────────────────────────────────────────────

    def buscar_melhor_resposta(self, msg: str, memoria: Memoria) -> ResultadoBusca:
        msg_limpa       = self._limpar(msg)
        intencao_ativa  = self._idx.get(memoria.dados.get("intencao_ativa", ""))

        # ── verifica se o contexto expirou ou entrou em loop ──
        if memoria.expirou(intencao_ativa) or memoria.loop_estourou(intencao_ativa):
            memoria.resetar_contexto()
            intencao_ativa = None

        # ── camada 1: tags esperadas ──
        resultado = self._camada_esperadas(msg_limpa, memoria)
        if resultado:
            return ResultadoBusca(resultado)

        # ── camada 2: mesmo tópico (fallback_topico) ──
        resultado = self._camada_topico(msg_limpa, memoria)
        if resultado:
            return ResultadoBusca(resultado)

        # ── se tinha contexto mas não encontrou nada: usa fallback_resposta ──
        if intencao_ativa and intencao_ativa.get("fallback_resposta"):
            resposta = random.choice(intencao_ativa["fallback_resposta"])
            return ResultadoBusca(intencao_ativa, veio_de_fallback=True, resposta_override=resposta)

        # ── camada 3: busca livre ──
        resultado = self._camada_livre(msg_limpa) or self._camada_ml(msg_limpa)
        if resultado:
            return ResultadoBusca(resultado)

        # ── camada 4: não entendeu ──
        return ResultadoBusca(self._nao_entendido())


# ---------------------------------------------------------------------------
# CHATBOT
# ---------------------------------------------------------------------------

class ChatBot:
    def __init__(self):
        self.cerebro = Cerebro(Config.ARQUIVO_INTENCOES)
        self.memoria = Memoria(Config.ARQUIVO_MEMORIA)

    def escutar(self):
        total = len(self.cerebro.base["intencoes"])
        print(f"🤖 Bot Online | {total} intenções carregadas\n")
        try:
            while True:
                msg = input("Você: ").strip()
                if not msg:
                    continue
                if msg.lower() in ("sair", "exit"):
                    break

                busca    = self.cerebro.buscar_melhor_resposta(msg, self.memoria)
                resposta = busca.escolher_resposta()

                # DEBUG — remova em produção
                print(f"  [tag={busca.intencao['tag']} | fallback={busca.veio_de_fallback}"
                      f" | aguardando={self.memoria.dados.get('aguardando', [])}"
                      f" | voltas={self.memoria.dados.get('voltas_no_topico', 0)}"
                      f" | sem_encaixe={self.memoria.dados.get('msgs_sem_encaixe', 0)}]")

                self.memoria.atualizar_apos_resposta(busca.intencao, busca.veio_de_fallback)
                print(f"Bot: {resposta}\n")

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    ChatBot().escutar()
