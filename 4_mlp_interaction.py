# 4_mlp.py
# Interfaz: CanvasGrid + Chart de conteos (GT) + Chart Macro-F1 por bloque con marcadores (spikes)
# Entrenamiento: SOLO durante aprendizaje (congelado en evaluación si freeze_after_learn=True),
#                con opción de "peer learning" mínimo (aprender de vecinos) también en evaluación.
# Clasificación: cada agente clasifica a UN vecino aleatorio o a TODOS (según flag) con un MLP 2 capas + softmax
# F1: por bloque (no acumulado) a partir de CLASIFICACIÓN DE VECINOS + F1 por clase; guarda σ y soporta autosave
# Parada: tras evaluar el último sigma durante eval_block ticks, el modelo se detiene (running=False)

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

# =========================
# Configuración
# =========================
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
EMOTIONS_ORDER = ['neutral','happy','sad','anger','disgust','fear','surprise']
EMO2ID = {e:i for i,e in enumerate(EMOTIONS_ORDER)}

COLOR_BY_EMO = {
    'angry':   '#e74c3c',
    'sad':     '#2c3e50',
    'fear':    '#6e2c00',
    'disgust': '#27ae60',
    'surprise':'#ff9ff3',
    'neutral': '#95a5a6',
    'happy':   '#f39c12',
    'anger':   '#e74c3c',
}

NEGATIVE = {'angry','anger','disgust','fear','sad'}
POSITIVE = {'happy','surprise','neutral'}

@dataclass
class Cfg:
    meta_csv: str = "data/2_clip_embeddings/filtered/metadata_embeddings.csv"
    npy_embs: str = "data/2_clip_embeddings/filtered/embeddings.npy"
    dataset: str  = "ALL"          # "KDEF" | "JAFFE" | "ALL"
    width: int = 5
    height: int = 5
    n_agents: int = 10
    schedule: str = "progressive"   # "progressive" | "increasing" | "cyclic" | "shuffle"
    sample_policy: str = "random"   # "random" | "sequential"
    # Optimizador / entrenamiento
    lr: float = 3e-4
    wd: float = 5e-2
    label_smoothing: float = 0.05
    batch_size: int = 64
    # MLP
    mlp_hidden: int = 512
    mlp_dropout: float = 0.1
    # Dinámica de agentes
    avoid_thresh: int = 2
    move_prob: float = 0.7
    seed: int = 42
    # Fases y σ
    learn_iters: int = 1000
    eval_block: int = 200
    sigmas: tuple = (0,1,2,3,4)
    freeze_after_learn: bool = True
    # mezcla opcional: {"JAFFE":5, "KDEF":5}
    mix_counts = {"JAFFE": 8, "KDEF": 2}
    #mix_counts: Optional[Dict[str, int]] = None
    # autosave (0 = desactivado)
    autosave_every: int = 0
    out_prefix: str = "results_J8_K2"
    # --- Peer learning mínimo ---
    peer_learning: bool = True           # activar aprendizaje desde vecinos
    peer_conf_thresh: float = 0.75       # umbral de confianza propia para aceptar muestra de vecino
    peer_learn_in_eval: bool = False      # permitir aprender de vecinos también durante evaluación
    # --- Vecindad: clasificar uno o todos ---
    observe_all_neighbors: bool = True  # False: 1 vecino aleatorio; True: todos los vecinos de Moore

CFG = Cfg()

# =========================
# Utilidades
# =========================
def normalize_emotion(e: str) -> Optional[str]:
    if not isinstance(e, str): return None
    e = e.strip().lower()
    synonyms = {
        "angry":"anger","anger":"anger",
        "happiness":"happy","happy":"happy",
        "sadness":"sad","sad":"sad",
        "surprised":"surprise","surprise":"surprise",
        "neutrality":"neutral","neutral":"neutral",
        "fearful":"fear","fear":"fear",
        "disgusted":"disgust","disgust":"disgust"
    }
    return synonyms.get(e, e)

# =========================
# Cabezal MLP (reemplaza adapter + prototipos)
# =========================
class MLPHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 512, pdrop: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.drop = nn.Dropout(pdrop)

        # Inicialización compatible con PyTorch antiguos
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')  # aprox. válida para GELU
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        logits = self.fc2(h)
        return logits

# =========================
# Streams por identidad
# =========================
def make_identity_streams(meta_csv: str, npy_path: str, dataset: str, schedule: str, seed: int = 42):
    meta = pd.read_csv(meta_csv)
    embs = np.load(npy_path).astype(np.float32)
    if dataset.upper() in {"KDEF","JAFFE"}:
        meta = meta[meta["dataset"].astype(str).str.upper()==dataset.upper()].copy()
    else:
        meta = meta[meta["dataset"].astype(str).str.upper().isin(["KDEF","JAFFE"])].copy()

    meta["emotion"] = meta["emotion"].apply(normalize_emotion)
    meta = meta[meta["emotion"].isin(set(EMOTIONS_ORDER))].copy()

    def build_df(df):
        if schedule == "increasing":
            return df.sort_values(["sigma"]).reset_index(drop=True)
        elif schedule == "shuffle":
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        elif schedule == "cyclic":
            blocks = [df[df["sigma"] == s].sample(frac=1.0, random_state=seed)
                      for s in sorted(df["sigma"].unique())]
            L = max((len(b) for b in blocks), default=0)
            rows = []
            for i in range(L):
                for b in blocks:
                    if i < len(b): rows.append(b.iloc[i])
            return pd.DataFrame(rows).reset_index(drop=True) if rows else df
        else:
            if "original" not in df.columns:
                df["original"] = df.get("name", pd.Series(range(len(df)))).astype(str)
            groups = [g.sort_values("sigma").reset_index(drop=True) for _, g in df.groupby("original")]
            L = max((len(g) for g in groups), default=0)
            rows = []
            for i in range(L):
                for g in groups:
                    if i < len(g): rows.append(g.iloc[i])
            return pd.DataFrame(rows).reset_index(drop=True) if rows else df

    streams = {}
    for pid, g in meta.groupby(meta["person_id"].astype(str)):
        dfp = build_df(g.copy())
        rows = dfp["row_index"].values
        X = embs[rows]
        y = dfp["emotion"].map(EMO2ID).values.astype(int)
        sigs = dfp["sigma"].values.astype(int)
        ds_name = str(g["dataset"].iloc[0]).upper() if "dataset" in g.columns else "UNK"
        streams[str(pid)] = {"X":X, "y":y, "sigmas":sigs, "idx":0, "order":EMOTIONS_ORDER, "dataset": ds_name}

    ids = list(streams.keys())
    random.Random(seed).shuffle(ids)
    d = embs.shape[1]
    return streams, EMOTIONS_ORDER, d, ids

# =========================
# Agente
# =========================
class EmotionAgent(Agent):
    def __init__(self, unique_id, model, identity: str, d: int, order: List[str]):
        super().__init__(unique_id, model)
        self.identity = identity
        self.order = order
        self.n_classes = len(order)
        self.device = model.device

        # Dataset cultural del agente (para análisis intra/cross)
        self.dataset = model.streams[self.identity].get("dataset", "UNK")

        # ---- MLP per-agent ----
        self.head = MLPHead(d_in=d, n_classes=self.n_classes,
                            hidden=CFG.mlp_hidden, pdrop=CFG.mlp_dropout).to(self.device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        self.ce = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)

        # Buffer de entrenamiento online (mis propias muestras con label)
        self.buf_X: List[np.ndarray] = []
        self.buf_y: List[int] = []

        # Lo que YO muestro (para que otros me clasifiquen)
        self.display_emotion = "neutral"
        self.last_x: Optional[np.ndarray] = None
        self.last_y: Optional[int] = None

        # Predicción y métricas sobre VECINOS
        self.pred_emotion = "neutral"
        self.conf = 0.0
        self.last_correct = 0

        # Historial de predicciones sobre vecinos (para F1 de bloque)
        self._y_true_hist: List[int] = []
        self._y_pred_hist: List[int] = []
        self._tick_hist: List[int] = []

        # Para estética/visualización
        self.trust = 1.0
        self.emotion_history: List[str] = []

        self._seq_pos_by_sigma: Dict[int, int] = {}

        # Metadatos del último vecino procesado (para CSV/inspección)
        self._last_target_dataset = ""
        self._last_target_id = ""
        self._last_target_true_str = ""

    def _online_train_step(self, xb: torch.Tensor, yb: torch.Tensor):
        self.head.train()
        logits = self.head(xb)
        loss = self.ce(logits, yb)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.opt.step()
        self.head.eval()

    def _infer(self, x_np: np.ndarray) -> Tuple[int, float]:
        X = torch.from_numpy(x_np).float().to(self.device).unsqueeze(0)
        self.head.eval()
        with torch.no_grad():
            logits = self.head(X)
            probs = F.softmax(logits, dim=-1)[0]
            yhat = int(torch.argmax(probs).item())
            conf = float(probs.max().item())
        return yhat, conf

    def _move_random(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        free_spaces = [p for p in possible_steps if self.model.grid.is_cell_empty(p)]
        if free_spaces:
            self.model.grid.move_agent(self, random.choice(free_spaces))

    def _pick_index_for_sigma(self, sig_array: np.ndarray, sigma_target: int) -> int:
        candidates = np.where(sig_array == sigma_target)[0]
        if len(candidates) == 0:
            return random.randrange(len(sig_array))
        if CFG.sample_policy.lower() == "random":
            return int(random.choice(candidates))
        p = self._seq_pos_by_sigma.get(sigma_target, 0)
        ridx = int(candidates[p % len(candidates)])
        self._seq_pos_by_sigma[sigma_target] = p + 1
        return ridx

    def step(self):
        sigma_target = 0 if self.model.in_learn_phase else self.model.current_sigma

        # 1) Yo elijo MI muestra y la muestro (para que otros me clasifiquen)
        s = self.model.streams[self.identity]
        if len(s["y"]) > 0:
            ridx = self._pick_index_for_sigma(s["sigmas"], sigma_target)
            x_np = s["X"][ridx]
            y_true = int(s["y"][ridx])

            # Actualizo lo que muestro y guardo mi muestra para vecinos
            self.display_emotion = self.order[y_true]
            self.last_x = x_np
            self.last_y = y_true

            # Entrenamiento (sobre mis datos)
            if self.model.training_enabled:
                self.buf_X.append(x_np); self.buf_y.append(y_true)
                if len(self.buf_y) >= CFG.batch_size:
                    sel = np.random.choice(len(self.buf_y), size=CFG.batch_size, replace=False)
                    xb = torch.from_numpy(np.stack([self.buf_X[i] for i in sel], 0)).float().to(self.device)
                    yb = torch.tensor([self.buf_y[i] for i in sel], device=self.device).long()
                else:
                    xb = torch.from_numpy(x_np).float().to(self.device).unsqueeze(0)
                    yb = torch.tensor([y_true], device=self.device).long()
                self._online_train_step(xb, yb)

            # Historial breve para display (opcional)
            self.emotion_history.append(self.display_emotion)
            if len(self.emotion_history) > 5:
                self.emotion_history.pop(0)

        # 2) YO clasifico a vecinos (uno aleatorio o todos)
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        neigh_agents = [o for o in neighbors if isinstance(o, EmotionAgent) and (o.last_x is not None) and (o.last_y is not None)]
        if CFG.observe_all_neighbors:
            targets = neigh_agents
        else:
            targets = random.sample(neigh_agents, 1) if neigh_agents else []

        self.last_correct = 0
        self.conf = 0.0
        n_preds = 0

        for target in targets:
            yhat_n, conf_n = self._infer(target.last_x)

            # Registro rendimiento sobre vecinos (para F1 de bloque)
            self._y_true_hist.append(target.last_y)
            self._y_pred_hist.append(yhat_n)
            self._tick_hist.append(self.model.schedule.time)

            # Actualizo estado visible (última pred)
            self.pred_emotion = self.order[yhat_n]
            self.last_correct = 1 if yhat_n == target.last_y else 0

            # Acumula confianza media de este tick
            self.conf += float(conf_n)
            n_preds += 1

            # Confianza social basada en acierto percibido
            alpha = 0.1
            self.trust = (1 - alpha) * self.trust + alpha * self.last_correct

            # Metadatos del objetivo (último procesado)
            self._last_target_dataset = getattr(target, "dataset", "UNK")
            self._last_target_id = str(target.identity)
            self._last_target_true_str = self.order[target.last_y]

            # --- PEER LEARNING MÍNIMO ---
            # Aprende del vecino si hay suficiente confianza. Usa la etiqueta que el vecino declara (last_y).
            if CFG.peer_learning and (self.model.training_enabled or CFG.peer_learn_in_eval):
                if conf_n >= CFG.peer_conf_thresh:
                    xb_peer = torch.from_numpy(target.last_x).float().to(self.device).unsqueeze(0)
                    yb_peer = torch.tensor([target.last_y], device=self.device).long()
                    self._online_train_step(xb_peer, yb_peer)
            # --- FIN PEER LEARNING ---

        # confianza media del tick
        self.conf = self.conf / n_preds if n_preds > 0 else 0.0

        if not targets:
            # Si no hubo vecino clasificable, limpia metadatos
            self._last_target_dataset = ""
            self._last_target_id = ""
            self._last_target_true_str = ""

        # 3) Dinámica de movimiento (basada en emociones visibles de vecinos)
        perceived = [other.display_emotion for other in neighbors if isinstance(other, EmotionAgent)]
        val_pos = sum(1 for e in perceived if e in POSITIVE)
        val_neg = sum(1 for e in perceived if e in NEGATIVE)
        avoid = (val_neg - val_pos) >= CFG.avoid_thresh

        if avoid:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            free_spaces = [p for p in possible_steps if self.model.grid.is_cell_empty(p)]
            if free_spaces:
                self.model.grid.move_agent(self, random.choice(free_spaces))
        else:
            if random.random() < CFG.move_prob:
                self._move_random()

# =========================
# Modelo
# =========================
class EmotionModel(Model):
    def __init__(self,
                 meta_csv: str = CFG.meta_csv,
                 emb_path: str = CFG.npy_embs,
                 dataset: str = CFG.dataset,
                 width: int = CFG.width,
                 height: int = CFG.height,
                 num_agents: int = CFG.n_agents,
                 schedule: str = CFG.schedule):
        super().__init__()
        random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)

        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        streams, order, d, ids = make_identity_streams(meta_csv, emb_path, dataset, schedule, seed=CFG.seed)
        if len(ids) == 0:
            raise RuntimeError("No se encontraron identidades en metadata filtrada.")
        self.streams = streams

        capacity = width * height
        if num_agents > capacity:
            print(f"[WARN] num_agents={num_agents} > capacity={capacity}. Se ajusta a {capacity}.")
            num_agents = capacity

        mix = CFG.mix_counts
        if mix:
            chosen = []
            for ds_name, count in mix.items():
                ds_name = ds_name.upper()
                pool = [pid for pid, s in self.streams.items() if s.get("dataset","") == ds_name]
                random.shuffle(pool)
                take = min(count, len(pool))
                if take < count:
                    print(f"[WARN] Solo hay {take} identidades disponibles en {ds_name} (pedidas {count}).")
                chosen.extend(pool[:take])
        else:
            if dataset.upper() in {"JAFFE","KDEF"}:
                pool = [pid for pid, s in self.streams.items() if s.get("dataset","") == dataset.upper()]
            else:
                pool = list(self.streams.keys())
            random.shuffle(pool)
            chosen = pool[:min(num_agents, len(pool))]

        chosen = chosen[:min(len(chosen), capacity)]

        all_cells = [(x, y) for x in range(width) for y in range(height)]
        random.shuffle(all_cells)
        next_free = 0

        for i, pid in enumerate(chosen):
            a = EmotionAgent(i, self, pid, d, order)
            self.schedule.add(a)
            if next_free >= len(all_cells):
                break
            self.grid.place_agent(a, all_cells[next_free])
            next_free += 1

        # Fase/σ y control de training
        self.learn_iters = int(CFG.learn_iters)
        self.eval_block  = int(CFG.eval_block)
        self.sigmas_seq  = list(CFG.sigmas)
        self.current_sigma = 0
        self.in_learn_phase = True
        self.training_enabled = True

        # Parada tras completar todos los bloques de σ
        self.stop_tick = self.learn_iters + self.eval_block * len(self.sigmas_seq)
        self.running = True
        self._stopped_once = False

        # marcadores y control de bloque
        self._learn_end_mark = 0.0
        self._sigma_change_mark = 0.0
        self.block_start_tick = 0  # F1 por bloque cuenta desde aquí

        def count_display(model, emo_name):
            target = normalize_emotion(emo_name)
            return sum(1 for ag in model.schedule.agents
                       if isinstance(ag, EmotionAgent) and ag.display_emotion == target)

        # F1 de bloque a partir de CLASIFICACIONES SOBRE VECINOS registradas
        def macro_f1_block(model):
            yt, yp = [], []
            t0 = model.block_start_tick
            for ag in model.schedule.agents:
                if isinstance(ag, EmotionAgent) and ag._tick_hist:
                    for y, yhat, tt in zip(ag._y_true_hist, ag._y_pred_hist, ag._tick_hist):
                        if tt >= t0:
                            yt.append(y); yp.append(yhat)
            if not yt:
                return 0.0
            return float(f1_score(np.array(yt), np.array(yp), average="macro"))

        def f1_class_block(model, emo_name):
            target = EMO2ID[normalize_emotion(emo_name)]
            yt, yp = [], []
            t0 = model.block_start_tick
            for ag in model.schedule.agents:
                if isinstance(ag, EmotionAgent) and ag._tick_hist:
                    for y, yhat, tt in zip(ag._y_true_hist, ag._y_pred_hist, ag._tick_hist):
                        if tt >= t0:
                            yt.append(1 if y == target else 0)
                            yp.append(1 if yhat == target else 0)
            if not yt or sum(yt) == 0:
                return 0.0
            return float(f1_score(np.array(yt), np.array(yp), average="binary", zero_division=0))

        self.datacollector = DataCollector(
            model_reporters={
                **{e: (lambda m, ee=e: count_display(m, ee)) for e in EMOTION_LABELS},
                "Macro-F1 (block)": macro_f1_block,
                **{f"F1_{e}": (lambda m, ee=e: f1_class_block(m, ee)) for e in EMOTION_LABELS},
                "sigma": lambda m: m.current_sigma,
                "LearnEndMark": lambda m: m._learn_end_mark,
                "SigmaChangeMark": lambda m: m._sigma_change_mark
            },
            agent_reporters={
                # Lo que YO muestro (para ver el GT local de cada agente)
                "self_display_emotion": lambda a: a.display_emotion,
                # Métricas de la última clasificación sobre vecino de ESTE tick (si varios, el último iterado)
                "gt_emotion":          lambda a: getattr(a, "_last_target_true_str", ""),
                "pred_emotion":        lambda a: a.pred_emotion,
                "correct":             lambda a: int(a.last_correct),
                "conf":                lambda a: float(a.conf),
                "sigma":               lambda a: a.model.current_sigma,
                # Identidades y grupos culturales
                "agent_dataset":       lambda a: a.dataset,
                "target_dataset":      lambda a: getattr(a, "_last_target_dataset", ""),
                "identity":            lambda a: a.identity,
                "target_id":           lambda a: getattr(a, "_last_target_id", ""),
                # Otros
                "pos":                 lambda a: a.pos,
                "trust":               lambda a: a.trust
            }
        )

        self.datacollector.collect(self)

    def _update_phase_and_markers(self):
        t = self.schedule.time
        self._learn_end_mark = 0.0
        self._sigma_change_mark = 0.0

        if t == self.learn_iters:
            self.in_learn_phase = False
            self.current_sigma = self.sigmas_seq[0]
            if CFG.freeze_after_learn:
                self.training_enabled = False
            self._sigma_change_mark = 1.05
            self._learn_end_mark = 1.05
            self.block_start_tick = t
            print(f"[LEARN→EVAL] t={t} | σ={self.current_sigma} | train_enabled={self.training_enabled}")

        elif t > self.learn_iters:
            idx = (t - self.learn_iters) // self.eval_block
            idx = min(idx, len(self.sigmas_seq)-1)
            sigma = self.sigmas_seq[idx]
            if sigma != self.current_sigma:
                self.current_sigma = sigma
                self._sigma_change_mark = 1.05
                self.block_start_tick = t
                print(f"[σ CHANGE] t={t} | σ={self.current_sigma}")

        else:
            self.in_learn_phase = True
            self.current_sigma = 0
            self.training_enabled = True

    def _autosave_csv(self):
        try:
            mdf = self.datacollector.get_model_vars_dataframe()
            adf = self.datacollector.get_agent_vars_dataframe()
            mdf.to_csv(f"{CFG.out_prefix}_model_timeseries.csv")
            adf.to_csv(f"{CFG.out_prefix}_agent_timeseries.csv", index=False)
            print(f"[SAVE] {CFG.out_prefix}_model_timeseries.csv  &  {CFG.out_prefix}_agent_timeseries.csv")
        except Exception as e:
            print(f"[SAVE][ERROR] {e}")

    def step(self):
        # Parada tras completar todos los bloques de σ con duración uniforme
        if self.schedule.time >= self.stop_tick:
            if not self._stopped_once:
                print(f"[STOP] t={self.schedule.time} alcanzado. Evaluación completa. Deteniendo modelo.")
                self._autosave_csv()
                self._stopped_once = True
            self.running = False
            return

        self._update_phase_and_markers()
        self.schedule.step()
        self.datacollector.collect(self)

        if CFG.autosave_every and (self.schedule.time % CFG.autosave_every == 0):
            self._autosave_csv()

# =========================
# Portrayal
# =========================
def agent_portrayal(agent: EmotionAgent):
    emo = agent.display_emotion
    color = COLOR_BY_EMO.get(emo, "#3498db")
    r = 0.5 + 0.5 * float(agent.trust)
    text = f"{agent.identity} ({getattr(agent,'dataset','?')})\n{emo}"
    return {
        "Shape": "circle",
        "Filled": True,
        "Layer": 0,
        "r": r,
        "Color": color,
        "text": text,
        "text_color": "white"
    }

# =========================
# Lanzamiento del servidor
# =========================
if __name__ == '__main__':
    modo_interactivo = True

    parametros_modelo = {
        "meta_csv": CFG.meta_csv,
        "emb_path": CFG.npy_embs,
        "dataset":  CFG.dataset,
        "width":    CFG.width,
        "height":   CFG.height,
        "num_agents": CFG.n_agents,
        "schedule": CFG.schedule
    }

    if modo_interactivo:
        grid = CanvasGrid(agent_portrayal, CFG.width, CFG.height, 600, 600)
        chart_counts = ChartModule(
            [{"Label": e, "Color": COLOR_BY_EMO.get(e, "#7f8c8d")} for e in EMOTION_LABELS],
            data_collector_name='datacollector'
        )
        chart_f1 = ChartModule(
            [
                {"Label": "Macro-F1 (block)", "Color": "#2ecc71"},
                {"Label": "LearnEndMark", "Color": "#e74c3c"},
                {"Label": "SigmaChangeMark", "Color": "#9b59b6"},
            ],
            data_collector_name='datacollector'
        )

        server = ModularServer(
            EmotionModel,
            [grid, chart_counts, chart_f1],
            "Emotional Agent Interaction — vecinos con MLP (F1 por bloque, peer learning mínimo)",
            parametros_modelo
        )
        server.port = 8521
        server.launch()
    else:
        pasos = 100000  # no importa, se detendrá solo al llegar a stop_tick
        model = EmotionModel(**parametros_modelo)
        for _ in range(pasos):
            model.step()
        print("✅ Fin (parada automática).")
