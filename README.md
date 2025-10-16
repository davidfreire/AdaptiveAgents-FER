# AdaptiveAgents-FER
Simulation framework for "Modeling Cultural Bias in Facial Expression Recognition with Adaptive Agents". Implements adaptive multi-agent learning with per-agent MLPs and cultural bias dynamics on a grid.


# Emotional Agents — per-agent MLP on a Grid

Simulation of emotional agents that interact on a 2D grid.  
Each agent carries a small MLP classifier trained on CLIP embeddings and can learn both from its own samples and from confident neighbors ("peer learning").  
The model tracks performance block-wise (per sigma) using macro-F1 and provides a live visualization.

---

## Features

- Grid-based agent simulation (built with **Mesa**)
- Individual **MLP** heads per agent (`torch`)
- Online learning and optional **peer learning**
- Phase scheduling: learn → evaluation by sigma blocks
- Metrics and **charts** for per-block Macro-F1
- Optional autosave of agent/model time series to CSV

---

## Installation

git clone https://github.com/davidfreire/AdaptiveAgents-FER.git

cd AdaptiveAgents-FER

python3 -m venv .venv

source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

---

## Running

Interactive mode (Mesa server)

Launches the grid and charts in your browser:

python 4_mlp.py

The default server port is 8521 → open http://localhost:8521

---

## Configuration

Main parameters are defined in `Cfg` (at the top of `4_mlp.py`):

| Parameter | Description | Default |
|------------|-------------|----------|
| `meta_csv` | Path to metadata CSV | `data/2_clip_embeddings/filtered/metadata_embeddings.csv` |
| `npy_embs` | Path to `.npy` embedding array | `data/2_clip_embeddings/filtered/embeddings.npy` |
| `width`, `height` | Grid size | `5 × 5` |
| `n_agents` | Number of active agents | `10` |
| `peer_learning` | Enable minimal peer learning | `True` |
| `freeze_after_learn` | Freeze weights after learning phase | `True` |

You can modify these values directly in `Cfg` or parameterize them later.

---

## Data

The code expects CLIP embeddings and metadata structured like:

data/2_clip_embeddings/filtered/embeddings.npy

data/2_clip_embeddings/filtered/metadata_embeddings.csv

---

## Outputs

During training/evaluation, the following files are generated:
```
results_*_model_timeseries.csv

results_*_agent_timeseries.csv
```
They contain time-series data for global F1 metrics and agent states.

---

## Notes

Each agent has its own classifier and dataset origin.

Learning occurs only in the training phase unless peer_learn_in_eval=True.

The system stops automatically when all sigma blocks are evaluated.

---

## License

MIT License © 2025 davidfreire
---

## Citation

[arXiv](https://arxiv.org/abs/2510.13557)

```
@InProceedings{Freire_AAIS25,
    author    = {David Freire-Obregón and José Salas-Cáceres and Javier Lorenzo-Navarro and Oliverio J. Santana and Daniel Hernández-Sosa and Modesto Castrillón-Santana},
    title     = {Modeling Cultural Bias in Facial Expression Recognition with Adaptive Agents},
    booktitle = {Proceedings of the International Symposium on Agentic Artificial Intelligence Systems (AAIS 2025)},
    month     = {November},
    year      = {2025}
}
```
---

