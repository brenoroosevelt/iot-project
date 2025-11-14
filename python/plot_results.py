#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera gráficos a partir dos resumos de treino fornecidos:
- distribuição de classes
- comparação de métricas (accuracy, balanced_accuracy, f1_macro) por modelo
- precision/recall/f1 por classe (0 e 1) por modelo

Salva imagens em ./plots/
Requisitos: matplotlib, pandas, numpy
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Dados (cole aqui os valores que você postou) ---
class_dist = {1: 433, 0: 109}

models_summary = {
    "LogReg": {"accuracy": 0.857, "balanced_accuracy": 0.883, "f1_macro": 0.853},
    "RF":     {"accuracy": 0.880, "balanced_accuracy": 0.840, "f1_macro": 0.860},
    "XGB":    {"accuracy": 0.945, "balanced_accuracy": 0.937, "f1_macro": 0.939},
}

class_reports = {
    "LogReg": {
        0: {"precision": 0.721, "recall": 0.974, "f1": 0.829, "support": 77},
        1: {"precision": 0.982, "recall": 0.793, "f1": 0.877, "support": 140},
    },
    "RF": {
        0: {"precision": 0.947, "recall": 0.701, "f1": 0.806, "support": 77},
        1: {"precision": 0.856, "recall": 0.979, "f1": 0.913, "support": 140},
    },
    "XGB": {
        0: {"precision": 0.933, "recall": 0.909, "f1": 0.921, "support": 77},
        1: {"precision": 0.951, "recall": 0.964, "f1": 0.957, "support": 140},
    }
}

# --- Preparar diretório de saída ---
outdir = os.path.join(os.getcwd(), "plots")
os.makedirs(outdir, exist_ok=True)

# --- Mostrar/Salvar resumo em CSV ---
df_summary = pd.DataFrame.from_dict(models_summary, orient="index").reset_index().rename(columns={"index": "model"})
df_summary.to_csv(os.path.join(outdir, "models_summary.csv"), index=False)

df_dist = pd.DataFrame(sorted(class_dist.items()), columns=["class", "count"])
df_dist.to_csv(os.path.join(outdir, "class_distribution.csv"), index=False)

rows = []
for mdl, cr in class_reports.items():
    for cls, stats in cr.items():
        rows.append({"model": mdl, "class": cls, "precision": stats["precision"],
                     "recall": stats["recall"], "f1": stats["f1"], "support": stats["support"]})
df_cr = pd.DataFrame(rows)
df_cr.to_csv(os.path.join(outdir, "classification_reports.csv"), index=False)

print("CSV salvo em:", outdir)

# --- Gráfico 1: distribuição de classes (bar) ---
plt.figure(figsize=(6,4))
plt.bar(df_dist["class"].astype(str), df_dist["count"])
plt.title("Distribuição de classes (equip_status)")
plt.xlabel("Classe (0=inativo, 1=ativo)")
plt.ylabel("Contagem")
plt.tight_layout()
f1 = os.path.join(outdir, "class_distribution.png")
plt.savefig(f1)
plt.close()
print("Salvo:", f1)

# --- Gráfico 2: comparação de métricas por modelo (accuracy, balanced_accuracy, f1_macro) ---
metrics = ["accuracy", "balanced_accuracy", "f1_macro"]
models = list(models_summary.keys())
x = np.arange(len(models))
width = 0.22

plt.figure(figsize=(8,4))
for i, m in enumerate(metrics):
    vals = [models_summary[mdl][m] for mdl in models]
    plt.bar(x + (i-1)*width, vals, width=width, label=m)
plt.xticks(x, models)
plt.ylim(0, 1.0)
plt.title("Comparação de métricas por modelo")
plt.ylabel("Valor")
plt.legend()
plt.tight_layout()
f2 = os.path.join(outdir, "metrics_comparison.png")
plt.savefig(f2)
plt.close()
print("Salvo:", f2)

# --- Gráfico 3: precision/recall/f1 por classe (um arquivo por classe) ---
metrics_pr = ["precision", "recall", "f1"]
for cls in [0, 1]:
    plt.figure(figsize=(8,4))
    x = np.arange(len(models))
    for i, m in enumerate(metrics_pr):
        vals = [class_reports[mdl][cls][m] for mdl in models]
        plt.bar(x + (i-1)*width, vals, width=width, label=m)
    plt.xticks(x, models)
    plt.ylim(0, 1.0)
    plt.title(f"Classe {cls} — Precision / Recall / F1 por modelo")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(outdir, f"class_{cls}_prf.png")
    plt.savefig(fname)
    plt.close()
    print("Salvo:", fname)

print("\nTodos os gráficos gerados em:", outdir)
print("Arquivos gerados:")
for fn in sorted(os.listdir(outdir)):
    print(" -", fn)
