# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import io
import sys
import math
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_PATH = "dataset.csv"

EXPECTED_COLS = [
    "timestamp","date","temp","temp_unit","hum","hum_unit",
    "equip_id","equip_status","weekend","ferias","prob_on"
]

PRIORITY_METRICS = ("f1_macro", "balanced_accuracy", "accuracy")  # ordem de desempate


# -----------------------------
# Utils de leitura e limpeza
# -----------------------------
def _read_raw_csv(path):
    """
    Tenta ler o CSV normalmente; se não achar 'timestamp' no header,
    procura a linha do cabeçalho real e relê a partir dali.
    Compatível com pandas 1.1.x (usa error_bad_lines/warn_bad_lines).
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Arquivo não encontrado: {}".format(path))

    # 1� tentativa: leitura direta
    try:
        df = pd.read_csv(path, dtype=str, error_bad_lines=False, warn_bad_lines=False)
        if "timestamp" in df.columns:
            return df
    except Exception:
        pass

    # Fallback: localizar linha de cabeçalho "timestamp,..." no arquivo bruto
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("timestamp,"):
            header_idx = i
            break

    if header_idx is not None:
        # Releitura a partir do header real
        buf = io.StringIO("\n".join(lines[header_idx:]))
        df = pd.read_csv(buf, dtype=str, error_bad_lines=False, warn_bad_lines=False)
        return df

    # Último recurso: ler sem header e atribuir colunas conhecidas pelo que existir
    buf = io.StringIO("\n".join(lines))
    tmp = pd.read_csv(buf, header=None, dtype=str, error_bad_lines=False, warn_bad_lines=False)
    # elimina linhas totalmente vazias
    tmp = tmp.dropna(how="all")
    # limita colunas conhecidas
    n = min(len(EXPECTED_COLS), tmp.shape[1])
    tmp = tmp.iloc[:, :n]
    tmp.columns = EXPECTED_COLS[:n]
    return tmp


def _coerce_numeric(s):
    """Converte Series para numérico, tratando strings vazias."""
    return pd.to_numeric(s, errors="coerce")


def _majority_vote(series):
    """Maioria binária; se empatar, arredonda pela média."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    m = s.mean()
    return int(round(m))


def preparar_dataframe(df_raw):
    # Remover colunas totalmente vazias e linhas vazias
    df = df_raw.copy()
    df.replace({"": np.nan, "null": np.nan, "None": np.nan}, inplace=True)
    df.dropna(how="all", inplace=True)

    # Se a primeira linha for um header "embaralhado" repetido em linhas seguintes, remove
    # (após converter timestamp vamos eliminar qualquer linha que não parseou)
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Normaliza nomes de colunas (tudo minúsculo)
    df.columns = [str(c).strip() for c in df.columns]

    # Converte campos básicos
    # timestamp pode vir com sujeira; converter coerentemente
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Jogue fora linhas com timestamp inválido
    df = df[~df["timestamp"].isna()].copy()

    # Normaliza sensores
    df["temp"] = _coerce_numeric(df["temp"])
    df["hum"]  = _coerce_numeric(df["hum"])

    # Normaliza flags auxiliares (podem estar vazias)
    df["weekend"] = _coerce_numeric(df.get("weekend", np.nan)).fillna(0).astype(int)
    df["ferias"]  = _coerce_numeric(df.get("ferias", np.nan)).fillna(0).astype(int)

    # prob_on pode vir vazio; tente converter
    if "prob_on" in df.columns:
        df["prob_on"] = _coerce_numeric(df["prob_on"])
    else:
        df["prob_on"] = np.nan

    # Normaliza equip_status -> 0/1
    # aceita: 'ativo','ligado','on','1','true','yes' como 1
    norm = (
        df["equip_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["equip_status"] = np.where(
        norm.isin(["1","true","t","sim","yes","y","on","ativo","ligado"]),
        1, 
        np.where(norm.isin(["0","false","f","nao","não","no","off","desligado","inativo"]), 0, np.nan)
    )
    # É possível que durante geração inicial venham linhas sem status — mantemos, e a agregação por janela faz a "maioria"

    # Ordena e remove duplicatas exatas de timestamp (mantém a última)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # Gera coluna 'date'
    df["date"] = df["timestamp"].dt.date.astype(str)

    # -------- Agregação por janela de 15 minutos --------
    df["bucket_15"] = df["timestamp"].dt.floor("15T")

    def _agg_group(g):
        # Última leitura da janela para sensores/unidades/id
        last = g.sort_values("timestamp").iloc[-1].copy()

        # maioria para status (0/1) se houver pelo menos 1 medição na janela
        if g["equip_status"].notna().any():
            last["equip_status"] = _majority_vote(g["equip_status"])
        else:
            last["equip_status"] = np.nan

        # weekend/ferias -> maioria
        last["weekend"] = _majority_vote(g["weekend"]) if g["weekend"].notna().any() else 0
        last["ferias"]  = _majority_vote(g["ferias"])  if g["ferias"].notna().any()  else 0

        # prob_on -> média (se existir)
        if "prob_on" in g.columns and g["prob_on"].notna().any():
            last["prob_on"] = float(np.nanmean(pd.to_numeric(g["prob_on"], errors="coerce")))
        else:
            last["prob_on"] = np.nan

        # timestamp vira o início da janela
        last["timestamp"] = last["bucket_15"]
        last["date"] = last["timestamp"].date().isoformat()
        return last[["timestamp","date","temp","temp_unit","hum","hum_unit",
                     "equip_id","equip_status","weekend","ferias","prob_on"]]

    df_agg = (
        df.groupby("bucket_15", as_index=False, group_keys=False)
          .apply(_agg_group)
          .reset_index(drop=True)
    )

    # Remove linhas sem target (ainda pode sobrar janela sem status)
    df_agg = df_agg[~df_agg["equip_status"].isna()].copy()

    # -------- Features temporais e lags --------
    df_agg["hour"]      = df_agg["timestamp"].dt.hour
    df_agg["minute"]    = df_agg["timestamp"].dt.minute
    df_agg["dayofweek"] = df_agg["timestamp"].dt.dayofweek

    # encoding cíclico
    df_agg["hour_sin"] = np.sin(2*np.pi*df_agg["hour"]/24.0)
    df_agg["hour_cos"] = np.cos(2*np.pi*df_agg["hour"]/24.0)
    df_agg["dow_sin"]  = np.sin(2*np.pi*df_agg["dayofweek"]/7.0)
    df_agg["dow_cos"]  = np.cos(2*np.pi*df_agg["dayofweek"]/7.0)
    df_agg["min_sin"]  = np.sin(2*np.pi*df_agg["minute"]/60.0)
    df_agg["min_cos"]  = np.cos(2*np.pi*df_agg["minute"]/60.0)

    # lags e médias móveis curtas (3 janelas de 15min = 45min)
    df_agg = df_agg.sort_values("timestamp").reset_index(drop=True)
    df_agg["status_prev"] = df_agg["equip_status"].shift(1).fillna(0).astype(int)
    df_agg["temp_ma3"] = df_agg["temp"].rolling(3, min_periods=1).mean()
    df_agg["hum_ma3"]  = df_agg["hum"].rolling(3, min_periods=1).mean()

    # feature list (inclui prob_on e weekend/ferias)
    feature_cols = [
        "temp","hum","temp_ma3","hum_ma3",
        "hour_sin","hour_cos","dow_sin","dow_cos","min_sin","min_cos",
        "status_prev","weekend","ferias","prob_on"
    ]
    # garante existência
    for c in feature_cols:
        if c not in df_agg.columns:
            df_agg[c] = np.nan

    # tipagens finais
    df_agg["temp"] = _coerce_numeric(df_agg["temp"])
    df_agg["hum"]  = _coerce_numeric(df_agg["hum"])
    df_agg["prob_on"] = _coerce_numeric(df_agg["prob_on"])

    df_agg = df_agg.dropna(subset=["temp","hum","equip_status"]).reset_index(drop=True)

    return df_agg, feature_cols


# -----------------------------
# Split e avaliação
# -----------------------------
def split_temporal_com_duas_classes(df_sub, min_train_ratio=0.6, max_train_ratio=0.9):
    """Procura corte temporal que assegure 2 classes em treino e teste."""
    df_sub = df_sub.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sub)
    if n < 60:
        return None, None
    fracs = np.linspace(min_train_ratio, max_train_ratio, 16)
    for f in fracs:
        cut = int(n * f)
        tr, te = df_sub.iloc[:cut], df_sub.iloc[cut:]
        if len(tr) < 40 or len(te) < 20:
            continue
        if tr["equip_status"].nunique() >= 2 and te["equip_status"].nunique() >= 2:
            return tr, te
    return None, None


def avaliar(y_true, y_pred):
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return m


def escolher_melhor(resultados):
    def key(r):
        return (r["metrics"]["f1_macro"], r["metrics"]["balanced_accuracy"], r["metrics"]["accuracy"])
    return max(resultados, key=key)


def treina_subset(df_sub, tag, feature_cols):
    df_sub = df_sub.dropna(subset=["temp","hum","equip_status"]).copy()
    if len(df_sub) < 50:
        print("⚠️ Subconjunto '{}' insuficiente (|X|={}). Gere mais dados.".format(tag, len(df_sub)))
        return None

    # Caso de 1 única classe → DummyClassifier
    if df_sub["equip_status"].nunique() < 2:
        uniq = int(df_sub["equip_status"].iloc[0])
        print("ℹ️ '{}': apenas 1 classe ({}). Usando DummyClassifier(most_frequent).".format(tag, uniq))
        X_all = df_sub[feature_cols].fillna(0)
        y_all = df_sub["equip_status"].astype(int).values
        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_all_s, y_all)
        out = "modelo_{}.pkl".format(tag)
        joblib.dump({"model": dummy, "scaler": scaler, "feature_cols": feature_cols}, out)
        print("✔ '{}': DummyClassifier salvo em {}".format(tag, out))
        return "Dummy", dummy, 1.0, scaler

    # Split temporal; fallback estratificado
    tr, te = split_temporal_com_duas_classes(df_sub)
    if tr is None:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        y_all = df_sub["equip_status"].values
        idx_tr, idx_te = next(sss.split(df_sub, y_all))
        tr = df_sub.iloc[idx_tr]
        te = df_sub.iloc[idx_te]
        print("ℹ️ '{}': usando StratifiedShuffleSplit como fallback.".format(tag))

    Xtr = tr[feature_cols].fillna(0)
    ytr = tr["equip_status"].astype(int).values
    Xte = te[feature_cols].fillna(0)
    yte = te["equip_status"].astype(int).values

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    modelos = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        modelos["XGB"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", nthread=4, use_label_encoder=False
        )

    resultados = []
    for nome, mdl in modelos.items():
        Xtr_fit = Xtr_s if nome == "LogReg" else Xtr
        Xte_eval = Xte_s if nome == "LogReg" else Xte

        mdl.fit(Xtr_fit, ytr)
        preds = mdl.predict(Xte_eval)
        mets = avaliar(yte, preds)

        print("\n[{}] {} - Acc: {:.3f} | BalAcc: {:.3f} | F1-macro: {:.3f}".format(
            tag, nome, mets["accuracy"], mets["balanced_accuracy"], mets["f1_macro"]
        ))
        print(classification_report(yte, preds, digits=3, zero_division=0))

        resultados.append({
            "name": nome, "model": mdl, "metrics": mets
        })

    best = escolher_melhor(resultados)
    out = "modelo_{}.pkl".format(tag)
    joblib.dump({"model": best["model"], "scaler": scaler, "feature_cols": feature_cols}, out)
    print("✔ '{}': melhor={} (F1-macro={:.3f} | BalAcc={:.3f} | Acc={:.3f}) salvo em {}".format(
        tag, best["name"], best["metrics"]["f1_macro"], best["metrics"]["balanced_accuracy"], best["metrics"]["accuracy"], out
    ))
    return best["name"], best["model"], best["metrics"]["accuracy"], scaler


# -----------------------------
# Main
# -----------------------------
import joblib

def main():
    print("🔍 Lendo e limpando dataset...")
    raw = _read_raw_csv(DATA_PATH)
    df, feature_cols = preparar_dataframe(raw)

    if df.empty:
        raise RuntimeError("Dataset vazio após limpeza. Gere mais dados no Node-RED e tente novamente.")

    print("✅ Dataset limpo (amostra):")
    print(df.head(), "\n")

    # Estatísticas rápidas
    dist = df["equip_status"].value_counts(dropna=False).to_dict()
    print("Distribuição de classes (geral):", dist)

    # Treina separado por ferias
    df["ferias"] = df["ferias"].fillna(0).astype(int)

    best_normais = treina_subset(df[df["ferias"] == 0], "normal", feature_cols)
    best_ferias  = treina_subset(df[df["ferias"] == 1], "ferias", feature_cols)

    # Fallback geral
    if best_normais and not best_ferias:
        joblib.dump({"model": best_normais[1], "scaler": best_normais[3], "feature_cols": feature_cols}, "modelo.pkl")
        print("ℹ️ Só 'normal' treinou; salvei fallback em modelo.pkl.")
    elif best_ferias and not best_normais:
        joblib.dump({"model": best_ferias[1], "scaler": best_ferias[3], "feature_cols": feature_cols}, "modelo.pkl")
        print("ℹ️ Só 'ferias' treinou; salvei fallback em modelo.pkl.")
    elif best_normais and best_ferias:
        # também grava um default apontando para o 'normal'
        joblib.dump({"model": best_normais[1], "scaler": best_normais[3], "feature_cols": feature_cols}, "modelo.pkl")
        print("ℹ️ Ambos treinaram; 'modelo.pkl' aponta para o modelo_normal por padrão.")
    else:
        raise RuntimeError("❌ Nenhum subconjunto conseguiu treinar. Gere mais dados ou ajuste o gerador (mais 'inativo').")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Erro no treino:", repr(e))
        sys.exit(1)
