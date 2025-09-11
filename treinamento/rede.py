import os, json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

# ================================
# 0) Escolha qual RUN usar (mude aqui)
# ================================
RUN_DIR = "runs/run_20250901_235809"   # <<<<<< troque para o run desejado
REGISTRY_PATH = os.path.join(RUN_DIR, "registry.json")
SELECTOR_PATH = os.path.join(RUN_DIR, "selector.joblib")

# ================================
# 1) Carregar selector e registry
# ================================
selector = joblib.load(SELECTOR_PATH)
with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

model_entries = registry.get("models", {})  # dict: nome -> {path, metrics, ...}

# ================================
# 2) Carregar CSV de teste/holdout
# ================================
df = pd.read_csv("Trafego/normal.csv")  # ou encoding="ISO-8859-1"

# ================================
# 3) Pré-processamento igual ao treino
# ================================
df.columns = df.columns.str.strip()

# Mesmas colunas removidas no treino
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp",
             "Init_Win_bytes_backward", "Init_Win_bytes_forward"]
df = df.drop(columns=drop_cols, errors="ignore")

# Label (se existir no CSV)
has_label = "Label" in df.columns
if has_label:
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
    y = df["Label"].values
    X = df.drop(columns=["Label"], errors="ignore")
else:
    y = None
    X = df

# Numérico + tratar inf/NaN
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

# Alinhar colunas com as usadas no treino (feature_names_in_ do selector)
missing = [c for c in selector.feature_names_in_ if c not in X.columns]
if missing:
    raise ValueError(f"As seguintes colunas esperadas pelo selector não estão no CSV: {missing}")

X = X[selector.feature_names_in_]
X_selected = selector.transform(X)

# ================================
# 4) Avaliar TODOS os modelos do registry
# ================================
def get_proba(model, Xmat):
    try:
        return model.predict_proba(Xmat)[:, 1]
    except Exception:
        if hasattr(model, "decision_function"):
            s = model.decision_function(Xmat)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else np.zeros_like(s)
        # último caso: usa a própria predição binária como score
        return model.predict(Xmat)

plt.figure(figsize=(8, 6))
results = []
pred_cols = []  # para salvar as predições

for name, meta in model_entries.items():
    model_path = meta["path"]
    if not os.path.exists(model_path):
        print(f"[!] Modelo '{name}' não encontrado em {model_path} — pulando.")
        continue

    clf = joblib.load(model_path)
    y_score = get_proba(clf, X_selected)

    # Se existir threshold salvo (em outra versão do registry), usa-o; senão, 0.5
    thr = meta.get("threshold", 0.5)
    y_pred = (y_score >= thr).astype(int)

    if has_label:
        print(f"\n=== {name} === (thr={thr:.4f})")
        print(classification_report(y, y_pred, digits=4))
        print("Matriz de Confusão:")
        print(confusion_matrix(y, y_pred))

        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        pr_auc  = average_precision_score(y, y_score)

        results.append({
            "modelo": name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": thr
        })

        plt.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC={roc_auc:.3f})")
    else:
        print(f"{name}: CSV sem Label — gerando apenas scores/predições.")

    # guardar colunas de saída por modelo
    col_score = f"{name}_score"
    col_pred  = f"{name}_pred"
    df[col_score] = y_score
    df[col_pred]  = y_pred
    pred_cols += [col_score, col_pred]

# ================================
# 5) Plot ROC (se houver Label) e salvar resultados
# ================================
if has_label and results:
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC - Comparação (holdout)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    df_result = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    out_metrics = os.path.join(RUN_DIR, "avaliacao_holdout_metricas.csv")
    df_result.to_csv(out_metrics, index=False)
    print(f"\n✅ Métricas do holdout salvas em: {out_metrics}")
    print(df_result.to_string(index=False))

# Sempre salva as predições (mesmo sem Label)
out_preds = os.path.join(RUN_DIR, "avaliacao_holdout_predicoes.csv")
cols_to_save = (["Label"] if has_label else []) + pred_cols
df_out = df[cols_to_save] if cols_to_save else pd.DataFrame(index=df.index)
df_out.to_csv(out_preds, index=False)
print(f"✅ Predições/scores salvos em: {out_preds}")

# ================================
# 6) (Opcional) Avaliar também o best_*.joblib se quiser
# ================================
best_files = [f for f in os.listdir(RUN_DIR) if f.startswith("best_") and f.endswith(".joblib")]
if best_files:
    best_path = os.path.join(RUN_DIR, best_files[0])
    print(f"\n🔎 Avaliando também o best: {os.path.basename(best_path)}")
    best = joblib.load(best_path)
    yb_score = get_proba(best, X_selected)
    yb_pred  = (yb_score >= 0.5).astype(int)
    if has_label:
        print(classification_report(y, yb_pred, digits=4))
        print(confusion_matrix(y, yb_pred))
