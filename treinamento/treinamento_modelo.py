from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier)
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings, os, json
from datetime import datetime

warnings.filterwarnings("ignore")

# ================================
# 0. Pastas de saída (um run por execução)
# ================================
run_dir = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
models_dir = os.path.join(run_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# ================================
# 1. Carregar CSVs
# ================================
linhas_para_ler2 = 164407 - 22551
df_ddos = pd.read_csv(
    "./Trafego/DDos.csv",
    skiprows=range(1, 22551),
    nrows=linhas_para_ler2
)

linhas_para_ler = 275048 - 87794
df_portscan = pd.read_csv(
    "./Trafego/PortScan.csv",
    skiprows=range(1, 87794),
    nrows=linhas_para_ler
)
df_normal = pd.read_csv(
    "./Trafego/Normal.csv",
    skiprows=range(1, 2),
    nrows=linhas_para_ler
)
df_portscan = df_portscan.iloc[87794:275048]

# Unir datasets
df = pd.concat([df_normal, df_portscan, df_ddos], ignore_index=True)
df.columns = df.columns.str.strip()

# ================================
# 2. Remover colunas categóricas
# ================================
drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
df = df.drop(columns=drop_cols, errors="ignore")

# ================================
# 3. Preparar rótulo
# ================================
df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)

# Balanceamento (downsample)
df_benign = df[df["Label"] == 0]
df_attack = df[df["Label"] == 1]
min_count = min(len(df_benign), len(df_attack))
df_benign_bal = resample(df_benign, replace=False, n_samples=min_count, random_state=42)
df_attack_bal = resample(df_attack, replace=False, n_samples=min_count, random_state=42)
df = pd.concat([df_benign_bal, df_attack_bal], ignore_index=True)

# ================================
# 4. Separar X e y
# ================================
# (mantém remoção das janelas TCP como você já fazia)
X = df.drop(columns=["Label", "Init_Win_bytes_backward", "Init_Win_bytes_forward"], errors="ignore")
y = df["Label"]

# ================================
# 5. Coagir numérico, tratar inf/NaN e exportar amostra
# ================================
X_num = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

# ================================
# 6. Split treino/teste (estratificado)
#    -> Selector será AJUSTADO só no TREINO (evita vazamento)
# ================================
X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_num, y, test_size=0.3, random_state=42, stratify=y
)

# ================================
# 7. Seleção de features (fit no train)
# ================================
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_train_num, y_train)

selected_mask = selector.get_support()
selected_features = X_num.columns[selected_mask].tolist()
print("📌 Features selecionadas:", selected_features)

# Transformar conjuntos
X_train = selector.transform(X_train_num)
X_test  = selector.transform(X_test_num)

# Persistir selector e features
joblib.dump(selector, os.path.join(run_dir, "selector.joblib"))
with open(os.path.join(run_dir, "selected_features.json"), "w") as f:
    json.dump(selected_features, f, indent=2)
print("✅ Selector e lista de features salvos.")

# ================================
# 8. Definição dos modelos a comparar
# ================================
modelos = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, max_depth=5,  random_state=42, class_weight="balanced"
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=100, max_depth=5,  random_state=42, class_weight="balanced"
    )
}

def get_proba(model, Xmat):
    """Tenta predict_proba; senão, normaliza decision_function; fallback: predição binária."""
    try:
        return model.predict_proba(Xmat)[:, 1]
    except AttributeError:
        if hasattr(model, "decision_function"):
            s = model.decision_function(Xmat)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else np.zeros_like(s)
        return model.predict(Xmat)

# ================================
# 9. Treinar, avaliar, SALVAR todos os modelos
# ================================
resultados = []
registry = {"run_dir": run_dir, "models": {}, "selected_features": selected_features}

plt.figure(figsize=(9, 7))

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_score = get_proba(modelo, X_test)

    # Métricas principais
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    rocau = roc_auc_score(y_test, y_score)
    prau  = average_precision_score(y_test, y_score)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, lw=1.6, label=f"{nome} (AUC={rocau:.3f})")

    # Logs
    print(f"\n=== {nome} ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Matriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Acumula resultados
    resultados.append({
        "modelo": nome, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": rocau, "pr_auc": prau
    })

    # Salvar modelo treinado (este é o candidato congelado do experimento)
    model_path = os.path.join(models_dir, f"{nome}.joblib")
    joblib.dump(modelo, model_path)

    # Salvar artefatos do modelo
    registry["models"][nome] = {
        "path": model_path,
        "metrics": {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": rocau, "pr_auc": prau
        },
        "confusion_matrix": cm.tolist()
    }

# Plot ROC de todos
plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curvas ROC - Comparação de Modelos")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "roc_comparacao.png"), dpi=160)
plt.show()



# Persistir registry
with open(os.path.join(run_dir, "registry.json"), "w") as f:
    json.dump(registry, f, indent=2)
print("✅ Registry salvo (paths + métricas por modelo).")



# Re-salva selector para empacotar junto do best
selector_path = os.path.join(run_dir, "selector.joblib")
joblib.dump(selector, selector_path)
print(f"✅ Selector salvo em '{selector_path}'")

print(f"\n📁 Artefatos desta execução em: {run_dir}")
