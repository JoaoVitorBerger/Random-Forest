import io, os, json
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.tree import plot_tree

# ================== CONFIG ==================
st.set_page_config(page_title="🛡️ Dashboard IDS - Random Forest", layout="wide")
st.title("🛡️ Dashboard IDS - Random Forest")

# ---------- Estado inicial (apenas 1 modelo) ----------
if "clf" not in st.session_state:
    st.session_state.clf = None           # sklearn estimator
if "selector" not in st.session_state:
    st.session_state.selector = None      # SelectKBest ou similar
if "results" not in st.session_state:
    st.session_state.results = None       # results.joblib (opcional)
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None # nomes das features selecionadas

# ---------- Helpers ----------
def load_joblib_from_uploader(file):
    raw = file.read()
    return joblib.load(io.BytesIO(raw))

def has_model_loaded():
    return st.session_state.clf is not None and st.session_state.selector is not None

def clean_like_training(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    return X

def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    if " Label" in df.columns:
        df[" Label"] = df[" Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
        df.rename(columns={" Label": "Label"}, inplace=True)
    elif "Label" in df.columns:
        df["Label"] = df["Label"].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)
    return df

def get_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else np.zeros_like(s)
        return model.predict(X)



menu = st.sidebar.radio("Navegação", ["Carregar Artefatos", "Testar Arquivo"])

# ========================
# Aba 1: Carregar Artefatos
# ========================
if menu == "Carregar Artefatos":
    st.subheader("📦 Carregar modelo, selector e (opcional) results")
    uploaded_model   = st.file_uploader("Modelo (.joblib)", type="joblib", key="u_model")
    uploaded_selector= st.file_uploader("Selector (.joblib)", type="joblib", key="u_selector")
    uploaded_results = st.file_uploader("Results (.joblib) — opcional", type="joblib", key="u_results")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Carregar e Persistir"):
            if not uploaded_model or not uploaded_selector:
                st.warning("Envie **modelo** e **selector**.")
            else:
                try:
                    st.session_state.clf = load_joblib_from_uploader(uploaded_model)
                    st.session_state.selector = load_joblib_from_uploader(uploaded_selector)
                    st.session_state.results = load_joblib_from_uploader(uploaded_results) if uploaded_results else None

                    # nomes de features (se o selector expõe)
                    try:
                        st.session_state.feature_names = st.session_state.selector.get_feature_names_out()
                    except Exception:
                        st.session_state.feature_names = None

                    st.success("✅ Artefatos carregados!")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
    with c2:
        if st.button("🧹 Limpar sessão"):
            st.session_state.clf = None
            st.session_state.selector = None
            st.session_state.results = None
            st.session_state.feature_names = None
            st.info("Sessão limpa.")

    if has_model_loaded():
        st.success("Modelo e selector ativos na sessão.")
        clf = st.session_state.clf
        selector = st.session_state.selector
        feature_names = st.session_state.feature_names

        st.subheader("⚙️ Parâmetros do Modelo")
        try:
            st.json(clf.get_params())
        except Exception:
            st.write("Parâmetros indisponíveis para este estimador.")

        # Importâncias (se houver)
        importances = getattr(clf, "feature_importances_", None)
        if importances is not None:
            st.subheader("🔥 Importância das Features")
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(len(importances))]
            feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15), ax=ax)
            st.pyplot(fig)

        # Uma árvore (se RF/ET têm estimators_)
        if hasattr(clf, "estimators_") and len(getattr(clf, "estimators_", [])) > 0:
            st.subheader("🌳 Visualização de uma Árvore")
            tree_id = st.slider("Índice da árvore", 0, len(clf.estimators_) - 1, 0)
            estimator = clf.estimators_[tree_id]
            fig, ax = plt.subplots(figsize=(20,10))
            plot_tree(
                estimator,
                feature_names=feature_names,
                class_names=["Benign", "Attack"],
                filled=True,
                fontsize=8,
                ax=ax
            )
            st.pyplot(fig)

        # Resultados do treino (se enviados)
        if st.session_state.results is not None:
            st.subheader("📊 Resultados salvos (treino/validação)")
            res = st.session_state.results
            try:
                st.write("**Matriz de Confusão (treino/val):**")
                cm = res.get("confusion_matrix")
                if cm is not None:
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
            except Exception:
                pass
            try:
                roc_auc = res.get("roc_auc", None)
                fpr = res.get("fpr", None)
                tpr = res.get("tpr", None)
                if roc_auc is not None and fpr is not None and tpr is not None:
                    st.write(f"**ROC AUC (treino/val):** {roc_auc:.4f}")
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0,1],[0,1],"k--")
                    ax.legend()
                    st.pyplot(fig)
            except Exception:
                pass
    else:
        st.info("Carregue o **modelo** e o **selector** e clique em *Carregar e Persistir*.")

# ========================
# Aba 2: Testar Arquivo
# ========================
if menu == "Testar Arquivo":
    st.header("📡 Testar Arquivo — Visão Geral + Avaliação")
    st.caption("Pré-processa como no treino, aplica o selector e avalia o **modelo carregado**.")

    if not has_model_loaded():
        st.error("⚠️ Carregue o **modelo** e o **selector** na aba *Carregar Artefatos*.")
    else:
        # Threshold manual (como você preferir)
        thr = st.sidebar.number_input("Threshold de classificação", 0.0, 1.0, 0.5, 0.01)

        uploaded_file = st.file_uploader("Envie um CSV de tráfego (ex.: Trafego/normal.csv)", type="csv", key="u_csv_eval")
        if uploaded_file:
            # Carrega CSV
            try:
                df_raw = pd.read_csv(uploaded_file, encoding="latin1")
            except Exception:
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file)
            df_raw.columns = df_raw.columns.str.strip()

            # Guardar colunas de contexto
            context_cols = [c for c in ["Flow ID", "Source IP", "Destination IP", "Timestamp"] if c in df_raw.columns]

            # Pré-processamento (igual treino)
            drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp",
                         "Init_Win_bytes_backward", "Init_Win_bytes_forward"]
            df_proc = df_raw.drop(columns=drop_cols, errors="ignore")
            df_proc = normalize_label_column(df_proc)

            y_true = df_proc["Label"].values if "Label" in df_proc.columns else None
            X = df_proc.drop(columns=["Label"], errors="ignore")
            X = clean_like_training(X)

            # Alinhar com o selector
            selector = st.session_state.selector
            missing = [c for c in selector.feature_names_in_ if c not in X.columns]
            if missing:
                st.error(f"As seguintes colunas esperadas pelo selector não estão no CSV: {missing}")
                st.stop()

            X = X[selector.feature_names_in_]
            X_sel = selector.transform(X)

            # Inferência
            clf = st.session_state.clf
            y_score = get_proba(clf, X_sel)
            y_pred = (y_score >= thr).astype(int)

            # ===== Métricas =====
            st.markdown("## 📈 Resultados de Classificação")
            if y_true is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
                c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
                c4.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")

                rep = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(rep).T, use_container_width=True)

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Matriz de Confusão")
                st.pyplot(fig)

                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_title("Curva ROC")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("CSV sem coluna 'Label' — exibindo apenas predições/scores.")

            # ===== Painéis de análise de rede =====
            st.markdown("---")
            st.markdown("## 🔎 Análise de Rede para Ação")

            df_view = df_raw.copy()
            df_view["Pred"]  = y_pred
            df_view["Score"] = y_score

            if "Timestamp" in df_view.columns:
                df_view["__ts"] = pd.to_datetime(df_view["Timestamp"], errors="coerce")
            else:
                df_view["__ts"] = pd.NaT

            HAS_SRC  = "Source IP" in df_view.columns
            HAS_DST  = "Destination IP" in df_view.columns
            HAS_DPT  = "Destination Port" in df_view.columns
            HAS_PROTO= "Protocol" in df_view.columns

            with st.expander("⚙️ Configurações de detecção / thresholds"):
                colA, colB, colC = st.columns(3)
                win_minutes = colA.slider("Janela p/ varredura (min)", 1, 60, 5)
                min_unique_ports = colB.slider("Mín. portas distintas p/ marcar scan", 5, 500, 50)
                min_unique_dsts  = colC.slider("Mín. destinos distintos p/ marcar scan", 5, 500, 50)
                colD, colE = st.columns(2)
                topn_rank = colD.slider("Top-N em rankings (IPs/Portas)", 5, 50, 10)
                score_cut = colE.slider("Corte de score p/ suspeito", 0.0, 1.0, float(thr), 0.05)

            sus_mask = (df_view["Pred"] == 1) & (df_view["Score"] >= score_cut)
            df_sus = df_view[sus_mask].copy()

            st.subheader("1) 🧭 Top origens, destinos e portas suspeitas")
            cols = st.columns(3)
            if HAS_SRC:
                top_src = df_sus["Source IP"].value_counts().head(topn_rank)
                cols[0].write("**Top Source IP (ataques)**")
                if not top_src.empty:
                    df_top_src = top_src.reset_index()
                    df_top_src.columns = ["Source IP", "Count"]
                    cols[0].bar_chart(df_top_src.set_index("Source IP"))
                else:
                    cols[0].info("Sem dados.")
            if HAS_DST:
                top_dst = df_sus["Destination IP"].value_counts().head(topn_rank)
                cols[1].write("**Top Destination IP (alvos)**")
                if not top_dst.empty:
                    df_top_dst = top_dst.reset_index()
                    df_top_dst.columns = ["Destination IP", "Count"]
                    cols[1].bar_chart(df_top_dst.set_index("Destination IP"))
                else:
                    cols[1].info("Sem dados.")
            if HAS_DPT:
                top_dport = df_sus["Destination Port"].value_counts().head(topn_rank)
                cols[2].write("**Top Destination Port (portas visadas)**")
                if not top_dport.empty:
                    df_top_dport = top_dport.reset_index()
                    df_top_dport.columns = ["Destination Port", "Count"]
                    cols[2].bar_chart(df_top_dport.set_index("Destination Port"))
                else:
                    cols[2].info("Sem dados.")

            st.subheader("2) ⏱️ Distribuição temporal de alertas")
            if df_view["__ts"].notna().any():
                df_time = df_sus.dropna(subset=["__ts"]).copy()
                if not df_time.empty:
                    df_time["Hora"] = df_time["__ts"].dt.hour
                    df_time["Dia"]  = df_time["__ts"].dt.date
                    c1, c2 = st.columns(2)
                    c1.write("**Alertas por hora do dia**")
                    c1.bar_chart(df_time["Hora"].value_counts().sort_index())
                    c2.write("**Alertas por dia (tendência)**")
                    c2.line_chart(df_time.groupby("Dia").size())
                else:
                    st.info("Sem timestamps válidos nos eventos suspeitos.")
            else:
                st.info("Coluna Timestamp ausente ou inválida.")

            st.subheader("3) 📦 Análise por protocolo")
            if HAS_PROTO:
                proto_counts = df_sus["Protocol"].value_counts()
                if not proto_counts.empty:
                    df_proto = proto_counts.head(topn_rank).reset_index()
                    df_proto.columns = ["Protocol", "Count"]
                    st.bar_chart(df_proto.set_index("Protocol"))
                else:
                    st.info("Sem protocolos.")
            else:
                st.info("Coluna 'Protocol' não encontrada no CSV.")

            st.subheader("4) 🔦 Detecção de varredura (heurística por janela)")
            if HAS_SRC and (HAS_DPT or HAS_DST) and df_view["__ts"].notna().any():
                df_win = df_view.dropna(subset=["__ts"]).copy()
                df_win["__bucket"] = df_win["__ts"].dt.floor(f"{win_minutes}min")

                agg = {"__ts": "count"}
                if HAS_DPT: agg["Destination Port"] = pd.Series.nunique
                if HAS_DST: agg["Destination IP"] = pd.Series.nunique

                scan = (df_win.groupby(["Source IP", "__bucket"])
                            .agg(agg)
                            .rename(columns={"__ts":"events","Destination Port":"unique_ports","Destination IP":"unique_dsts"})
                            .reset_index())

                if "unique_ports" not in scan.columns: scan["unique_ports"] = 0
                if "unique_dsts" not in scan.columns: scan["unique_dsts"] = 0

                scan["scan_flag"] = (scan["unique_ports"] >= min_unique_ports) | (scan["unique_dsts"] >= min_unique_dsts)

                scan_ip = (scan.groupby("Source IP")
                               .agg(buckets_suspeitos=("scan_flag","sum"),
                                    max_unique_ports=("unique_ports","max"),
                                    max_unique_dsts=("unique_dsts","max"),
                                    total_eventos=("events","sum"))
                               .sort_values(["buckets_suspeitos","max_unique_ports","max_unique_dsts"], ascending=False)
                               .head(topn_rank))
                st.write("**IPs com comportamento de varredura (top):**")
                st.dataframe(scan_ip, use_container_width=True)
            else:
                st.info("Necessário: Source IP, Timestamp e (Destination Port ou Destination IP).")

            st.subheader("5) 🔥 Mapa de calor: Source IP × Destination Port (suspeitos)")
            if HAS_SRC and HAS_DPT:
                top_src_list  = df_sus["Source IP"].value_counts().head(topn_rank).index.tolist()
                top_port_list = df_sus["Destination Port"].value_counts().head(topn_rank).index.tolist()
                piv = (df_sus[df_sus["Source IP"].isin(top_src_list) & df_sus["Destination Port"].isin(top_port_list)]
                       .pivot_table(index="Source IP", columns="Destination Port", values="Pred", aggfunc="count", fill_value=0))
                if not piv.empty:
                    fig, ax = plt.subplots(figsize=(min(12, 2+0.5*len(top_port_list)), max(4, 0.5*len(top_src_list))))
                    sns.heatmap(piv, annot=False, cmap="Reds", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Sem dados suficientes p/ heatmap.")
            else:
                st.info("Para o heatmap: Source IP e Destination Port.")

            st.subheader("6) ⛔ Exportar blocklist (IPs suspeitos)")
            if HAS_SRC:
                grp = df_sus.groupby("Source IP").agg(
                    total_eventos=("Pred", "size"),
                    score_medio=("Score", "mean"),
                    primeiro_evento=("__ts", "min"),
                    ultimo_evento=("__ts", "max"),
                )

                def _top_k_join(series, k=5):
                    s = pd.Series(series).dropna()
                    if s.empty: return ""
                    top = s.value_counts().head(k).index.tolist()
                    return ",".join(map(str, top))

                if HAS_DPT:
                    grp["top_dest_ports"] = df_sus.groupby("Source IP")["Destination Port"].apply(lambda s: _top_k_join(s, k=5))

                grp = grp.sort_values(["total_eventos", "score_medio"], ascending=False)
                if grp["primeiro_evento"].notna().any():
                    grp["primeiro_evento"] = grp["primeiro_evento"].astype(str)
                if grp["ultimo_evento"].notna().any():
                    grp["ultimo_evento"] = grp["ultimo_evento"].astype(str)

                csv_bytes = grp.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button("💾 Baixar blocklist (CSV)", data=csv_bytes, file_name="blocklist_ips_suspeitos.csv", mime="text/csv")
                st.dataframe(grp.head(50), use_container_width=True)
            else:
                st.info("Coluna 'Source IP' não encontrada — não é possível gerar a blocklist.")
