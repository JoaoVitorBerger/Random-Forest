# 🚀 Classificação de Tráfego de Rede (Treinamento + Validação)

Este projeto implementa um **pipeline completo de Machine Learning** para detectar ataques de rede em datasets de tráfego.  
O código está dividido em duas etapas principais:

1. **Treinamento (`train.py`)** → Treina modelos, gera métricas e salva artefatos.  
2. **Validação (`evaluate.py`)** → Usa os modelos já treinados para avaliar dados novos (holdout ou produção).

---

## 📂 Estrutura do Projeto



---

## ⚙️ Fluxo do Pipeline

### 🔹 1. Treinamento (`treinamento_modelo.py`)
- Lê datasets de tráfego:  
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`  
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`  
  - `Wednesday-workingHours.pcap_ISCX.csv`  
- Faz **pré-processamento**:  
  - Remove colunas irrelevantes (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`).  
  - Converte `Label` para binário (**0 = BENIGN**, **1 = ataque**).  
  - Balanceia benignos e ataques via **downsampling**.  
  - Trata valores inválidos (`NaN`, `inf`) substituindo por `0`.  
- Separa treino/teste (70%/30%, estratificado).  
- Seleciona **20 features mais relevantes** usando `SelectKBest (f_classif)`.  
- Treina dois modelos:  
  - `RandomForestClassifier`  
  - `ExtraTreesClassifier`  
- Avalia com métricas:  
  - `accuracy`, `precision`, `recall`, `f1`, `ROC AUC`, `PR AUC`.  
- Gera e salva:  
  - Modelos `.joblib`  
  - Seletor `.joblib`  
  - Lista de features `.json`  
  - Métricas `registry.json`  
  - Curva ROC `roc_comparacao.png`  

---

### 🔹 2. Validação (`rede.py`)
- Lê os artefatos salvos (`registry.json`, modelos `.joblib`, `selector.joblib`).  
- Carrega um dataset de teste (ex.: `Trafego/normal.csv`).  
- Aplica o **mesmo pré-processamento do treino**.  
- Gera predições e scores para cada modelo.  
- Se o dataset tiver coluna `Label`, calcula:  
  - classification report (precision, recall, f1, accuracy)  
  - matriz de confusão  
  - curva ROC comparativa  
- Salva resultados em:  
  - `avaliacao_holdout_metricas.csv`  
  - `avaliacao_holdout_predicoes.csv`  

---

## 📦 Dependências

As bibliotecas necessárias estão listadas em `requirements.txt`:

## ▶️ Como Executar

### 1. Criar ambiente virtual (opcional, mas recomendado)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac

pip install -r requirements.txt

python treinamento_modelo.py

### Edite essa variável com as runs que deseja
RUN_DIR = "runs/run_20250901_235809"

python rede.py