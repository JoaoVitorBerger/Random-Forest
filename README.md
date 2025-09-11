# 🛡️ Dashboard IDS - Random Forest

Este projeto implementa um **Dashboard interativo em Streamlit** para **análise e avaliação de modelos de detecção de intrusões (IDS)** treinados com algoritmos de aprendizado de máquina (ex.: Random Forest, Extra Trees).  

O objetivo é oferecer uma interface visual para:
- Carregar modelos já treinados e seus seletores de features.
- Avaliar novos arquivos de tráfego de rede (CSV).
- Gerar métricas de classificação e curvas ROC.
- Visualizar insights sobre IPs, portas, protocolos e comportamentos suspeitos.
- Exportar blocklists de IPs maliciosos detectados.

---

## 📂 Estrutura do Projeto

---

## ⚙️ Funcionalidades do Dashboard

### 🔹 1. Carregar Artefatos
- Upload de:
  - Modelo treinado (`.joblib`).
  - Selector de features (`.joblib`).
  - Resultados opcionais (`results.joblib`).
- Visualização:
  - Parâmetros do modelo.
  - Importância das features.
  - Árvore de decisão individual (se modelo for RandomForest/ExtraTrees).
  - Métricas e curva ROC do treino/validação (se fornecidas).

---

### 🔹 2. Testar Arquivo
- Upload de um CSV com tráfego de rede (ex.: `Trafego/normal.csv`).
- Pré-processamento automático:
  - Remoção de colunas irrelevantes (`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, etc.).
  - Normalização da coluna `Label` (se existir: BENIGN → 0, Attack → 1).
  - Conversão para numérico, tratamento de `NaN`/`inf`.
  - Aplicação do **selector** salvo no treinamento.
- Inferência:
  - Predição das classes (Benign/Attack).
  - Cálculo de **scores** (probabilidade de ataque).
  - Threshold configurável (default: 0.5).

#### 🔎 Outputs
- Métricas de classificação (accuracy, precision, recall, f1).
- Classification report detalhado.
- Matriz de confusão.
- Curva ROC com AUC.
- Painéis de análise de tráfego:
  - **Top Source/Destination IPs**.
  - **Top portas alvo**.
  - **Distribuição temporal dos ataques**.
  - **Distribuição por protocolo**.
  - **Detecção de varredura** (IPs com tentativas em múltiplas portas/destinos).
  - **Mapa de calor (Source IP × Destination Port)**.
- Exportação de **blocklist** com IPs suspeitos em CSV.

---

## 📦 Dependências

As bibliotecas necessárias estão listadas em `requirements.txt`:


## Instale as dependências do projeto
pip install -r requirements.txt

## Execute o servidor localmente
streamlit run analise.py
