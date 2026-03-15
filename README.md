# 📈 Behavior Score CMV - Painel de Análise e Decisão

[![Live Badge](https://img.shields.io/badge/-Live-2B5482?style=flat-square&logo=streamlit&logoColor=fff)](https://behavior-score-app.streamlit.app/)
[![Projeto Badge](https://img.shields.io/badge/-project--core-2B5482?style=flat-square&logo=github&logoColor=fff)](https://github.com/rafa-trindade/hackathon-pod-squad3-core)
[![Projeto Badge](https://img.shields.io/badge/-project--ops-2B5482?style=flat-square&logo=github&logoColor=fff)](https://github.com/rafa-trindade/hackathon-pod-squad3-ops)

> Portfólio interativo do modelo de credit scoring comportamental desenvolvido no **Hackathon Empresa Telco + Oracle PoD Academy 2025–2026** (Squad 03).  
> Este repositório expõe apenas o painel Streamlit com uma amostra anonimizada dos dados - o projeto completo vive nos repositórios de core e ops abaixo.

---

## 🔗 Ecossistema do Projeto

| Repositório | Descrição |
|---|---|
| [`hackathon-pod-squad3-core`](https://github.com/rafa-trindade/hackathon-pod-squad3-core) |  Engine de processamento, arquitetura medalhão e gestão de performance com governança de dados nativa |
| [`hackathon-pod-squad3-ops`](https://github.com/rafa-trindade/hackathon-pod-squad3-ops) |  Infraestrutura como código (IaC), orquestração de pipelines e estratégias de Cloud Readiness |
| [`behavior-score-obs`](https://github.com/rafa-trindade/behavior-score-obs) | Painel de observabilidade de pipeline de dados: qualidade, integridade, profiling e FinOps |
| **[`behavior-score-app`](https://github.com/rafa-trindade/behavior-score-app)**  | **Painel interativo de análise de risco de crédito, simulação de política e motor de decisão** |


---

## 🎯 Contexto de Negócio

A Empresa Telco possui uma base de ~35 milhões de clientes **Pré-Pago**. O desafio é identificar quais desses clientes têm perfil para migrar para o plano **Controle** (pós-pago de entrada), expandindo receita com risco controlado.

O **Behavior Score CMV v1.0** é um modelo CatBoost que combina dados de bureau de crédito com comportamento interno de recarga, pagamento e atraso para gerar um score de 0 a 1000 - substituindo a régua legada de bureau e gerando valor incremental via swap de carteira.

---

## 📊 O que este painel mostra

### Análise
- **Visão Geral** - volumetria por safra, bad rate histórico e KPIs do modelo (Gini, KS, PSI)
- **Análise Demográfica** - segmentação por região, estado e faixa de score com mapa de risco por UF
- **Análise Univariada** - curva de risco e distribuição por classe para qualquer variável da ABT
- **Análise Multivariada** - matriz de correlação Spearman e heatmap de risco combinado entre variáveis
- **Ranking de Variáveis** - IV e Gini univariado para todas as features
- **Estabilidade Temporal** - PSI entre safras para detecção de drift

### Performance
- KPIs oficiais OOT: KS 34.53% / Gini 46.60% / PSI 0.0010
- Cards de Swap-In e Swap-Out sobre o Grupo Controle (N = 110.268)
- Evolução incremental do KS por bloco de variáveis adicionado ao modelo

### Impacto para o Negócio
- Motor de rentabilidade com projeção de EBITDA incremental por taxa de aprovação
- Comparativo Legado (bureau) vs Modelo em 7 cenários de cutoff
- Bridge financeiro: Upsell Líquido + PDD Protegida = EBITDA Incremental
- Extrapolação configurável para escala nacional (35M clientes)

### Decisão
- **Estratégia de Política** - otimizador multi-objetivo (crescimento × risco × eficiência), fronteira de Pareto, sweet spot automático por KS e clusterização K-Means com personas de negócio
- **Motor de Decisão** - escoragem individual por CPF com gauge de score, rating A–E, decisão Aprovado/Mesa/Reprovado e drivers de risco (feature importance top 8)

---

## 🏗️ Arquitetura do Projeto Original

```
OCI Object Storage (S3)
    └── Medallion: Raw → Bronze → Silver → Gold (ABT)
            ↓
    Apache Airflow (Orquestração)
            ↓
    DuckDB + httpfs (Consultas analíticas)
            ↓
    CatBoost (Treinamento + Validação OOT)
            ↓
    Artefato .pkl (modelo + features + metadata)
            ↓
    Streamlit (Painel de Análise e Decisão)
```

**Stack:** Python · CatBoost · DuckDB · Apache Airflow · Terraform · OCI · Streamlit · Plotly

---

## 📁 Estrutura do Repositório

```
behavior-score-app/
├── .streamlit/
│   └── config.toml                       # Tema e configurações do servidor
├── assets/
│   └── style.css                         # Estilos customizados
├── config/
│   └── data_connections.py               # Stub de conexão (S3 desativado: utiliza apenas amostra local)
├── data/
│   ├── sample_abt_model_features.parquet # Amostra anonimizada da ABT
│   └── base_escorada_swap_v1.parquet     # Base do grupo controle para análise de swap
├── models/
│   └── behavior_catboost_v1.pkl          # Artefato do modelo (features + metadata + model)
├── utils/
│   ├── utils.py                          # Funções de dados, plots e scoring
│   └── br_states.geojson                 # Geometria dos estados brasileiros
└── app.py                                # Aplicação principal Streamlit
```

---

## 📈 Resultados do Modelo (OOT - Fev/25 e Mar/25)

| Métrica | Resultado | Meta | Status |
|---|---|---|---|
| KS (Separação) | 34.53% | > 30% | ✅ |
| Gini (Discriminação) | 46.60% | > 40% | ✅ |
| PSI (Estabilidade) | 0.0010 | < 0.25 | ✅ |
| Swap-In Bad Rate | 33.18% | - | Risco absorvido com segurança |
| Swap-Out Bad Rate | 51.41% | - | Inadimplentes barrados |
| Bads evitados | 3.652 | - | FPDs diretos protegidos |

---

## 📝 Nota sobre os dados

Os dados exibidos neste painel são uma **amostra anonimizada** extraída da ABT de desenvolvimento. CPFs e informações pessoais identificáveis foram removidos ou mascarados. O modelo `.pkl` é o artefato oficial treinado sobre os dados completos da produção.

---

## 👤 Autor

**Rafael Araujo Trindade**  
Data & Analytics Engineer Professional  
[rafa-trindade.github.io](https://rafa-trindade.github.io) · [GitHub](https://github.com/rafa-trindade)