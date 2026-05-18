# Pipeline de Comparação de Modelos para Imputação de Dados Meteorológicos

## Sobre o Projeto

Este pipeline implementa uma solução de pesquisa para avaliar e comparar a eficiência de diversos modelos de machine learning na **imputação de dados meteorológicos faltantes**. O estudo abrange dados de **96 estações meteorológicas localizadas na região Centro-Oeste do Brasil**.

### Objetivo

Comparar o desempenho de modelos tradicionais (baselines) e modelos deep learning (SAITS, ImputeFormer, Transformer) na tarefa de imputação de dados meteorológicos, considerando múltiplos cenários de falta de dados.

---

## Dados

### Fonte INMET (Instituto Nacional de Meteorologia)

- Dados meteorológicos observados das 96 estações
- Variáveis: Temperatura, Umidade, Chuva, Pressão, Radiação Global, Ponto de Orvalho, Velocidade do Vento
- **Disponibilidade:** [https://www.gov.br/inmet/pt-br](https://www.gov.br/inmet/pt-br)

### Fonte ERA5-Land (Reanalysis Data)

- Dados de reanalise obtidos via **API oficial do Copernicus Climate Data Store**
- Variáveis complementares para feature engineering
- **Como obter:** [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

---

## Modelos Avaliados

### Baselines

- **Média:** Imputação pela mediana dos dados de treinamento
- **LOCF** (Last Observation Carried Forward)
- **LOCF + Mediana:** Combinação de LOCF com mediana
- **KNN:** K-Nearest Neighbors Imputer
- **MissForest:** Random Forest-based imputation
- **XGBoost:** Gradient Boosting por target

### Deep Learning

- **SAITS:** Self-Attention-based Imputation for Time Series
- **ImputeFormer:** Transformer-based imputation
- **Transformer:** Architecture padrão para séries temporais

---

## Estrutura do Projeto

```
.
├── config/
│   └── experimento_base.json      # Configuração central dos experimentos
├── src/
│   ├── models/                     # Implementações dos modelos deep
│   │   ├── SAITS.py
│   │   ├── ImputeFormer.py
│   │   └── Transformer.py
│   ├── baselines.py                # Implementações dos baselines
│   ├── metrics.py                  # Cálculo de métricas
│   ├── tune_parametrs.py           # Tuning de hiperparâmetros
│   └── utils.py                    # Utilitários gerais
├── run_models.py                   # Script principal (execução direta)
├── launcher_multigpu.py            # Orquestrador multi-GPU
└── README.md
```

---

## Cenários de Teste

O pipeline executa os modelos em diferentes cenários de falta de dados (MCAR - Missing Completely At Random):

- **Base:** 20% de dados faltantes (padrão na validação)
- **Variações:** 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%

Cada cenário é avaliado com janelas temporais de 72h (3 dias) para capturar padrões de curto prazo.

---

## Instalação e Preparação

### 1. Dependências

```bash
pip install -r requirements.txt
```

### 2. Dados

Coloque o arquivo de dados em:

```
data/raw/original.parquet
```

O arquivo deve conter colunas conforme especificado em `config/experimento_base.json`.

---

## Execução

### Ambiente Local (GPU única ou CPU)

#### Executar um cenário com um modelo específico (SAITS)

```bash
python run_models.py \
  --action scenario \
  --scenario base \
  --config config/experimento_base.json \
  --model_group all \
  --run_saits true \
  --run_imputeformer false \
  --run_transformer false \
  --run_baselines false
```

#### Executar um cenário com todos os modelos deep

```bash
python run_models.py \
  --action scenario \
  --scenario base \
  --config config/experimento_base.json \
  --model_group all \
  --run_saits true \
  --run_imputeformer true \
  --run_transformer true \
  --run_baselines false
```

#### Executar tuning local de XGBoost

```bash
python run_models.py \
  --action tune \
  --config config/experimento_base.json \
  --model_group baselines \
  --run_baselines true
```

#### Executar todos os cenários com XGBoost

```bash
python run_models.py \
  --action run_all \
  --config config/experimento_base.json \
  --model_group baselines \
  --run_baselines true
```

---

### Ambiente Multi-GPU (Servidor com múltiplas GPUs)

O script `launcher_multigpu.py` distribui as tarefas entre GPUs disponíveis, executando em paralelo.

#### Executar apenas SAITS em todos os cenários

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model saits \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar apenas ImputeFormer em todos os cenários

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model imputeformer \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar apenas Transformer em todos os cenários

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model transformer \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar todos os 3 modelos deep em todos os cenários

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model both \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar com Tuning + Cenários (SAITS)

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model saits \
  --run_tuning_phase \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar com Tuning + Cenários (todos os deep)

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model both \
  --run_tuning_phase \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar Deep + Baselines com Tuning

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model both \
  --run_baselines \
  --run_tuning_phase \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar Deep + Baselines (sem tuning)

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model both \
  --run_baselines \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar apenas Baselines em todos os cenários

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model baselines \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

#### Executar Baselines com Tuning

```bash
python launcher_multigpu.py \
  --gpus 0,1,2,3,4 \
  --only_model baselines \
  --run_tuning_phase \
  --root_dir /caminho/do/projeto \
  --config config/experimento_base.json
```

---

## Consolidação de Resultados

Após executar os experimentos, consolidar os resultados manualmente:

```bash
python run_models.py \
  --action consolidate \
  --config config/experimento_base.json \
  --out_dir outputs/main_experiment
```

---

## Configuração

O arquivo `config/experimento_base.json` controla todos os parâmetros do experimento:

- `seed`: Reprodutibilidade (padrão: 42)
- `arquivo_entrada`: Caminho para dados brutos
- `pasta_saida`: Diretório de saída dos resultados
- `target_cols`: Variáveis meteorológicas a imputar
- `feature_cols`: Features de entrada (INMET + ERA5-Land)
- `window_size`: Tamanho da janela temporal (padrão: 72h)
- `MASKARAMENTO_ALEATORIO_DOS_CENARIOS`: Proporções de missing data a testar
- `physical_limits`: Limites físicos para validação dos dados

### Exemplo: Executar apenas XGBoost

Edite `config/experimento_base.json`:

```json
{
  "run_saits": false,
  "run_imputeformer": false,
  "run_transformer": false,
  "run_xgb_per_target_baseline": true,
  "run_median_baseline": false,
  "run_locf_baseline": false
}
```

---

## Saídas

Os resultados são salvos em `outputs/main_experiment/` e incluem:

- Previsões de imputação por cenário
- Métricas de desempenho (MAE, RMSE, MAPE, etc.)
- Relatórios consolidados em CSV
- Logs de execução

---

## Notas Importantes

- **GPU Multi:** Os comandos com `launcher_multigpu.py` distribuem tarefas entre GPUs. Ideal para servidor com múltiplas GPUs.
- **GPU Única/CPU:** Use `run_models.py` para testes locais ou ambiente com recurso limitado.
- **Tempo de Execução:** Experimentos completos podem levar horas dependendo da configuração.
- **Reprodutibilidade:** O seed é fixado para garantir resultados reproduzíveis.
