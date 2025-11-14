# 🧠 Projeto IoT + Machine Learning

Integração entre **Node-RED**, **Mosquitto (MQTT)** e **Python** para coleta de dados, geração de dataset e treinamento automatizado de modelos de aprendizado de máquina.

---

## 🎯 Objetivos

- Criar um pipeline automatizado de coleta e processamento de dados de sensores (ou simulações).  
- Treinar modelos de Machine Learning a partir de datasets gerados pelo Node-RED.  
- Armazenar e reutilizar os modelos `.pkl` em execuções futuras.  
- Facilitar a execução e integração via containers Docker.

---

## 📁 Estrutura de Pastas

```
iot-project/
├── docker-compose.yml         # Orquestra todos os serviços
│
├── shared/                    # Pasta compartilhada entre Node-RED e Python
│   ├── dataset.csv            # Dataset gerado pelo Node-RED
│   └── models/                # Modelos treinados (.pkl)
│
├── mosquitto/
│   ├── data/                  # Configurações e fluxos persistentes do Node-RED
|   └── mosquitto.conf         # Configuração mosquitto
|   ├── certs/                 # Certificados
|
├── nodered/
│   └── data/                  # Configurações e fluxos persistentes do Node-RED
|       └── flows.json         # fluxos 
│
└── python/
    ├── treino.py              # Script de treinamento de Machine Learning
    └── Dockerfile             # Configuração da imagem Python
    ├── plots/                 # Gerar gráficos e novos models
```

---

## ⚙️ Pré-requisitos

- Docker e Docker Compose instalados  
- Porta `1883` (MQTT) e `1880` (Node-RED) disponíveis  

---

## 🚀 Como Executar

1. **Clone o projeto:**
   ```bash
   git clone https://github.com/brenoroosevelt/iot-project.git
   cd iot-project
   ```

2. **Suba os serviços:**
   ```bash
   docker-compose up --build -d
   ```

   Isso iniciará:
   - Mosquitto (broker MQTT)
   - Node-RED (interface visual) em [http://localhost:1880](http://localhost:1880)

3. **Gere o dataset:**
   - No Node-RED, o fluxo grava os dados em `/shared/dataset.csv`.

4. **Execute o treinamento:**
   ```bash
   docker-compose run trainer
   ```

5. **Verifique as saídas:**
   - O arquivo `dataset.csv` ficará em `./shared/`.
   - Os modelos treinados serão salvos em `./shared/models/`.

6. **Gerando Gráficos**
   ```bash
   docker run --rm   -v "$(pwd)/python:/app"   -w /app   python:3.10-slim   bash -c "pip install matplotlib pandas && python plot_results.py"
   ```
---

## 🧾 Licença

Projeto de uso livre para fins acadêmicos, de pesquisa e desenvolvimento experimental.

