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
|
├── nodered/
│   └── data/                  # Configurações e fluxos persistentes do Node-RED
│
└── python/
    ├── treino.py              # Script de treinamento de Machine Learning
    └── Dockerfile             # Configuração da imagem Python
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
   - No Node-RED, o fluxo grava os dados em `/data/shared/dataset.csv`.

4. **Execute o treinamento:**
   ```bash
   docker-compose run trainer
   ```

5. **Verifique as saídas:**
   - O arquivo `dataset.csv` ficará em `./shared/`.
   - Os modelos treinados serão salvos em `./shared/models/`.

---

## 🧩 Próximos Passos

- Integrar o treinamento automático a partir de eventos MQTT.  
- Expor previsões por API HTTP no Node-RED.  
- Adicionar logs e persistência de métricas de desempenho.

---

## 🧾 Licença

Projeto de uso livre para fins acadêmicos, de pesquisa e desenvolvimento experimental.

---

## 👨‍💻 Autor

Desenvolvido por **[Seu Nome / UFMS]**  
Ambiente experimental de integração **IoT + Machine Learning**.
