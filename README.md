# 🏎️ Progetto Guida Autonoma F1 - Tracciato di Monza (CARLA Simulator)

Progetto per l'esame di **Fondamenti di Intelligenza Artificiale** (3° Anno - 1° Semestre).
Il codice implementa la gestione del tracciato di Monza in ambiente di simulazione CARLA, applicando l'algoritmo di Reinforcement Learning Multi-Agente **MAPPO** per il controllo dinamico e competitivo di due vetture da corsa.

---

## 🧠 Modello RL: Multi-Agent Proximal Policy Optimization (MAPPO)

Il sistema adotta un approccio **Centralized Training with Decentralized Execution (CTDE)**, ideale per la gestione di scenari competitivi e cooperativi di guida autonoma:

### 1. Actor (Decentralizzato - `src/models.py`)

Ogni veicolo agisce come un agente autonomo guidato da una rete neurale locale:

- **Input (Stato Locale - $18$ dimensioni)**:
  - $16$ letture raycast normalizzate per il rilevamento dinamico degli ostacoli e dei limiti fisici del tracciato (con una portata differenziata da $15\text{m}$ a $110\text{m}$).
  - Velocità lineare dell'auto normalizzata rispetto al limite di riferimento di $v_{max} = 70.0\text{ m/s}$.
  - Angolo di deviazione relativo (normalizzato nell'intervallo $[-1.0, 1.0]$) rispetto alla direzione tangente del waypoint della pista.
- **Output (Spazio delle Azioni - $2$ dimensioni):**
  - Controllo continuo dello sterzo (intervallo $[-1.0, 1.0]$).
  - Controllo continuo della trazione (acceleratore/freno integrato nell'intervallo $[-1.0, 1.0]$).

### 2. Critic (Centralizzato - `src/models.py`)

Utilizzato esclusivamente durante la fase off-line di apprendimento per mitigare i problemi di non-stazionarietà dell'ambiente multi-agente:

- **Input (Stato Globale - $36$ dimensioni):** Concatena gli stati locali di tutti gli agenti attivi nella simulazione ($18\text{ input} \times 2\text{ veicoli}$).
- **Output (Valore dello Stato):** Stima del valore dello stato globale per il calcolo del vantaggio temporale ($GAE$) di ciascun agente.

---

## 🛠️ Requisiti di Sistema e Prerequisiti

Per garantire la compatibilità con l'API di CARLA 0.9.12 (fornita tramite file `.egg`), il progetto richiede:

- **Python 3.7.x** (Consigliato Python 3.7.9 a 64-bit)
- **CARLA Simulator 0.9.12**
- **PyTorch** (con supporto CUDA consigliato per l'accelerazione GPU)

---

## 📂 Struttura del Progetto

```text
F1A/
├── config/
│   └── config.py                     # Costanti di sistema, configurazione sensori e iperparametri
├── src/
│   ├── __init__.py
│   ├── actor.pth                     # (Generato) Pesi correnti salvati per la rete Actor
│   ├── buffer.py                     # Accumulo delle transazioni locali e globali per l'addestramento
│   ├── camera.py                     # Gestione dello spectator live e registrazione video telemetrico
│   ├── connection.py                 # Gestione della connessione sincrona con il server CARLA
│   ├── critic.pth                    # (Generato) Pesi correnti salvati per la rete Critic
│   ├── environment.py                # Gestione fisica dell'ambiente, spawning e raycasting geometrico
│   ├── logger.py                     # Salvataggio e monitoraggio delle metriche di addestramento su CSV
│   ├── models.py                     # Classi PyTorch per le reti neurali Actor-Critic
│   ├── mappo.py                      # Algoritmo MAPPO, calcolo del GAE e ottimizzazione reti
│   ├── pth-history/                  # (Generata) Storico dei checkpoint salvati periodicamente
│   ├── reward.py                     # Calcolo della ricompensa basata su stabilità, velocità e traiettoria
│   ├── train.py                      # Entry-point per l'addestramento
│   ├── training_log.csv              # (Generato) Telemetria ed esportazione dei dati di addestramento
│   └── video_addestramento/          # (Generato) Cartella per il salvataggio automatico dei video degli episodi
├── .gitignore
├── carla-0.9.12-py3.7-win-amd64.egg  # File API di CARLA (da incollare qui)
├── demo.py                           # Entry-point per la valutazione e la demo
├── Monza.npz                         # Dataset geometrico dei waypoint del circuito (da incollare qui)
├── README.md
└── requirements.txt                  # Pacchetti necessari per il progetto
```

## 🚀 Installazione e Configurazione

### 1. Download degli Asset e del Simulatore

Il progetto richiede il simulatore CARLA e i file specifici del tracciato di Monza:

- **CARLA 0.9.12:** Scarica la release ufficiale [da questo link](https://github.com/carla-simulator/carla/releases/tag/0.9.12).
- **Pacchetto Mappa di Monza:** Scarica lo zip della mappa (che include il file dei waypoint) [da questo link](https://roar.berkeley.edu/monza-map/).
  - _Nota di posizionamento:_ Estrai il file `Monza.npz` dallo zip e incollalo **all'interno della cartella radice del progetto (`F1A/`)**, esattamente accanto al file `README.md` e alla cartella `src/`.

### 2. Installazione delle dipendenze Python

Apri il terminale nella cartella del progetto (`F1A`) utilizzando l'interprete **Python 3.7** e digita:

```bash
py -3.7 -m pip install -r requirements.txt
```

### 3. Come Eseguire il Progetto

Il progetto richiede l'esecuzione in due passaggi: prima l'avvio del simulatore (Server) e successivamente l'avvio dello script di controllo (Client).

#### Passo 1: Avviare il Server CARLA

1. Apri il terminale del tuo computer (Prompt dei comandi o PowerShell).
2. Naviga fino alla cartella in cui hai installato CARLA 0.9.12.
3. Avvia il simulatore eseguendo il file principale (puoi usare i flag `-quality_level=Low`, `-dx12` o `-RenderOffScreen` se necessario):
   ```bash
   CarlaUE4.exe
   ```
4. Assicurati che il simulatore sia avviato correttamente e rimanga in esecuzione in background.

#### Passo 2A: Come Eseguire l'Addestramento (`train.py`)

1. Apri un secondo terminale sul tuo computer.
2. Naviga fino alla cartella radice del progetto:
   ```bash
   cd percorso/della/tua/cartella/F1A
   ```
3. Avvia lo script `train.py` assicurandoti di usare l'eseguibile di Python 3.7:
   ```bash
   python src/train.py
   ```
   Nota: Se hai più versioni di Python installate, usa il percorso assoluto o il comando specifico, ad esempio `py -3.7 src/train.py`

📋 Output atteso nel terminale:

```text
📦 File CARLA .egg caricato da: .../PythonAPI/carla/dist/carla-0.9.12-py3.7-win-amd64.egg
🚀 SCRIPT AVVIATO CON SUCCESSO CON PYTHON 3.7!
🗺️ Waypoint di Monza caricati correttamente! (2775 punti trovati, Lunghezza: 5617.05m)
WARNING: Version mismatch detected: You are trying to connect to a simulator that might be incompatible with this API
WARNING: Client API version     = 0.9.12
WARNING: Simulator API version  = 0.9.12-dirty
✅ Client collegato a CARLA con successo! (Modalità sincrona attiva)
🚗 Tutto pronto!
```

#### Passo 2B: Come Eseguire la Demo (`demo.py`)

Se vuoi osservare le prestazioni del modello addestrato senza sovrascrivere i pesi, assicurati di avere il Server CARLA in esecuzione e lancia lo script di valutazione.

- Nota: Questo script richiede che i file dei pesi addestrati (actor.pth e critic.pth) siano presenti nelle directory previste.

1. Dal terminale posizionato nella radice del progetto (`F1A`), esegui:

   ```bash
   python demo.py
   ```

   Nota: Se hai più versioni di Python installate, usa il percorso assoluto o il comando specifico, ad esempio `py -3.7 demo.py`

## 🔄 Ripresa dell'Addestramento

Il sistema gestisce automaticamente la ripresa dell'addestramento:

- Se sono presenti i file di checkpoint `actor.pth` e `critic.pth`, i pesi delle reti neurali verranno caricati automaticamente.

- Il `TrainingLogger` rileverà l'ultimo episodio registrato nel file `training_log.csv` e proseguirà con la numerazione corretta dei dati.

## 📹 Registrazione Video

Durante l'addestramento, il sistema può salvare automaticamente i video degli episodi significativi all'interno della cartella `video_addestramento/`:

- **Salvataggio automatico**: I video vengono generati quando un episodio supera la soglia specificata (`SOGLIA_STEP_SALVATAGGIO` in `config.py`).

- **Formato e Parametri**: I video vengono renderizzati secondo la risoluzione (`FRAME_W`, `FRAME_H`) e il framerate (`VIDEO_FPS`) configurati.

## 📈 Logging e Monitoraggio

Durante l'addestramento, tutte le metriche chiave dell'episodio vengono salvate progressivamente nel file training_log.csv:

- **Prestazioni di Guida**: Velocità media e massima (km/h), distanza totale percorsa (m), percentuale di progresso sul tracciato.

- **Competizione**: Giri completati, numero di sorpassi, tempo trascorso in testa, tempo sul miglior giro.

- **Sicurezza e Condizioni di Fine**: Distanza minima registrata tra i veicoli e causa della conclusione dell'episodio (`reason`: collisione tra auto, impatto contro limiti della pista, inattività `still` o limite massimo di step `max_steps`).

- **Ottimizzazione**: Loss dell'Actor (Policy Loss) e Loss del Critic (Value Loss) ad ogni step di aggiornamento.
