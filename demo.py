import glob
import sys
import os
import torch
import numpy as np
import math
import time

try:
    import keyboard
except ImportError:
    keyboard = None

# --- Setup Python 3.7 & EGG CARLA ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Se demo.py è direttamente dentro F1A:
project_root = current_dir
# Se demo.py si trovasse dentro F1A/src, risale di uno
if os.path.basename(current_dir) == "src":
  project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
  sys.path.insert(0, project_root)

# Cerca l'.egg di CARLA nella cartella corrente o sotto-cartelle
egg_files = glob.glob(os.path.join(project_root, "*.egg")) + glob.glob(
    os.path.join(project_root, "src", "*.egg")
)

if egg_files:
  sys.path.insert(0, os.path.abspath(egg_files[0]))
  print(f"📦 File CARLA .egg caricato da: {egg_files[0]}")
else:
  print(f"⚠️ ATTENZIONE: Nessun file .egg trovato nella radice ({project_root})")

# Controlla se la versione principale è 3 e la secondaria è 7s
if sys.version_info.major != 3 or sys.version_info.minor != 7:
    sys.exit(f"❌ Errore: Questo script richiede rigorosamente Python 3.7 per caricare l'.egg di CARLA 0.9.12.\n"
            f"Attualmente stai usando Python {sys.version_info.major}.{sys.version_info.minor}.\n")

print("🚀 SCRIPT AVVIATO CON SUCCESSO CON PYTHON 3.7!")

import carla
from config.config import STATE_DIM, ACTION_DIM, MAX_VELOCITY
from src.connection import connect_to_carla
from src.environment import reset_environment, get_state
from src.models import Actor

# Cartella dei modelli (pth-history)
DIR_MODELS = os.path.join(project_root, "src", "pth-history")

# Specifica qui il numero di step desiderato per ciascun agente
STEPS_ACTOR_0 = 50000    # Modello Agente 0 (Rosso) (Valori: 50000, 100000, ecc.)
STEPS_ACTOR_1 = 1000000    # Modello Agente 1 (Turchese) (Valori: 50000, 100000, ecc.)

# Durata massima
DEMO_MAX_STEPS = 4000

def apply_initial_velocity(vehicle, speed_kmh=40.0):
    """ Impone una spinta iniziale al veicolo """
    if vehicle and vehicle.is_alive:
        speed_mps = speed_kmh / 3.6
        fwd = vehicle.get_transform().get_forward_vector()
        target_vel = carla.Vector3D(x=fwd.x * speed_mps, y=fwd.y * speed_mps, z=fwd.z * speed_mps)
        vehicle.set_target_velocity(target_vel)

def update_spectator_view(spectator, vehicle):
    """ Posiziona lo spectator in inseguimento dietro al veicolo selezionato """
    if vehicle and vehicle.is_alive:
        try:
            t_trans = vehicle.get_transform()
            t_loc = t_trans.location
            t_rot = t_trans.rotation

            yaw_rad = math.radians(t_rot.yaw)
            sp_x = t_loc.x - 12.0 * math.cos(yaw_rad)
            sp_y = t_loc.y - 12.0 * math.sin(yaw_rad)
            sp_z = t_loc.z + 5.0

            spectator.set_transform(carla.Transform(
                carla.Location(x=sp_x, y=sp_y, z=sp_z),
                carla.Rotation(pitch=-22.0, yaw=t_rot.yaw, roll=0.0)
            ))
        except Exception:
            pass

def clean_up_entities(vehicles, collision_sensors):
    """ Rimuove in modo sicuro sensori e auto dal mondo CARLA """
    for s in collision_sensors:
        try:
            if s and s.is_alive:
                s.destroy()
        except Exception:
            pass
    for v in vehicles:
        try:
            if v and v.is_alive:
                v.destroy()
        except Exception:
            pass

def get_actor_path(base_dir, steps):
    """
    Costruisce il percorso completo per il modello di un agente:
    base_dir / checkpoint-{steps}_steps / actor.pth
    Ritorna il percorso se il file esiste, altrimenti None.
    """
    if not os.path.exists(base_dir):
        print(f"❌ Errore: La cartella history non esiste al percorso: {base_dir}")
        return None

    # Nome cartella e percorso file .pth
    folder_name = f"checkpoint-{steps}_steps"
    checkpoint_dir = os.path.join(base_dir, folder_name)
    model_path = os.path.join(checkpoint_dir, "actor.pth")

    if not os.path.exists(checkpoint_dir):
        print(f"⚠️ Attenzione: La cartella del checkpoint non esiste: {checkpoint_dir}")
        return None

    if not os.path.exists(model_path):
        print(f"⚠️ Attenzione: Il file actor.pth non è stato trovato in: {model_path}")
        return None

    return model_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Accelerazione PyTorch Demo: {device}")

    # 1. Connessione a CARLA
    client, world = connect_to_carla()
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()

    # 2. Caricamento Attori
    actor_0 = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    actor_1 = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)

    # Risoluzione dei percorsi
    path_actor_0 = get_actor_path(DIR_MODELS, STEPS_ACTOR_0)
    path_actor_1 = get_actor_path(DIR_MODELS, STEPS_ACTOR_1)

    # Controllo critico: blocco dello script se i pesi non possono essere caricati
    if path_actor_0 is None or path_actor_1 is None:
        print("\n❌ Impossibile avviare la demo: uno o entrambi i modelli 'actor.pth' non sono stati trovati.")
        print(f"   Percorso cercato per Agente 0 (step {STEPS_ACTOR_0}): {os.path.join(DIR_MODELS, f'checkpoint-{STEPS_ACTOR_0}_steps', 'actor.pth')}")
        print(f"   Percorso cercato per Agente 1 (step {STEPS_ACTOR_1}): {os.path.join(DIR_MODELS, f'checkpoint-{STEPS_ACTOR_1}_steps', 'actor.pth')}")
        sys.exit(1)

    # Caricamento effettivo dei pesi PyTorch con gestione eccezioni
    try:
        actor_0.load_state_dict(torch.load(path_actor_0, map_location=device))
        print(f"✅ Agente 0 (Rosso) caricato con successo da: {path_actor_0}")
    except Exception as e:
        sys.exit(f"❌ Errore durante il caricamento del modello Agente 0: {e}")

    try:
        actor_1.load_state_dict(torch.load(path_actor_1, map_location=device))
        print(f"✅ Agente 1 (Turchese) caricato con successo da: {path_actor_1}")
    except Exception as e:
        sys.exit(f"❌ Errore durante il caricamento del modello Agente 1: {e}")

    actor_0.eval()
    actor_1.eval()

    print("\n" + "="*60)
    print("🕹️ CONTROLLI DEMO:")
    print(" 👉 [SPAZIO] : Far partire la gara dalla griglia")
    print(" 👉 [C]      : Switchare la visuale tra Agente 0 e Agente 1")
    print(" 👉 [R]      : Resettare istantaneamente l'episodio")
    print(" 👉 [Ctrl+C] : Uscire dalla Demo")
    print("="*60 + "\n")

    vehicles = []
    collision_sensors = []

    try:
        while True:  # Loop continuo per permettere il reset con 'R'
            # Pulizia e reset ambiente
            clean_up_entities(vehicles, collision_sensors)
            vehicles, collision_sensors, collision_types = reset_environment(
                world, [], [], blueprint_library
            )

            selected_agent_idx = 0
            last_switch_time = time.time()
            active_agents = [True, True]
            still_counters = [0, 0]

            # Inquadra la griglia di partenza
            update_spectator_view(spectator, vehicles[selected_agent_idx])

            print("🚦 PRONTI SULLA GRIGLIA! Premi [SPAZIO] per partire (o [R] per resettare)...")

            # ------------------------------------------------------------------
            # A. PAUSA INIZIALE
            # ------------------------------------------------------------------
            reset_requested = False
            while True:
                world.tick()
                update_spectator_view(spectator, vehicles[selected_agent_idx])

                current_time = time.time()
                # Tasto C: Switch visuale
                if keyboard and keyboard.is_pressed('c') and (current_time - last_switch_time > 0.3):
                    selected_agent_idx = 1 - selected_agent_idx
                    print(f"🎥 Visuale sposta sull'Agente {selected_agent_idx}")
                    last_switch_time = current_time

                # Tasto R: Reset istantaneo
                if keyboard and keyboard.is_pressed('r') and (current_time - last_switch_time > 0.5):
                    print("🔄 Reset richiesto dall'utente!")
                    reset_requested = True
                    last_switch_time = current_time
                    break

                # Tasto SPAZIO: Via alla gara
                if keyboard and keyboard.is_pressed('space'):
                    print("🟢 VIA! Spinta iniziale applicata.")
                    break
                elif not keyboard:
                    input("Premi INVIO sul terminale per far partire la gara...")
                    break

                time.sleep(0.02)

            if reset_requested:
                continue

            # Applicazione spinta iniziale
            for v in vehicles:
                apply_initial_velocity(v, speed_kmh=40.0)

            # ------------------------------------------------------------------
            # B. CICLO GARA / INFERENCE
            # ------------------------------------------------------------------
            for step in range(DEMO_MAX_STEPS):
                current_time = time.time()

                # Controllo Reset Manuale con 'R' durante la corsa
                if keyboard and keyboard.is_pressed('r') and (current_time - last_switch_time > 0.5):
                    print(f"🔄 Episodio resettato manualmente al passo {step}!")
                    last_switch_time = current_time
                    break

                # Se non c'è più nessuno in pista, interrompe il loop e aspetta il reset o riparte
                if not any(active_agents):
                    print(f"💥 Nessuna auto rimasta in pista al passo {step}. Inizio nuovo episodio...")
                    time.sleep(1.0)
                    break

                # Controllo Switch manuale con 'C'
                if keyboard and keyboard.is_pressed('c') and (current_time - last_switch_time > 0.3):
                    other_idx = 1 - selected_agent_idx
                    if active_agents[other_idx]:
                        selected_agent_idx = other_idx
                        print(f"🎥 Visuale live spostata sull'Agente {selected_agent_idx}")
                    else:
                        print(f"⚠️ Impossibile passare all'Agente {other_idx}: è già stato eliminato!")
                    last_switch_time = current_time

                # Calcolo Azioni per Agenti Attivi
                for i in range(2):
                    if not active_agents[i]:
                        continue

                    vehicle = vehicles[i]
                    state_np = get_state(vehicle, world)
                    state_tensor = torch.from_numpy(state_np).float().to(device).unsqueeze(0)

                    with torch.no_grad():
                        mean, _ = actor_0(state_tensor) if i == 0 else actor_1(state_tensor)
                        actions = torch.tanh(mean).squeeze(0).cpu().numpy()

                    steer = float(actions[0])
                    acc_brake = float(actions[1])
                    vehicle.apply_control(carla.VehicleControl(
                        steer=steer,
                        throttle=max(0.0, acc_brake),
                        brake=max(0.0, -acc_brake)
                    ))

                    # Rilevamento Blocco
                    speed_kmh = state_np[16] * (MAX_VELOCITY * 3.6)
                    still_counters[i] = still_counters[i] + 1 if (speed_kmh < 5.0 and step > 40) else 0

                world.tick()

                # Gestione Collisioni ed Eliminazioni
                for i in range(2):
                    if not active_agents[i]:
                        continue

                    col_type = collision_types[i]
                    if col_type == "car_collision":
                        print(f"💥 COLLISIONE TRA AUTO al passo {step}!")
                        active_agents = [False, False]
                        break
                    elif col_type == "wall_collision" or still_counters[i] > 200:
                        reason = "Muro/Fuori Pista" if col_type == "wall_collision" else "Auto Bloccata"
                        print(f"❌ Agente {i} ELIMINATO ({reason}) al passo {step}.")

                        # Distruzione immediata dell'auto distrutta
                        try:
                            collision_sensors[i].destroy()
                            vehicles[i].destroy()
                        except Exception:
                            pass

                        active_agents[i] = False
                        collision_types[i] = None

                        # SWITCH AUTOMATICO DELLA TELECAMERA
                        # Se la telecamera stava puntando l'auto che si è appena distrutta:
                        if selected_agent_idx == i:
                            other_agent = 1 - i
                            if active_agents[other_agent]:
                                selected_agent_idx = other_agent
                                print(f"📹 Switch automatico telecamera -> Agente {selected_agent_idx} (superstite)")

                # Aggiornamento telecamera live sull'agente corretto
                if active_agents[selected_agent_idx]:
                    update_spectator_view(spectator, vehicles[selected_agent_idx])

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Demo interrotta dall'utente.")

    finally:
        print("🧹 Pulizia e ripristino ambiente...")
        clean_up_entities(vehicles, collision_sensors)

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        print("✅ Demo conclusa.")

if __name__ == "__main__":
    main()