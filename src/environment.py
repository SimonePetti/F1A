import carla
import math
import os
import numpy as np
from config.config import *

waypoint_locations = None
TOTAL_WAYPOINTS = 0
TRACK_LENGTH_METERS = 0.0

src_dir = os.path.dirname(os.path.abspath(__file__))
f1a_root_dir = os.path.dirname(src_dir)
absolute_waypoint_path = os.path.join(f1a_root_dir, WAYPOINT_FILE)

# --- Inizializzazione Tracciato ---
if os.path.exists(absolute_waypoint_path):
    data = np.load(absolute_waypoint_path)
    waypoint_locations = data['locations']
    if TRUE_START_INDEX != 0:
        # Ruota l'array dei waypoint in modo che il punto di partenza sia all'indice corretto
        waypoint_locations = np.roll(waypoint_locations, -TRUE_START_INDEX, axis=0)
    # Inverti l'asse Y per allineare il sistema di coordinate con quello del simulatore
    waypoint_locations[:, 1] = -waypoint_locations[:, 1]

    # Calcola la lunghezza totale del tracciato
    TOTAL_WAYPOINTS = len(waypoint_locations)
    wp_xy = waypoint_locations[:, :2]
    segment_distances = np.linalg.norm(wp_xy - np.roll(wp_xy, -1, axis=0), axis=1)
    TRACK_LENGTH_METERS = np.sum(segment_distances)

    print(f"🗺️ Waypoint di Monza caricati correttamente! ({TOTAL_WAYPOINTS} punti trovati, Lunghezza: {TRACK_LENGTH_METERS:.2f}m)")
else:
    print(f"❌ Errore fatale: Impossibile trovare il file dei waypoint in: {absolute_waypoint_path}")

def spawn_initial_vehicles(world, waypoint_locations, num_agents=2, frame_w=800, frame_h=600):
    """
    Spawna i veicoli affiancati sul primo waypoint.
    Ritorna la lista dei veicoli spawnati.
    """
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Calcolo geometrico per lo spawn affiancato
    wp_zero = waypoint_locations[0]
    wp_one = waypoint_locations[1]

    # Calcoliamo la direzione e l'angolo esatto dal punto 0 al punto 1 della pista
    yaw_rad = math.atan2(wp_one[1] - wp_zero[1], wp_one[0] - wp_zero[0])
    yaw_deg = math.degrees(yaw_rad)
    rotation_auto = carla.Rotation(pitch=0.0, roll=0.0, yaw=yaw_deg)

    right_vector = carla.Vector3D(x=-math.sin(yaw_rad), y=math.cos(yaw_rad), z=0)
    VEHICLE_WIDTH = 2.0
    LATERAL_MARGIN = 1.2
    offset = VEHICLE_WIDTH / 2 + LATERAL_MARGIN

    vehicles = []

    for i in range(num_agents):
        lateral = offset if i % 2 == 0 else -offset

        spawn_transform = carla.Transform(
            carla.Location(
                x=wp_zero[0] + right_vector.x * lateral,
                y=wp_zero[1] + right_vector.y * lateral,
                z=wp_zero[2] + 0.1
            ),
            rotation_auto
        )

        bp_locale = blueprint_library.filter('vehicle.tesla.model3')[0]
        if i == 0:
            bp_locale.set_attribute('color', '255,0,0')    # Rosso Corsa
        else:
            bp_locale.set_attribute('color', '0,242,255')  # Turchese/Cyan

        vehicle = world.try_spawn_actor(bp_locale, spawn_transform)

        if vehicle:
            vehicles.append(vehicle)
        else:
            # Se CARLA fallisce lo spawn, solleva un'eccezione per indicare il problema
            raise RuntimeError(f"Errore critico: Impossibile spawnare il veicolo {i} nella posizione indicata.")

    return vehicles