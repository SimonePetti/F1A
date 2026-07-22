import math
from config.config import W_PROGRESS, W_DIRECTION, W_SPEED, W_COLLISION, W_COMPETITION, MAX_VELOCITY_KMH, DEAD_ZONE

class RewardFunction:
    def __init__(self):
        """
        Inizializza la funzione di reward ereditando i pesi globali.
        """
        pass

    def calculate_reward(self, progress, angle_norm, speed_norm, collision, is_leading, distance_between_agents, real_physical_distance,
                         current_steer, prev_steer, just_overtook, loc_0, loc_1, rot_0, episode_step):
        """Calcola la reward dell'agente."""

        speed_kmh = speed_norm * MAX_VELOCITY_KMH

        # 1. Progresso lungo la pista
        R_progress = progress * W_PROGRESS

        # 2. Allineamento con la pista (penalità progressiva e moderata)
        abs_angle = abs(angle_norm)
        if abs_angle < DEAD_ZONE:
            R_direction = 0.0
        else:
            R_direction = -(abs_angle ** 2) * W_DIRECTION

        # 3. Penalità sterzo brusco a velocità elevate
        steer_delta = abs(current_steer - prev_steer)

        # Blindiamo il valore tra 0.0 e 1.0 per garantire il comportamento del filtro quadratico
        steer_delta_clipped = min(max(steer_delta, 0.0), 1.0)

        # Salva lo sterzo in rettilineo (straight_alignment ≈ 1.0), ma lascia sterzare in curva (straight_alignment → 0.0)
        straight_alignment = max(0.0, 1.0 - abs(angle_norm))

        # Calcolo finale pulito
        R_steer_penalty = -0.5 * (steer_delta_clipped ** 2) * speed_norm * straight_alignment

        # 4. Velocità e incentivo alla marcia
        R_speed = speed_norm * W_SPEED

        # Penalità se l'auto va troppo lenta
        if speed_kmh < 15.0 and episode_step > 40:
            R_speed -= 0.2

        # 5. Logica di Gara e Competizione
        R_competition = 0.0

        if just_overtook:
            R_competition = 3.0  # Premio per il sorpasso effettuato in sicurezza

        elif distance_between_agents < 20.0:
            dx = loc_1.x - loc_0.x
            dy = loc_1.y - loc_0.y
            theta = math.radians(rot_0.yaw)

            # Proiezione locale nel sistema di riferimento dell'auto
            d_long = dx * math.cos(theta) + dy * math.sin(theta)
            d_lat = -dx * math.sin(theta) + dy * math.cos(theta)

            d_long_abs = abs(d_long)
            lateral_distance = abs(d_lat)

            # Filtro per evitare falsi positivi di is_side_by_side / is_danger_zone quando le auto sono molto distanti ma la distanza reale è minore della distanza lungo la pista.
            if distance_between_agents > 6.0 and real_physical_distance < (distance_between_agents * 0.4):
                is_geometry_coherent = False
            else:
                is_geometry_coherent = True

            # Soglie fisiche basate sulle bounding box delle auto (4.8m x 2.2m)
            # Sovrapposizione longitudinale delle carrozzerie lungo l'asse X locale
            has_long_overlap = d_long_abs < 4.8

            # Spazio laterale di sicurezza: la distanza tra i centri deve superare l'ingombro
            # delle due auto (1.10m + 1.10m = 2.20m) più la tolleranza minima di 10cm (Totale > 2.30m)
            has_lateral_safety = lateral_distance > 2.30

            # Affiancamento pulito: le sagome si sovrappongono in lunghezza mantenendo il margine laterale
            is_side_by_side = has_long_overlap and has_lateral_safety and is_geometry_coherent

            # Zona rossa / sorpasso pericoloso: due casi principali di rischio collisione
            is_danger_zone = (d_long_abs < 7.5) and (lateral_distance <= 2.30) and is_geometry_coherent

            if is_danger_zone:
                R_competition = -0.5
            elif speed_kmh >= 20.0:
                proximity_factor = (20.0 - distance_between_agents) / 20.0
                if is_side_by_side:
                    R_competition = 0.2 # Premio affiancamento pulito
                elif not is_leading and speed_kmh > 40.0 and lateral_distance > 2.30 and d_long_abs >= 4.8:
                    # Inseguitore attivo: velocità sostenuta (>40 km/h),
                    # fuori dalla scia di collisione (>2.30m) e in traiettoria di attacco arretrata (d_long_abs >= 4.8m)
                    R_competition = proximity_factor * 0.15
        else:
            if is_leading and speed_kmh >= 50.0:
                R_competition = 0.1 # Agenti staccati: piccolo bonus al leader se mantiene alta velocità

        R_competition *= W_COMPETITION

        # 6. Collisione
        R_collision = -W_COLLISION if collision else 0.0

        # Totale
        return float(R_progress + R_direction + R_steer_penalty + R_speed + R_competition + R_collision)