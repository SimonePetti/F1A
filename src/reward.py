import math
from config.config import W_PROGRESS, W_DIRECTION, W_SPEED, W_COLLISION, W_COMPETITION, DEAD_ZONE

class RewardFunction:
    def __init__(self):
        """
        Inizializza la funzione di reward ereditando i pesi globali.
        """
        pass

    def calculate_reward(self, progress, angle_norm, speed_norm, collision, is_leading, distance_between_agents, real_physical_distance, 
                         acc_brake, current_steer, prev_steer, just_overtook, loc_0, loc_1, rot_0):
        """
        Calcola la ricompensa dell'agente.
        """
        # 1. Progresso
        # Se l'auto viaggia a meno di 25 km/h (7 m/s), non prende punti di progresso
        if speed_norm < 0.175:
            R_progress = 0.0
        else:
            R_progress = progress * W_PROGRESS

        # 2. Direzione (Penalità se l'angolo devia dalla pista)
        if abs(angle_norm) < DEAD_ZONE:
            R_direction = 0.0
        else:
            R_direction = -abs(angle_norm) * W_DIRECTION
        
        # 3. Penalità sterzo dinamica: più vai veloce, più lo zigzag è punito
        steer_delta = abs(current_steer - prev_steer)
        if steer_delta > 0.05 and speed_norm > 0.2:
            R_steer_penalty = -15.0 * steer_delta * speed_norm
        else:
            R_steer_penalty = 0.0

        # 4. Velocità (Punisce solo se l'auto è ferma e non sta dando gas)
        if speed_norm < 0.05:
            if acc_brake > 0.3:  # Almeno il 30% di gas per azzerare il malus
                R_speed = 0.0
            else:
                R_speed = -2.0 * W_SPEED
        else:
            R_speed = speed_norm * W_SPEED
        
        # 5. Interazione e Competitività (Logica di gara)
        R_competition = 0.0

        # Controllo sorpasso appena completato
        if just_overtook:
            R_competition = 30.0  # Premio per sorpasso pulito

        # Distanze ravvicinate lungo il tracciato
        elif distance_between_agents < 15.0:
            
            # 1. Calcolo del vettore distanza globale tra le due auto (in metri)
            dx = loc_1.x - loc_0.x
            dy = loc_1.y - loc_0.y

            # 2. Convertiamo lo yaw dell'Auto 0 da gradi a radianti per orientare il sistema
            theta = math.radians(rot_0.yaw)

            # 3. PROIEZIONE LOCALE (Visto dagli occhi dell'Auto 0)
            d_long = dx * math.cos(theta) + dy * math.sin(theta)
            d_lat = -dx * math.sin(theta) + dy * math.cos(theta)

            d_long_abs = abs(d_long)
            lateral_distance = abs(d_lat)

            # 4. Filtro per evitare falsi positivi di is_side_by_side / is_danger_zone quando le auto sono molto distanti ma la distanza reale è minore della distanza lungo la pista.
            if distance_between_agents > 6.0 and real_physical_distance < (distance_between_agents * 0.4):
                is_geometry_coherent = False
            else:
                is_geometry_coherent = True

            # 5. Soglie fisiche basate sulle bounding box delle auto (4.8m x 2.2m)
            # Sovrapposizione longitudinale delle carrozzerie lungo l'asse X locale
            has_long_overlap = d_long_abs < 4.8
            
            # Spazio laterale di sicurezza: la distanza tra i centri deve superare l'ingombro 
            # delle due auto (1.10m + 1.10m = 2.20m) più la tolleranza minima di 10cm (Totale > 2.30m)
            has_lateral_safety = lateral_distance > 2.30

            # 6. Identificazione is_side_by_side e is_danger_zone
            # Affiancamento pulito: le sagome si sovrappongono in lunghezza mantenendo il margine laterale
            is_side_by_side = has_long_overlap and has_lateral_safety and is_geometry_coherent

            # Zona rossa / sorpasso pericoloso: due casi principali di rischio collisione
            is_danger_zone = (d_long_abs < 7.5) and (lateral_distance <= 2.30) and is_geometry_coherent

            # 7. Applicazione reward e penalità
            if is_danger_zone:
                # Sanzione immediata a chiunque crei o subisca la situazione di pericolo.
                R_competition = -3.0
            
            # I premi positivi si attivano solo se l'auto viaggia a velocità sostenuta (> 25 km/h)
            elif speed_norm < 0.175:
                R_competition = 0.0
            
            else:
                # Dinamiche di gara in sicurezza (Spazio laterale > 2.30m o Distanza longitudinale >= 7.5m)
                proximity_factor = (15.0 - distance_between_agents) / 15.0

                if is_side_by_side:
                    R_competition = 2.2  # Premio affiancamento pulito
                
                elif not is_leading:
                    # Inseguitore attivo: velocità sostenuta (>72 km/h) per coprire le staccate,
                    # fuori dalla scia di collisione (>2.30m) e in traiettoria di attacco arretrata (d_long_abs >= 4.8m)
                    if speed_norm > 0.5 and lateral_distance > 2.30 and d_long_abs >= 4.8 and is_geometry_coherent:
                        R_competition = proximity_factor * 2.0
                    else:
                        # Inseguitore passivo: segue a distanza di sicurezza o in scia lontana
                        R_competition = proximity_factor * 0.1
                
                else:
                    # Leader sotto pressione: l'avversario è vicino ma a distanza di sicurezza.
                    R_competition = 0.0
        
        # Agenti distanti sulla pista (Oltre i 15 metri)
        else:
            if is_leading and speed_norm >= 0.175:
                R_competition = 0.5  # Bonus fuga per il leader che distanzia l'avversario

        R_competition *= W_COMPETITION

        # 6. Collisione
        R_collision = -W_COLLISION if collision else 0.0

        # Somma per ottenere la reward totale
        return float(R_progress + R_direction + R_steer_penalty + R_speed + R_competition + R_collision)