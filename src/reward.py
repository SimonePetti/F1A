from config.config import W_PROGRESS, W_DIRECTION, W_SPEED, W_COLLISION, DEAD_ZONE

class RewardFunction:
    def __init__(self):
        """
        Inizializza la funzione di reward ereditando i pesi globali.
        """
        pass

    def calculate_reward(self, progress, angle_norm, speed_norm, collision, acc_brake, current_steer, prev_steer):
        """
        Calcola la ricompensa di base focalizzandosi sulla stabilità di guida e velocità.
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

        # 5. Collisione
        R_collision = -W_COLLISION if collision else 0.0

        # Somma per ottenere la reward totale
        return float(R_progress + R_direction + R_steer_penalty + R_speed + R_collision)