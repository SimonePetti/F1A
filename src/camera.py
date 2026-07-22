import carla
import math
import numpy as np
import time
import os
import cv2
import threading

try:
    import keyboard
except ImportError:
    keyboard = None

from config.config import VIDEO_FPS, FRAME_W, FRAME_H, SOGLIA_STEP_SALVATAGGIO

class CameraManager:
    def __init__(self, world, video_save_dir="video_addestramento"):
        self.world = world
        self.spectator = world.get_spectator()
        self.blueprint_library = world.get_blueprint_library()

        # Colore della macchina degli agenti
        self.colore_auto_0 = "N/A"
        self.colore_auto_1 = "N/A"

        # Configurazione Video Recorder
        self.video_save_dir = video_save_dir
        self.recorder_sensor = None
        self.video_out = None
        self.camera_attiva = False
        self.video_lock = threading.Lock()

        # Buffer Thread-Safe per la cattura immagini (evita crash di concorrenza)
        self.latest_frame = None

        # Parametri video
        self.video_fps = VIDEO_FPS
        self.frame_w = FRAME_W
        self.frame_h = FRAME_H
        self.soglia_step_salvataggio = SOGLIA_STEP_SALVATAGGIO
        self.episode_step_counter = 0

        # Configurazione Spectator Live (Utente)
        self.target_vehicle_idx = 0 # Parte puntando l'Agente 0 di default
        self.last_switch_time = 0.0

        # Variabili di stato per Telemetria
        self.telemetria = {
            "speed_0": 0.0,
            "speed_1": 0.0,
            "distance": 0.0,
            "reward_0": 0.0,
            "reward_1": 0.0,
            "laps_0": 0,
            "laps_1": 0
        }

        os.makedirs(self.video_save_dir, exist_ok=True)

    def start_episode(self, episode_id, yaw_deg, wp_zero, colore_0="N/A", colore_1="N/A"):
        """Inizializza lo spectator e prepara il recorder"""
        # Pulizia preventiva del vecchio writer se rimasto aperto
        self._close_writer_safe()

        self.current_episode_id = episode_id
        self.episode_step_counter = 0
        self.target_vehicle_idx = 0
        self.colore_auto_0 = colore_0
        self.colore_auto_1 = colore_1
        self.latest_frame = None

        yaw_rad = math.radians(yaw_deg)
        sp_loc = carla.Location(
            x=wp_zero[0] - 15.0 * math.cos(yaw_rad),
            y=wp_zero[1] - 15.0 * math.sin(yaw_rad),
            z=wp_zero[2] + 12.0
        )
        sp_rot = carla.Rotation(pitch=-30.0, yaw=yaw_deg, roll=0.0)

        try:
            self.spectator.set_transform(carla.Transform(sp_loc, sp_rot))
        except Exception:
            pass

        # Crea o riutilizza il sensore della fotocamera
        if self.recorder_sensor is None or not self.recorder_sensor.is_alive:
            try:
                camera_bp = self.blueprint_library.find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', str(self.frame_w))
                camera_bp.set_attribute('image_size_y', str(self.frame_h))
                camera_bp.set_attribute('fov', '80')

                initial_transform = carla.Transform(sp_loc, sp_rot)
                self.recorder_sensor = self.world.spawn_actor(camera_bp, initial_transform)
                self.recorder_sensor.listen(lambda image: self._on_carla_image(image))
            except Exception as e:
                print(f"❌ [CameraManager] Errore nello spawn telecamera: {e}")
                return

        # Prepara il VideoWriter di OpenCV
        temp_video_path = os.path.join(self.video_save_dir, f"episodio_{episode_id}_temp.mp4")
        with self.video_lock:
            self.video_out = cv2.VideoWriter(
                temp_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.video_fps,
                (self.frame_w, self.frame_h)
            )
            self.camera_attiva = True

        print(f"🎥 [CameraManager] Registrazione avviata per Episodio {episode_id}")

    def _on_carla_image(self, carla_image):
        """Callback ultra-leggera eseguita nel thread di CARLA: salva solo il frame grezzo"""
        if not self.camera_attiva:
            return

        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        if array.size == carla_image.height * carla_image.width * 4:
            array = np.reshape(array, (carla_image.height, carla_image.width, 4))
            bgr_frame = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
            with self.video_lock:
                self.latest_frame = bgr_frame

    def update_camera_positions(self, vehicles, statistical_leader, speed_0, speed_1, distance, reward_0, reward_1, laps_0, laps_1):
        """Aggiorna la posizione della telecamera e scrive il frame con overlay nel thread principale"""
        self.episode_step_counter += 1
        if not vehicles or len(vehicles) < 2:
            return

        # Update Telemetria
        self.telemetria["speed_0"] = speed_0
        self.telemetria["speed_1"] = speed_1
        self.telemetria["distance"] = distance
        self.telemetria["reward_0"] = reward_0
        self.telemetria["reward_1"] = reward_1
        self.telemetria["laps_0"] = laps_0
        self.telemetria["laps_1"] = laps_1

        # 1. Tasto 'C' per switch visuale Spectator
        current_time = time.time()
        if keyboard is not None:
            try:
                if keyboard.is_pressed('c'):
                    if current_time - self.last_switch_time > 0.3:
                        self.target_vehicle_idx = 1 - self.target_vehicle_idx
                        print(f"🎥 [Spectator] Visuale utente spostata sull'Agente {self.target_vehicle_idx}")
                        self.last_switch_time = current_time
            except Exception:
                pass

        # Costanti d'inseguimento
        DISTANZA_DIETRO = 12.0
        ALTEZZA_Z = 5.0
        INCLINAZIONE_PITCH = -22.0

        # 2. Spostamento Spectator (Live View)
        try:
            tracked_vehicle = vehicles[self.target_vehicle_idx]
            if tracked_vehicle and tracked_vehicle.is_alive:
                t_trans = tracked_vehicle.get_transform()
                t_loc = t_trans.location
                t_rot = t_trans.rotation

                t_yaw_rad = math.radians(t_rot.yaw)
                sp_x = t_loc.x - DISTANZA_DIETRO * math.cos(t_yaw_rad)
                sp_y = t_loc.y - DISTANZA_DIETRO * math.sin(t_yaw_rad)
                sp_z = t_loc.z + ALTEZZA_Z

                self.spectator.set_transform(carla.Transform(
                    carla.Location(x=sp_x, y=sp_y, z=sp_z),
                    carla.Rotation(pitch=INCLINAZIONE_PITCH, yaw=t_rot.yaw, roll=0.0)
                ))
        except Exception:
            pass

        # 3. Spostamento Sensore Telecamera
        if self.recorder_sensor is not None and self.recorder_sensor.is_alive:
            try:
                leader_vehicle = vehicles[statistical_leader]
                if leader_vehicle and leader_vehicle.is_alive:
                    leader_transform = leader_vehicle.get_transform()
                    leader_loc = leader_transform.location
                    leader_rot = leader_transform.rotation

                    yaw_rad = math.radians(leader_rot.yaw)
                    cam_x = leader_loc.x - DISTANZA_DIETRO * math.cos(yaw_rad)
                    cam_y = leader_loc.y - DISTANZA_DIETRO * math.sin(yaw_rad)
                    cam_z = leader_loc.z + ALTEZZA_Z

                    self.recorder_sensor.set_transform(carla.Transform(
                        carla.Location(x=cam_x, y=cam_y, z=cam_z),
                        carla.Rotation(pitch=INCLINAZIONE_PITCH, yaw=leader_rot.yaw, roll=0.0)
                    ))
            except Exception:
                pass

        # 4. Scrittura Frame nel Video
        with self.video_lock:
            if self.camera_attiva and self.video_out is not None and self.latest_frame is not None:
                frame_to_write = self.latest_frame.copy()
                self.latest_frame = None  # Reset buffer per evitare frame duplicati

                # Overlay Telemetria
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5

                stringhe_info = [
                    f"Episodio: {self.current_episode_id} | Step: {self.episode_step_counter} | Distacco: {self.telemetria['distance']:.2f} m",
                    f"Agente 0 ({self.colore_auto_0}) -> Giri: {self.telemetria['laps_0']} | Vel: {self.telemetria['speed_0']:.1f} km/h | Rwd: {self.telemetria['reward_0']:.2f}",
                    f"Agente 1 ({self.colore_auto_1}) -> Giri: {self.telemetria['laps_1']} | Vel: {self.telemetria['speed_1']:.1f} km/h | Rwd: {self.telemetria['reward_1']:.2f}"
                ]

                colori = [(0, 255, 0), (255, 255, 255), (255, 255, 0)]
                y_offset = 30

                for testo, colore in zip(stringhe_info, colori):
                    cv2.putText(frame_to_write, testo, (20, y_offset), font, scale, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                    cv2.putText(frame_to_write, testo, (20, y_offset), font, scale, colore, thickness=1, lineType=cv2.LINE_AA)
                    y_offset += 22

                self.video_out.write(frame_to_write)

    def _close_writer_safe(self):
        """Chiude il VideoWriter di OpenCV in totale sicurezza"""
        with self.video_lock:
            self.camera_attiva = False
            if self.video_out is not None:
                try:
                    self.video_out.release()
                except Exception:
                    pass
                self.video_out = None

    def end_episode(self):
        """Salva o rimuove il file video al termine dell'episodio"""
        self._close_writer_safe()

        temp_video_path = os.path.join(self.video_save_dir, f"episodio_{self.current_episode_id}_temp.mp4")
        if os.path.exists(temp_video_path):
            if self.episode_step_counter >= self.soglia_step_salvataggio:
                final_video_path = os.path.join(self.video_save_dir, f"episodio_{self.current_episode_id}_{self.episode_step_counter}_steps.mp4")
                try:
                    if os.path.exists(final_video_path):
                        os.remove(final_video_path)
                    os.rename(temp_video_path, final_video_path)
                    print(f"💾 [Recorder] Video salvato: {final_video_path}")
                except Exception as e:
                    print(f"⚠️ Impossibile rinominare il file video: {e}")
            else:
                try:
                    os.remove(temp_video_path)
                    print(f"🗑️ [Recorder] Video scartato (< {self.soglia_step_salvataggio} step).")
                except Exception:
                    pass

    def cleanup(self):
        """Distruzione completa del sensore alla fine del programma"""
        self._close_writer_safe()
        if self.recorder_sensor is not None:
            try:
                if self.recorder_sensor.is_alive:
                    self.recorder_sensor.destroy()
            except Exception:
                pass
            finally:
                self.recorder_sensor = None