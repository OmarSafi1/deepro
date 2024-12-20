import cv2
from utils import measure_distance, get_foot_position


class CalculSpeedwdistance:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def ajoutspeeddist(self, tracks):
        distancetotale = {}

        for obj, obj_tracks in tracks.items():
            if obj == "ball" or obj == "referees":
                continue
            nbparts = len(obj_tracks)




            for frame_num in range(0, nbparts, self.frame_window):
                last_frame = min(frame_num + self.frame_window, nbparts - 1)


                for track_id, _ in obj_tracks[frame_num].items():
                    if track_id not in obj_tracks[last_frame]:
                        continue

                    start_position = obj_tracks[frame_num][track_id]['position_transformed']
                    end_position = obj_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    vitesse = distance_covered / time_elapsed
                    speed_kph = vitesse * 3.6

                    if obj not in distancetotale:
                        distancetotale[obj] = {}

                    if track_id not in distancetotale[obj]:
                        distancetotale[obj][track_id] = 0

                    distancetotale[obj][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[obj][frame_num_batch]:
                            continue
                        tracks[obj][frame_num_batch][track_id]['speed'] = speed_kph
                        tracks[obj][frame_num_batch][track_id]['distance'] = distancetotale[obj][track_id]

    def orsemspeeddist(self, frames, tracks, speed_icon_path, distance_icon_path):
        # Load the icons
        speed_icon = cv2.imread(speed_icon_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
        distance_icon = cv2.imread(distance_icon_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

        # Resize icons for consistency
        icon_size = (20, 20)
        speed_icon = cv2.resize(speed_icon, icon_size, interpolation=cv2.INTER_AREA)
        distance_icon = cv2.resize(distance_icon, icon_size, interpolation=cv2.INTER_AREA)

        # Extract alpha channel and RGB channels
        speed_alpha = speed_icon[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]
        speed_rgb = speed_icon[:, :, :3]

        distance_alpha = distance_icon[:, :, 3] / 255.0
        distance_rgb = distance_icon[:, :, :3]

        output_frames = []
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj == "ball" or obj == "referees":
                    continue
                for _, track_info in obj_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        # Place speed icon
                        icon_pos_speed = (int(position[0]), int(position[1]))
                        for c in range(3):  # Blend each color channel
                            frame[icon_pos_speed[1]:icon_pos_speed[1] + icon_size[1],
                            icon_pos_speed[0]:icon_pos_speed[0] + icon_size[0], c] = (
                                    speed_alpha * speed_rgb[:, :, c] +
                                    (1 - speed_alpha) * frame[icon_pos_speed[1]:icon_pos_speed[1] + icon_size[1],
                                                        icon_pos_speed[0]:icon_pos_speed[0] + icon_size[0], c]
                            )

                        cv2.putText(frame, f"{speed:.1f}km/h", (icon_pos_speed[0] + 25, icon_pos_speed[1] + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                        # Place distance icon
                        icon_pos_distance = (int(position[0]), int(position[1] + 30))
                        for c in range(3):  # Blend each color channel
                            frame[icon_pos_distance[1]:icon_pos_distance[1] + icon_size[1],
                            icon_pos_distance[0]:icon_pos_distance[0] + icon_size[0], c] = (
                                    distance_alpha * distance_rgb[:, :, c] +
                                    (1 - distance_alpha) * frame[
                                                           icon_pos_distance[1]:icon_pos_distance[1] + icon_size[1],
                                                           icon_pos_distance[0]:icon_pos_distance[0] + icon_size[0], c]
                            )

                        cv2.putText(frame, f"{distance:.1f}m", (icon_pos_distance[0] + 25, icon_pos_distance[1] + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            output_frames.append(frame)
        return output_frames
