from ultralytics import YOLO
import supervision as sv
import pickle

import os
import numpy as np



import pandas as pd
import cv2
import sys

import os
import sys


sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()


    import pandas as pd
    import numpy as np

    def positionduballon (self, ball_positions):
        # Safely extract bounding boxes and ensure valid data
        ball_positions = [
            x.get(1, {}).get('bbox', [np.nan, np.nan, np.nan, np.nan]) for x in ball_positions
        ]

        # Convert to DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values using linear interpolation, then fill remaining NaNs using forward fill
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')

        # Optionally, for smoother transitions, use a cubic interpolation
        df_ball_positions = df_ball_positions.interpolate(method='polynomial', order=3)

        # Fill any remaining NaNs with forward fill if any missing values still exist
        df_ball_positions = df_ball_positions.ffill().bfill()

        # Repack the interpolated data back into the original structure
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detectsparts(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def dessinertriangleballetjoueur(self, frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    def ajouterposition(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def getsuivijoueur(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detectsparts(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def mettretheegg(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame



    def dessinerpossession(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rounded rectangle for the background
        overlay = frame.copy()
        rect_color = (0, 0, 0)  # Black background for text
        rect_thickness = -1  # Solid fill
        corner_radius = 15  # Radius of the rounded corners

        # New Rectangle coordinates for the box (shifted left)
        top_left = (50, 850)  # Adjusted X position to the left
        bottom_right = (600, 970)  # Adjusted X position to the left

        # Create a mask for rounded corners
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, top_left, bottom_right, rect_color, rect_thickness)
        frame = cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0)

        # Drawing text with improved font and positioning
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # Calculate percentages
        total_frames = team_1_num_frames + team_2_num_frames
        team_1_percent = (team_1_num_frames / total_frames) * 100 if total_frames > 0 else 0
        team_2_percent = (team_2_num_frames / total_frames) * 100 if total_frames > 0 else 0

        # Font settings for modern design
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 1.2
        font_color = (255, 255, 255)  # White text
        thickness = 2
        line_type = cv2.LINE_AA

        # Text position (shifted left)
        text_1 = f"Team 1 Ball Control: {team_1_percent:.2f}%"
        text_2 = f"Team 2 Ball Control: {team_2_percent:.2f}%"

        # New text position (shifted left)
        text_1_position = (70, 900)  # Adjusted X position to the left
        text_2_position = (70, 950)  # Adjusted X position to the left

        # Display text on frame with shadows for better visibility
        shadow_offset = 3
        shadow_color = (0, 0, 0)  # Black shadow

        # Adding shadow effect
        for i, (text, position) in enumerate([(text_1, text_1_position), (text_2, text_2_position)]):
            # Draw shadow text first
            cv2.putText(frame, text, (position[0] + shadow_offset, position[1] + shadow_offset), font, font_scale,
                        shadow_color, thickness + 2, line_type)
            # Draw actual text
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.mettretheegg(frame, player["bbox"], color, track_id)

                if player.get('has_ball',False):
                    frame = self.dessinertriangleballetjoueur(frame, player["bbox"], (0, 191, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.mettretheegg(frame, referee["bbox"], (0, 255, 230))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.dessinertriangleballetjoueur(frame, ball["bbox"], (0, 0, 0))


            # Draw Team Ball Control
            frame = self.dessinerpossession(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames