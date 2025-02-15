from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import sys 
import cv2
import numpy as np
import pandas as pd 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def add_position_to_track(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    
                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else: 
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position
        
                        
                        
                        
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0,len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        
        return detections
    
    def interpolate_ball_positions(self, ball_positions):
        # example ball_position [{track_id:{"bbox":[]}}]
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing value 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}}for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
    
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks        
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "player":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names 
            
            clas_names_inv = {v:k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            
            # Convert GoadKeeper to player object 
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = clas_names_inv["player"]
                    
            # track Objects
            detecion_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["player"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            # print(tracks) => {'player': [{}], 'referees': [{}], 'ball': [{}]}
            for frame_detection in detecion_with_tracks:
                # print(frame_detection) => (array([1310.2, 447.96, 1351.2, 515.87], dtype=float32), None, 0.9369546, 2, 1, {'class_name': 'player'})
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == clas_names_inv['player']:
                    tracks['player'][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == clas_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == clas_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
                    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        
        y2 = int(bbox[3])
        
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        # Draw ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        
        )
        
        regtangle_width = 40 
        regtangle_height = 20 
        x1_rect = x_center - regtangle_width//2 # move left the half with from center
        x2_rect = x_center + regtangle_width//2
        y1_rect = (y2 - regtangle_height//2) + 15
        y2_rect = (y2 + regtangle_height//2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 15
            if track_id > 99:
                x1_text -= 10 
            cv2.putText(
                frame, 
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
        return frame
    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        traingle_point = np.array([
            [x, y], 
            [x-10, y-20],
            [x+10, y-20]
        ])
        cv2.drawContours(frame, [traingle_point], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [traingle_point], 0, color, 2)
        
        return frame
        
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw Summaries tranparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4        
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each time has ball 
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        team_1 = team1_num_frames/(team1_num_frames+team2_num_frames)
        team_2 = team2_num_frames/(team2_num_frames +team1_num_frames)
        
        cv2.putText(frame, f"Team 1 Ball Control {team_1*100:.2f}%", (1400, 900),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
        cv2.putText(frame, f"Team 2 Ball Control {team_2*100:.2f}%", (1400, 950),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),3)
        
        return frame
        
        
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referees_dict = tracks["referees"][frame_num]
            
            # Draw Players
            for track_id, player in player_dict.items():
                
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))
            
            # Draw Referees
            for track_id, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 0, 255))
            
            # Draw Ball
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0,0,255))
            # Draw Team Ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
    
            
            output_video_frames.append(frame)
        
        return output_video_frames
                         
                
            
        