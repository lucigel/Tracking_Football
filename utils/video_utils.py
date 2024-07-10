import cv2 

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        sucess, frame = cap.read()
        
        if not sucess:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        output.write(frame)
    output.release()
    
