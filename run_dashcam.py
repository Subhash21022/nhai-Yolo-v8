import glob
from ultralytics import YOLO

print('?? Searching for video files...')
video_files = glob.glob('*.mp4') + glob.glob('*.avi') + glob.glob('*.mkv')

if not video_files:
    print('? ERROR: Could not find any video files! Check that your video is actually inside the D:\\test2 folder.')
else:
    my_video = video_files[0]
    print(f'? Found video: {my_video}')
    print('?? Initializing Synth-Reflect AI...')
    model = YOLO('best.pt')
    results = model.predict(source=my_video, save=True, conf=0.5)
    print('? Processing Complete! Check your runs folder.')
