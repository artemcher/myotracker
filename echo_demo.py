import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Event handler for mouse clicks
selected_points = []
colors = [hsv_to_rgb([1.0/30*i, 1., 1.]) for i in range(30)]

def on_click(event, ax):
    if event.button == 1 and event.inaxes == ax:  # Left mouse button clicked
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        selected_points.append([x, y])
        ax.plot(x, y, 'o', color=colors[len(selected_points)%30], markersize=4)
        plt.draw()

def load_video(video_path):
    video = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        success, frame = cap.read()
        if not success: break
        video.append(frame)
    cap.release()
    
    video = np.array(video, np.uint8)
    return video

def select_points(video):
    points = []
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(Image.fromarray(video[0]))
    ax.axis('off')
    fig.canvas.manager.set_window_title('Select points to track. Close window to finish.')
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, ax))
    plt.tight_layout()
    plt.show()

def predict_points(model, video, points):
    h, w = video.shape[1:3]
    video_256 = np.array([cv2.cvtColor(cv2.resize(frame, (256,256)), cv2.COLOR_RGB2GRAY) for frame in video])
    points_256 = points / np.array([w, h])[None,:] * 256.0

    input_video = torch.from_numpy(video_256[None,...]/255.0).half()
    input_queries = torch.from_numpy(points_256[None,...]).half()
    
    pred_points_256 = model.cuda().forward(
        input_video.cuda(),
        input_queries.cuda()
    )[0].detach().cpu().numpy()

    pred_points = pred_points_256 / 256.0 * np.array([w, h])[None,:]
    return pred_points.astype(np.int32)

def make_overlay_video(video, pred_points):
    overlay_video = []
    for t in range(len(video)):
        overlay_frame = video[t]
        frame_points = pred_points[t]

        for i, p in enumerate(frame_points):
            color = [int(colors[i%30][c]*255.0) for c in range(3)]
            cv2.circle(overlay_frame, p, 3, color=color, thickness=-1)
        overlay_video.append(overlay_frame)
    return np.array(overlay_video, np.uint8)      

if __name__ == '__main__':
    model = torch.load("checkpoints/myotracker_051975.pt").cuda()
    video_path = #"../data/sample_videos/0942.mp4"
    video_name = video_path.split('/')[-1].split('.')[0]
    
    video = load_video(video_path)[10:]
    select_points(video) # stores in global 'selected_points'
    pred_points = predict_points(model, video, selected_points)
    overlay_video = make_overlay_video(video, pred_points)
    imageio.mimwrite(f"{video_name}.mp4", overlay_video, quality=8, fps=10)

