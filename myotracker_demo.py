import cv2
import torch
import numpy as np

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
    cap.release()
    return np.array(frames)

def select_points(frame):
    points = []
    display = frame.copy()
    window_name = "Select points (ESC to finish)"

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([y, x])
            # Draw only the new point
            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)
    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

def predict_points(model, video, points, device):
    H, W = video.shape[1:3]
    video_resized = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (256, 256)) for f in video]
    video_tensor = torch.from_numpy(np.array(video_resized)[None, :, None] / 255.).float().to(device)

    points_scaled = points / np.array([H, W]) * 256.0
    points_tensor = torch.from_numpy(points_scaled[None, ...]).float().to(device)

    with torch.no_grad():
        pred_scaled = model(video_tensor, points_tensor)[0].cpu().numpy()

    pred = pred_scaled / 256.0 * np.array([H, W])
    return pred.astype(np.int32)

def make_overlay_video(video, pred_points):
    output = []
    num_points = pred_points.shape[1]

    base_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    while len(base_colors) < num_points:
        base_colors.append(tuple(np.random.randint(0, 255, 3)))

    for t, frame in enumerate(video):
        frame_copy = frame.copy()
        for i, (y, x) in enumerate(pred_points[t]):
            color = tuple(int(c) for c in base_colors[i])
            cv2.circle(frame_copy, (x, y), 2, color, -1)
        output.append(cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
    return output

def save_video(path, frames, fps=15):
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.jit.load("myotracker/models/myotracker_scripted.pt", map_location=device)
    model = model.to(device)

    video_path = "sample.mp4"
    output_path = "sample_predicted.mp4"

    video = load_video(video_path)
    points = select_points(video[0])
    predictions = predict_points(model, video, points, device)
    overlay = make_overlay_video(video, predictions)

    save_video(output_path, overlay)
    print(f"Saved to {output_path}")