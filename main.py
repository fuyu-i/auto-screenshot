import cv2
import os
import time
import argparse

def is_too_dark(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold


def is_blurry(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_similar(frame1, frame2, threshold):
    diff = cv2.absdiff(frame1, frame2)
    mean_diff = diff.mean()
    return mean_diff < threshold


def ensure_dirs(base):
    dirs = {
        "valid": os.path.join(base, "valid"),
        "dark": os.path.join(base, "dark"),
        "blurry": os.path.join(base, "blurry"),
        "similar": os.path.join(base, "similar"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def main(args):
    folders = ensure_dirs(args.output)

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("[Error] - Could not open video source.")
        return
    
    prev_frame = None
    frame_count = 0
    last_timestamp = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] - Failed to read frame.")
            continue


        now = time.time()
        if now - last_timestamp >= args.interval:
            last_timestamp = now

            save_path = None

            if is_too_dark(frame, args.dark_thresh):
                save_path = folders["dark"]
                print("[Warning] - Skipped: Too dark")
            elif is_blurry(frame, args.blur_thresh):
                save_path = folders["blurry"]
                print("[Warning] - Skipped: Blurry")
            elif prev_frame is not None and is_similar(prev_frame, frame, args.sim_thresh):
                save_path = folders["similar"]
                print("[Warning] - Skipped: Similar to previous frame")
            else:
                save_path = folders["valid"]
                print("[Info] - Frame is valid")
                prev_frame = frame.copy()

            filename = os.path.join(save_path, f"frmae_{frame_count:04}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[Info] - Saved: {filename}")

            frame_count += 1
    
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto capture video frames")
    parser.add_argument("--output", type=str, default="output_frames")
    parser.add_argument("--source", type=str, default="rtmp://127.0.0.1/live/test")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--dark_thresh", type=float, default=40.0)
    parser.add_argument("--blur_thresh", type=float, default=100.0)
    parser.add_argument("--sim_thresh", type=float, default=2.0)

    args = parser.parse_args()

    try:
        args.source = int(args.source)
    except ValueError:
        pass

    main(args)