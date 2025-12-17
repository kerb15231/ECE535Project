import os, cv2, torch, logging, argparse, warnings
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from config import data_config
from utils.helpers import (
    get_model, draw_bbox_gaze, get_dataloader,
)

from utils.defenses import low_activity_pipeline
from utils.triggers import apply_trigger
import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--backdoor_weight", type=str, default="output/backdoored_model/best_model.pt")
    parser.add_argument("--mitigated_weight", type=str, default="output/clean_data_model/best_model.pt")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4")
    parser.add_argument("--output", type=str, default="output_dual.mp4")
    parser.add_argument("--dataset", type=str, default="mpiigaze")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--defense", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--poison-rate", type=float, default=0.0)
    parser.add_argument("--poison-target", nargs=2, type=float, default=[0.0, 0.0])
    return parser.parse_args()

# ------------------ Preprocessing ------------------
def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def open_video_source(source):
    if source.isdigit() or source == '0':
        cap = cv2.VideoCapture(int(source))
        if cap.isOpened(): return cap
    elif os.path.exists(source):
        cap = cv2.VideoCapture(source)
        if cap.isOpened(): return cap
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            logging.info(f"Using fallback webcam index {i}")
            return cap
    raise IOError("Cannot open webcam or video file")

def predict_angles(model, image, idx_tensor, binwidth, angle):
    pitch, yaw = model(image)
    pitch_p = F.softmax(pitch, dim=1)
    yaw_p = F.softmax(yaw, dim=1)
    pitch_deg = torch.sum(pitch_p * idx_tensor, dim=1) * binwidth - angle
    yaw_deg = torch.sum(yaw_p * idx_tensor, dim=1) * binwidth - angle
    return np.radians(pitch_deg.cpu().item()), np.radians(yaw_deg.cpu().item())

# ------------------ Main ------------------
def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)
    face_detector = uniface.RetinaFace()

    # Backdoored model
    backdoor_model = get_model(params.model, params.bins, inference_mode=False)
    backdoor_model.load_state_dict(torch.load(params.backdoor_weight, map_location=device))
    backdoor_model.to(device).eval()

    # Mitigated model
    mitigated_model = get_model(params.model, params.bins, inference_mode=True)
    mitigated_model.load_state_dict(torch.load(params.mitigated_weight, map_location=device))
    mitigated_model.to(device).eval()

    # Low Activity defended model
    defended_model = None
    if params.defense:
        defended_model_path = "low_activity_model.pt"
        if os.path.exists(defended_model_path):
            defended_model = get_model(params.model, params.bins, inference_mode=False)
            defended_model.load_state_dict(torch.load(defended_model_path, map_location=device))
            defended_model.to(device).eval()
            logging.info("Low activity defended model loaded from disk.")
        else:
            clean_loader = get_dataloader(params, mode="train")
            defended_model = low_activity_pipeline(
                backdoor_model,
                clean_loader,
                device,
                bins=params.bins,
                binwidth=params.binwidth,
                angle=params.angle,
                epochs=20,
                lr=1e-4,
                prune_ratio=0.5
            )
            torch.save(defended_model.state_dict(), defended_model_path)
            logging.info("Low activity defended model computed and saved to disk.")

    cap = open_video_source(params.source)
    out = None

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break
            bboxes, keypoints = face_detector.detect(frame)

            # Two versions of the frame
            frame_rb = frame.copy()   # red + blue
            frame_all = frame.copy()  # red + blue + green

            for bbox, _ in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                image_crop = frame[y_min:y_max, x_min:x_max]

                if params.attack:
                    # Apply the trigger to the cropped face
                    image_crop, patch_bbox = apply_trigger(image_crop)

                    # patch_bbox is usually (y0, y1, x0, x1) relative to the crop
                    y0, y1, x0, x1 = patch_bbox

                    # Offset to full-frame coordinates
                    abs_x0, abs_x1 = x_min + x0, x_min + x1
                    abs_y0, abs_y1 = y_min + y0, y_min + y1

                    # Draw the trigger box on the demo frames
                    cv2.rectangle(frame_rb, (abs_x0, abs_y0), (abs_x1, abs_y1), (255, 255, 255), -1)
                    cv2.rectangle(frame_all, (abs_x0, abs_y0), (abs_x1, abs_y1), (255, 255, 255), -1)

                image = pre_process(image_crop).to(device)

                # Backdoored model (red)
                pitch_rad, yaw_rad = predict_angles(backdoor_model, image, idx_tensor,
                                                    params.binwidth, params.angle)
                draw_bbox_gaze(frame_rb, bbox, pitch_rad, yaw_rad, color=(0,0,255))
                draw_bbox_gaze(frame_all, bbox, pitch_rad, yaw_rad, color=(0,0,255))

                # Mitigated model (blue)
                pitch_rad2, yaw_rad2 = predict_angles(mitigated_model, image, idx_tensor,
                                                      params.binwidth, params.angle)
                draw_bbox_gaze(frame_rb, bbox, pitch_rad2, yaw_rad2, color=(255,0,0))
                draw_bbox_gaze(frame_all, bbox, pitch_rad2, yaw_rad2, color=(255,0,0))

                # Defended model (green, only on frame_all)
                if params.defense and defended_model is not None:
                    pitch_rad3, yaw_rad3 = predict_angles(defended_model, image, idx_tensor,
                                                          params.binwidth, params.angle)
                    draw_bbox_gaze(frame_all, bbox, pitch_rad3, yaw_rad3, color=(0,255,0))

            # Show both windows
            if params.view:
                if params.attack and params. defense:
                    cv2.imshow("Red+Blue (Backdoor vs Clean) with trigger", frame_rb)
                    cv2.imshow("All Models (Clean vs Defense)", frame_all)
                else:
                    cv2.imshow("Red+Blue (Backdoor vs Clean) no trigger", frame_rb)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if params.output:
                out.write(frame)

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()

# ------------------ Entry ------------------
if __name__ == "__main__":
    args = parse_args()
    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")
    if args.dataset in data_config:
        cfg = data_config[args.dataset]
        args.bins = cfg["bins"]
        args.binwidth = cfg["binwidth"]
        args.angle = cfg["angle"]
        args.poison_target = [0.0, 0.0]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    main(args)