import torch, logging, argparse, warnings
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
from config import data_config
from utils.helpers import (
    get_model,get_dataloader)

from utils.defenses import run_all_defenses, evaluate

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--backdoor_weight", type=str, default= "output/backdoored_model/best_model.pt")
    parser.add_argument("--mitigated_weight", type=str, default= "output/clean_data_model/best_model.pt")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4")
    parser.add_argument("--output", type=str, default="output_dual.mp4")
    parser.add_argument("--dataset", type=str, default="mpiigaze")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--defense", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--poison-rate", type=float, default=0.0,
                    help="Fraction of training samples to poison.")
    parser.add_argument("--poison-target", nargs=2, type=float, default=[0.0, 0.0],
                    help="Target gaze (pitch yaw in degrees) for poisoned samples.")
    return parser.parse_args()

# ------------------ Main ------------------
def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Backdoored model
    backdoor_model = get_model(params.model, params.bins, inference_mode=False)
    backdoor_model.load_state_dict(torch.load(params.backdoor_weight, map_location=device))
    backdoor_model.to(device).eval()

    # Mitigated model
    mitigated_model = get_model(params.model, params.bins, inference_mode=True)
    mitigated_model.load_state_dict(torch.load(params.mitigated_weight, map_location=device))
    mitigated_model.to(device).eval()

    clean_loader = get_dataloader(params, mode="train")
    params.trigger_test = True    
    poison_loader = get_dataloader(params, mode="test")

    evaluate(mitigated_model, clean_loader, device, params.bins, params.binwidth, params.angle)

    results = run_all_defenses(
        backdoor_model,
        clean_loader,
        poison_loader,
        device,
        bins=params.bins,
        binwidth=params.binwidth,
        angle=params.angle,
        target_gaze=tuple(params.poison_target),
        epochs=1,
        prune_ratio = 0.05
    )
    
    print(results)

# ------------------ Entry ------------------
if __name__ == "__main__":
    # You’d normally parse args here, e.g.:
    # params = parse_args()
    # main(params)
    pass
    args = parse_args()
    if not args.view and not args.output and not args.recover:
        raise Exception("At least one of --view, --output, or --recover must be provided.")
    if args.dataset in data_config:
        cfg = data_config[args.dataset]
        args.bins = cfg["bins"]
        args.binwidth = cfg["binwidth"]
        args.angle = cfg["angle"]
        args.poison_target = [0.0, 0.0]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    main(args)