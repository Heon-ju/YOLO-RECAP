import argparse
from multiprocessing import freeze_support
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Data")
    p.add_argument("--model", type=str, default=r"None", help="Path to model")
    p.add_argument("--dataset_config", type=str, default="None", help="Path to dataset")
    p.add_argument("--epochs", type=int, default=100, help="number of epochs")
    p.add_argument("--imgsz", type=int, default=640, help="image size")

    p.add_argument("--output_dir", type=str, default=r"runs/train", help="Ultralytics 'project' directory")
    p.add_argument("--run_name", type=str,default="exp",help="Run name")
    p.add_argument("--data_dir", type=str, default=None, help="dataset directory")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"model           : {args.model}")
    print(f"dataset_config  : {args.dataset_config}")
    print(f"epochs          : {args.epochs}")
    print(f"imgsz           : {args.imgsz}")
    print(f"output_dir      : {args.output_dir}")
    print(f"run_name        : {args.run_name}")

    model = YOLO(args.model)

    model.train(
        data=args.dataset_config,
epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        project=args.output_dir,
        name=args.run_name,
    )

if __name__ == "__main__":
    freeze_support()
    main()
