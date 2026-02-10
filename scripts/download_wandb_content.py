import os
import wandb

run_path = "likhtepi-ben-gurion-university-of-the-negev/RNA-Velocity/5e1ten1e"
out_dir = "wandb_downloads"
os.makedirs(out_dir, exist_ok=True)

api = wandb.Api()
run = api.run(run_path)

for file in run.files():
    # Filter for image panels or specific subfolders if needed
    if file.name.endswith((".png", ".jpg")) or file.name.startswith(("streamline/", "latent_time/")):
        print(f"Downloading {file.name} …")
        file.download(root=out_dir, replace=True)
