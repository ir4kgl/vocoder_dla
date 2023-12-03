import gdown

checkpoint_dir = "./checkpoints/"

gdown.download("https://drive.google.com/uc?id=1iw_Ak2-vuKCbAN4PnwRPeW6T-mJ0acPg",
               checkpoint_dir + "final_run/" + "checkpoint.pth", quiet=True)

