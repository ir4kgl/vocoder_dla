import gdown

checkpoint_dir = "./checkpoints/"

gdown.download("?",
               checkpoint_dir + "final_run/" + "checkpoint.pth", quiet=True)
