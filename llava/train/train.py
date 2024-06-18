from llava.train.train_compress import train as train_compress

import os
os.environ["WANDB_DISABLED"] = "true"




if __name__ == "__main__":
    train_compress(attn_implementation="sdpa")

