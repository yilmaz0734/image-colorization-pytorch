import numpy as np
from utils import read_image
import sys


def main(argv):
    if len(argv) != 2:
        print("Usage: python evaluate.py estimations.npy img_names.txt")
        exit()
        
    with open(argv[1], "r") as f:
        files = f.readlines()
        
    estimations = np.load(argv[0])
    
    acc = 0
    for i, file in enumerate(files):
        cur = read_image(file.rstrip()).reshape(-1).astype(np.int64)
        est = estimations[i].reshape(-1).astype(np.int64)
    
        cur_acc = (np.abs(cur - est) < 12).sum() / cur.shape[0]
        acc += cur_acc
    acc /= len(files)
    print(f"{acc:.2f}/1.00")


if __name__ == "__main__":
    main(sys.argv[1:])
