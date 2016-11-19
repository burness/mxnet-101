import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file', type=str,default="102flower.log",
                    help='the path of log file')
args = parser.parse_args()


TR_RE = re.compile('.*\s.*Train-accuracy=([\d\.]+)')
VA_RE = re.compile('.*\sValidation-accuracy=([\d\.]+)')

log = open(args.log_file).read()

log_tr = [float(x) for x in TR_RE.findall(log)]
print log_tr
log_va = [float(x) for x in VA_RE.findall(log)]
print log_va
idx_tr = np.arange(len(log_tr))
idx_va = np.arange(len(log_va))
plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(idx_tr, log_tr, 'o', linestyle='-', color="r",
         label="Train accuracy")

plt.plot(idx_va, log_va, 'o', linestyle='-', color="b",
         label="Validation accuracy")

plt.legend(loc="best")
plt.xticks(np.arange(min(idx_tr), max(idx_tr)+1, 5))
plt.yticks(np.arange(0, 1, 0.2))
plt.ylim([0,1])
plt.show()