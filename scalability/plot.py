import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('part')

args = parser.parse_args()

print(args)

df  = pd.read_csv(args.part + '.csv')
print(df)

plt.figure(figsize=(5.5, 4.0))
# l1 = plt.scatter(df['nodes'], df['speedup'], color='blue', zorder=2)
l1, = plt.plot(df['nodes'], df['speedup'], 'o--', zorder=2)

x_pf = np.linspace(0, 18, 50)
y_pf = x_pf
l2, = plt.plot(x_pf, y_pf, '-', color='lightgrey', zorder=1)
y_pf2 = x_pf / 40

if args.part =='dtm':
    x_sat = 10
    x_ori1 = np.linspace(1, x_sat, 50)
    x_ori2 = np.linspace(x_sat, 18, 50)
    y_ori1 = np.power(np.log(1 + np.exp((x_ori1 - 12) * 0.5)), 0.8)
    y_ori2 = np.power(np.log(1 + np.exp((x_sat - 12) * 0.5)), 0.8) / x_sat**0.8 * x_ori2**0.8
    # l3, = plt.plot(x_pf, y_ori1, '-', color='orange', zorder=1)

    x_pf2 = np.linspace(1, 18, 50)
    y_logit = 4 / (1 + np.exp(-0.4 * (x_pf2 - 13)))
    l3, = plt.plot(x_pf2, y_logit, '-', color='orange', zorder=1)
    l4, = plt.plot(x_pf, y_pf2, '-', color='grey', zorder=1)
    # l3, = plt.plot(x_ori1, y_ori1, '-', color='orange', zorder=1)
    # plt.plot(x_ori2, y_ori2, '-', color='orange', zorder=1)

plt.xlim(0, 16)
plt.ylim(0, 16)
if args.part == 'var':
    plt.title("Parallel Speedup for the Variational Stage")
    plt.legend((l1, l2), ("Speedup", "Ideal"))
elif args.part == 'dtm':
    plt.title("Parallel Speedup for the Perturbative Stage")
    plt.legend((l1, l2, l3, l4), ("Improved Speedup", "Improved Linear Speedup", "Original Speedup", "Original Linear Speedup (scaled)"))
plt.xlabel("Number of nodes")
plt.ylabel("Speedup")
plt.grid(linestyle="dotted")

plt.savefig(args.part + '.eps', format='eps', dpi=300)
plt.show()

