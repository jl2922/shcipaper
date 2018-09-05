import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df)

df['log_n_dets'] = df['log_n_dets'] + 9

def model(x):
    y = 41 + 13.4*x + 1.15*x**2 + 9
    return y

plt.figure(figsize=(5.5, 4.0))
l1, = plt.plot(df['log_eps_pt'], df['log_n_dets'], 'o', color='green', zorder=2)

x_fit = np.linspace(-9, -5)
y_fit = model(x_fit)

x_pred = np.log10(3.0e-9)
y_pred = model(x_pred)
plt.plot(x_fit, y_fit, '--', color='lightgrey', zorder=1)
l2, = plt.plot(x_pred, y_pred, '^', color='blue', zorder=2)
plt.title("Estimation of Effective PT Determinants")
plt.xlim(-9.0, -5.8)
plt.grid(linestyle="dotted")
plt.xlabel("log$_{10}(\epsilon_2$)")
plt.ylabel("log$_{10}$(number of determinants)")

plt.legend((l2,), ('Est. of $\epsilon_2=3*10^{-9}$',))

plt.savefig('n_dets.eps', format='eps', dpi=300)
plt.show()

exit(0)

l1, = plt.plot(df['nodes'], df['speedup'], 'o--', color='blue', zorder=2)

x_pf = np.linspace(0, 18, 50)
y_pf = x_pf
l2, = plt.plot(x_pf, y_pf, '-', color='lightgrey', zorder=1)

if args.part =='dtm':
    x_sat = 10
    x_ori1 = np.linspace(0, x_sat, 50)
    x_ori2 = np.linspace(x_sat, 18, 50)
    y_ori1 = np.power(np.log(1 + np.exp((x_pf - 12) * 0.5)), 0.8)
    y_ori2 = np.power(np.log(1 + np.exp((x_sat - 12) * 0.5)), 0.8) / x_sat**0.8 * x_ori2**0.8
    # l3, = plt.plot(x_pf, y_ori1, '-', color='orange', zorder=1)
    y_logit = 4 / (1 + np.exp(-0.4 * (x_pf - 13)))
    l3, = plt.plot(x_pf, y_logit, '-', color='orange', zorder=1)
    # l3, = plt.plot(x_ori1, y_ori1, '-', color='orange', zorder=1)
    # plt.plot(x_ori2, y_ori2, '-', color='orange', zorder=1)

plt.xlim(0, 16)
plt.ylim(0, 16)
if args.part == 'var':
    plt.title("Parallel Speedup for the Variational Part")
    plt.legend((l1, l2), ("Speedup", "Ideal"))
elif args.part == 'dtm':
    plt.title("Parallel Speedup for the PT Part")
    plt.legend((l1, l2, l3), ("Speedup", "Ideal", "Original SHCI (Estimate)"))
plt.xlabel("Number of nodes")
plt.ylabel("Speedup")
plt.grid(linestyle="dotted")

plt.savefig(args.part + '.eps', format='eps', dpi=300)
plt.show()

