#!/usr/bin/env python
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_file', default='result.json')
parser.add_argument('--save_figure', type=bool, default=True)
parser.add_argument('--show_figure', type=bool, default=True)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--preprint', type=bool, default=False)
parser.add_argument('--n_points', type=int, default=0)
args = parser.parse_args()

# Read JSON
result = open(args.result_file).read()
result = json.loads(result)

if args.preprint is True:
    plt.figure(figsize=(5.5, 4.0))

# Construct x and y
x = []
y = []
energy_vars = result['energy_var']
for eps_var, energy_var in energy_vars.iteritems():
    if result['energy_total'].get(eps_var) is None:
        continue
    energy_totals = result['energy_total'][eps_var]
    eps_pt = 1.0
    for eps_pt_iter, energy_total_iter in energy_totals.iteritems():
        eps_pt_iter = float(eps_pt_iter)
        if eps_pt_iter < eps_pt:
            eps_pt = eps_pt_iter
            energy_total = energy_total_iter['value']
    y.append(energy_total)
    x.append(energy_var - energy_total)
x = np.array(x)
y = np.array(y)

if args.n_points > 0:
    smallest_points = y.argsort()[:args.n_points]
    x = x[smallest_points]
    y = y[smallest_points]

# Fit
def model_aug(x):
    x_aug = (x, )
    for i in range(2, (args.order + 1)):
        x_aug = x_aug + (x**i, )
    x_aug = np.column_stack(x_aug)
    x_aug = sm.add_constant(x_aug)
    return x_aug

x_aug = model_aug(x)
weights = 1.0 / x**2
fit = sm.WLS(y, x_aug, weights).fit()
print(fit.summary())
alpha = 0.05
point = np.zeros(args.order + 1)
point[0] = 1.0
predict = fit.get_prediction(point).summary_frame(alpha=alpha)
predict = predict.iloc[0]
energy = fit.params[0]
uncert = predict['mean_ci_upper'] - predict['mean']
print('(%.2f Conf.) Extrapolated Energy: %.10f +- %.10f' % ((1.0 - alpha, fit.params[0], uncert)))
if np.isnan(uncert):
    uncert = 9999
result['energy_total']['extrapolated'] = {
    'value': energy,
    'uncert': uncert
}
with open(args.result_file, 'w') as result_file:
    json.dump(result, result_file, indent=2)


# Plot
x_max = 0.08
x_fit = np.linspace(0, x_max, 50)
x_fit_aug = model_aug(x_fit)
y_fit = fit.predict(x_fit_aug)
params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)
shci_handle, = plt.plot(x, y, marker='o', ls='')
plt.plot(x_fit, y_fit, color='grey', ls='--', zorder=0.2)
plt.xlabel('$E_{var} - E_{tot}$ (Ha)')
plt.ylabel('$E_{tot}$ (Ha)')
plt.title('Extrapolation to Full CI energy limit ($\epsilon_{1}\\to0$)')
plt.xlim(0, x_max)
ax = plt.gca()
ax.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.grid(True, ls=':')

dmrg_e_var = np.array([-2099.8346, -2099.8617, -2099.8743, -2099.8818])
dmrg_e_tot = np.array([-2099.9080, -2099.9109, -2099.9129, -2099.9141])
dmrg_x = dmrg_e_var - dmrg_e_tot
dmrg_y = dmrg_e_tot
dmrg_hamdle, = plt.plot(dmrg_x, dmrg_y, marker='*', markersize=7, ls='', zorder=0.5)
dmrg_fit = np.polyfit(dmrg_x[0:], dmrg_y[0:], 2)
dmrg_y_fit = np.poly1d(dmrg_fit)(x_fit)
plt.plot(x_fit, dmrg_y_fit, color='grey', ls='--', zorder=0.1)

ccsdt_handle = plt.hlines(-2099.88954277, 0, x_max, color='grey', linestyles='--')
plt.text(0, -2099.890, ' CCSD(T)', verticalalignment='top')
# plt.hlines(-2099.9061, 0, x_max, color='grey', linestyles='--')
# plt.hlines(-2099.9071, 0, x_max, color='grey', linestyles='--')
# plt.text(0, -2099.907, ' DMRG M=16000 ', verticalalignment='top', horizontalalignment='left')
# plt.hlines(-2099.9192, 0, x_max, color='grey', linestyles='--')
# plt.hlines(-2099.9195, 0, x_max, color='grey', linestyles='--')
# plt.text(x_max, -2099.919, ' DMRG Extrapolated ', verticalalignment='bottom', horizontalalignment='right')
# plt.hlines(-2099.9201, 0, x_max, color='grey', linestyles='--')
# plt.text(x_max, -2099.9207, ' DMRG+PT Extrapolated ', verticalalignment='top', horizontalalignment='right')
# plt.legend([shci_handle, dmrg_hamdle, ccsdt_handle], ['SHCI', 'pDMRG', 'CCSD(T)'])
plt.legend([shci_handle, dmrg_hamdle], ['SHCI', 'p-DMRG'], loc='lower right')
if args.save_figure:
    plt.savefig('extrapolate.eps')
if args.show_figure:
    plt.show()
