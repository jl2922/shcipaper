import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df  = pd.read_csv("speedup.csv")
df['Fast SHCI'] = df['Fast SHCI'] / 2
df['SHCI'] = df['SHCI'] / 2
df['Brute Force'] = df['Brute Force'] / 2
df['Elements'] = df['Elements'] * (df['Fast SHCI'][3] / df['Elements'][3])
print(df)

plt.figure(figsize=(5.5, 4.0))

xfit = np.linspace(0, 50, 50)
s1 = (df['Fast SHCI'] > 0.01)
z1 = np.polyfit(df['Dets'][s1].as_matrix(), df['Fast SHCI'][s1].as_matrix(), 2)
plt.plot(xfit, np.polyval(z1, xfit), '--', color='lightgrey', zorder=1)
s1 = (df['SHCI'] > 0.01)
z1 = np.polyfit(df['Dets'][s1].as_matrix(), df['SHCI'][s1].as_matrix(), 2)
plt.plot(xfit, np.polyval(z1, xfit), '--', color='lightgrey', zorder=1)
s1 = (df['Brute Force'] > 0.01)
z1 = np.polyfit(df['Dets'][s1].as_matrix(), df['Brute Force'][s1].as_matrix(), 2)
plt.plot(xfit, np.polyval(z1, xfit), '--', color='lightgrey', zorder=1)

l4 = plt.scatter(df['Dets'], df['Elements'], color='grey', marker='x', zorder=2)
l1 = plt.scatter(df['Dets'], df['Fast SHCI'], color='green', zorder=2)
l2 = plt.scatter(df['Dets'], df['SHCI'], color='orange', zorder=2)
l3 = plt.scatter(df['Dets'], df['Brute Force'], color='red', zorder=2)
s1 = (df['Elements'] > 0.01)
z1 = np.polyfit(df['Dets'][s1].as_matrix(), df['Elements'][s1].as_matrix(), 2)
plt.plot(xfit, np.polyval(z1, xfit), '--', color='lightgrey', zorder=1)

plt.xlim(xmin=0, xmax=40)
plt.ylim(ymin=0, ymax=180)
plt.title("Hamiltonian Matrix Construction Time")
plt.xlabel("Number of Determinants (in millions)")
plt.ylabel("CPU Time (in core hours)")
plt.grid(linestyle="dotted")
plt.legend((l3, l2, l1, l4), ("Brute Force", "Original SHCI", "Improved SHCI", "# Non-Zero Elements (Scaled)"))

plt.savefig('speedup.eps', format='eps', dpi=300)
plt.show()
# df.plot()  # plots all columns against index
# df.plot(kind='scatter',x='x',y='y') # scatter plot
# df.plot(kind='density')  # estimate density function
# df.plot(kind='hist')  # histogram
