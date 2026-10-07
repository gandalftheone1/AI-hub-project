import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results_technique2.csv")

distances = df["Bond_Distance_A"].values
x = np.arange(len(distances))  
x_labels = [f"{d:.2f}" for d in distances]

true_fci = df["True_Clean_Energy_Ha"].values
raw_energy = df["Raw_Energy_Ha"].values
mit_energy = df["Mitigated_Energy_Ha"].values

raw_err = df["Raw_Error_Ha"].values
mit_err = df["Mitigated_Error_Ha"].values
pct_red = df["Noise_Reduction_Pct"].values

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

ax1.plot(x, true_fci, 'k--', label='FCI Ground Truth (Ideal)', linewidth=2)
ax1.plot(x, raw_energy, 'r-x', label='Raw RL Optimization', linewidth=1.5, markersize=6)
ax1.plot(x, mit_energy, 'g-o', label='Hybrid RL + Transformer Denoised', linewidth=1.5, markersize=6)

ax1.set_ylabel('Energy Expectation (Hartree)', fontweight='bold')
ax1.set_title('H2 Molecule Dissociation Curve: Hybrid RL + Transformer Pipeline', fontweight='bold', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)


ax2.plot(x, raw_err, 'r--x', label='Raw RL Error |E_raw - E_fci|', linewidth=1.5, markersize=6)
ax2.plot(x, mit_err, 'g-o', label='Mitigated Error |E_denoised - E_fci|', linewidth=1.5, markersize=6)

ax2.set_ylabel('Absolute Error (Ha)', fontweight='bold')
ax2.set_title('Energy Expectation Residuals', fontweight='bold', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)


bar_width = 0.35
ax3.bar(x - bar_width/2, raw_err, width=bar_width, label='Raw Error', color='#e74c3c', alpha=0.85)
ax3.bar(x + bar_width/2, mit_err, width=bar_width, label='Mitigated Error', color='#2ecc71', alpha=0.85)

for i, pct in enumerate(pct_red):
    color = 'green' if pct >= 0 else 'red'
    prefix = '+' if pct >= 0 else ''
    y_pos = max(raw_err[i], mit_err[i]) + 0.03
    ax3.text(x[i], y_pos, f"{prefix}{pct:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)

ax3.set_xticks(x)
ax3.set_xticklabels(x_labels)
ax3.set_xlim([-0.6, len(x) - 0.4])  
ax3.set_xlabel('Bond Distance (Å)', fontweight='bold')
ax3.set_ylabel('Absolute Error (Hartree)', fontweight='bold')
ax3.set_title('Energy Error Mitigation & Noise Elimination', fontweight='bold', fontsize=11)
ax3.legend(loc='upper right')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("Final_RL_Hybrid_KeyGraphs.png", dpi=300)