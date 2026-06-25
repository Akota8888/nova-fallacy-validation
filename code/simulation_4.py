import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# Simulation parameters
c = 299792458.0   # Speed of light (m/s)
M_hull = 12000.0  # Dry hull mass (kg)
P_laser = 1.5e9   # Laser power (W, 1.5 GW)
R_mirror = 0.999  # Reflection coefficient
L_cavity = 12.0   # Cavity length (m)
dt = 1e-9         # 1 ns time step
N = 5000          # Number of steps

# State initialisation
p_hull = 0.0
photons = []  # each entry: [position, momentum, direction (+1 or -1)]

times, p_hulls, p_photons_list, p_totals = [], [], [], []

for i in range(N):
    # 1. Emit photon packet from laser emitter
    em = (P_laser * dt) / c
    photons.append([0.0, em, 1])
    p_hull -= em  # recoil on hull

    # 2. Propagate photons and resolve wall impacts
    active = []
    for pos, mom, d in photons:
        pos += d * c * dt

        if pos >= L_cavity and d == 1:
            # Impact on front mirror
            p_hull += mom * (1.0 + R_mirror)
            active.append([L_cavity, mom * R_mirror, -1])

        elif pos <= 0.0 and d == -1:
            # Impact on rear wall
            p_hull -= mom * (1.0 + R_mirror)
            active.append([0.0, mom * R_mirror, 1])

        else:
            active.append([pos, mom, d])

    photons = active

    # 3. Compute total photon momentum
    p_ph = sum(p[1] * p[2] for p in photons)
    p_tot = p_hull + p_ph

    times.append(i * dt)
    p_hulls.append(p_hull)
    p_photons_list.append(p_ph)
    p_totals.append(p_tot)

# --- Save CSV ---
with open('simulation_results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time_s', 'p_hull_kg_m_s', 'p_photon_kg_m_s', 'p_total_kg_m_s'])
    for i in range(N):
        w.writerow([times[i], p_hulls[i], p_photons_list[i], p_totals[i]])

# --- Plot (IEEE two-column width: 3.5 inches) ---
t_ns = np.array(times) * 1e9
p_h  = np.array(p_hulls)
p_ph = np.array(p_photons_list)
p_tot = np.array(p_totals)

scale = max(np.max(np.abs(p_h)), 1e-30)  # normalise for display

fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.plot(t_ns, p_h  / scale, color='#1f77b4', lw=0.7,
        label=r'Hull momentum $p_{\mathrm{hull}}$')
ax.plot(t_ns, p_ph / scale, color='#ff7f0e', lw=0.7, ls='--',
        label=r'Photon momentum $p_{\mathrm{em}}$')
ax.plot(t_ns, p_tot / scale, color='#d62728', lw=1.3,
        label=r'$P_{\mathrm{total}} = 0$')

ax.set_xlabel(r'Time (ns)', fontsize=8)
ax.set_ylabel(r'Normalised Momentum', fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)
ax.set_xlim(0, t_ns[-1])
ax.axhline(0, color='gray', lw=0.4, ls=':')
ax.grid(True, lw=0.3, alpha=0.5)

fig.tight_layout(pad=0.4)
fig.savefig('simulation_results.pdf', bbox_inches='tight')  # for LaTeX
fig.savefig('simulation_results.png', dpi=200, bbox_inches='tight')  # preview
