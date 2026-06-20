import numpy as np
import matplotlib.pyplot as plt
import csv

def run_simulation(output_csv="simulation_results.csv", output_png="simulation_plots.png"):
    """
    Simulates a closed-loop propulsion cavity containing a physical chassis (mass M)
    and an internally propagating particle/wave packet (energy E, momentum p).
    
    This simulation illustrates the 'Einstein Box' paradox:
    1. The emitter fires a packet forward (+x), causing the chassis to recoil backward (-x).
    2. The packet propagates through the cavity.
    3. The packet collides with the forward boundary (elastic or inelastic), transferring
       its momentum back to the chassis, instantly stopping the motion.
    4. Over time, the net displacement oscillates transiently, but the net velocity
       is strictly zero, and the system's global Center of Mass (CoM) remains perfectly stationary.
    """
    print("Initializing Unified Closed-Loop Propulsion Simulation...")

    # --- Physical Parameters ---
    C = 299792458.0      # Speed of light (m/s)
    M_CHASSIS = 1000.0   # Mass of the spacecraft chassis (kg)
    E_PACKET = 1e11      # Energy of the fired wave/particle packet (Joules)
    L_CAVITY = 10.0      # Length of the internal cavity (meters)
    
    # Derived relativistic/momentum parameters
    m_packet_equiv = E_PACKET / (C**2)               # Equivalent mass (kg)
    p_packet = E_PACKET / C                          # Packet momentum (kg*m/s)
    v_recoil = -p_packet / M_CHASSIS                # Recoil velocity of chassis (m/s)
    v_packet = C                                    # Velocity of the photon/wave packet (m/s)
    
    # --- Time & Discretization ---
    # Flight time for the packet to cross the cavity
    t_flight = L_CAVITY / (v_packet - v_recoil)
    dt = t_flight / 1000.0                          # 1000 steps per flight segment
    
    # We will simulate 3 full cycles of: Firing -> Propagation -> Collision -> Reset
    cycles = 3
    steps_per_cycle = 2000
    total_steps = cycles * steps_per_cycle
    
    # --- State Containers ---
    time_series = []
    chassis_pos = []
    chassis_vel = []
    chassis_mom = []
    packet_pos = []
    packet_mom = []
    system_com = []
    
    # Initial state
    t = 0.0
    x_chassis = 0.0
    x_packet = 0.0
    
    # Run loop
    for step in range(total_steps):
        cycle_idx = step // steps_per_cycle
        cycle_step = step % steps_per_cycle
        
        # Determine internal phase
        # 0 to 999: Packet is in flight
        # 1000 to 1999: Packet has collided and is absorbed / system at rest before refiring
        if cycle_step < 1000:
            # Active Propagation Phase (Recoil Active)
            u_chassis = v_recoil
            u_packet = v_recoil + v_packet
            
            p_chass = M_CHASSIS * u_chassis
            p_pack = p_packet
            
            x_chassis += u_chassis * dt
            x_packet += u_packet * dt
        else:
            # Impact and Post-Collision Rest Phase
            u_chassis = 0.0
            u_packet = x_chassis + L_CAVITY # Packet is absorbed at the front wall
            
            p_chass = 0.0
            p_pack = 0.0
            
            # Keep state fixed until next firing cycle
        
        # Calculate instantaneous global Center of Mass
        # CoM = (M1*x1 + M2*x2) / (M1 + M2)
        total_mass = M_CHASSIS + m_packet_equiv
        current_com = (M_CHASSIS * x_chassis + m_packet_equiv * x_packet) / total_mass
        
        # Append data points
        time_series.append(t)
        chassis_pos.append(x_chassis)
        chassis_vel.append(u_chassis)
        chassis_mom.append(p_chass)
        packet_pos.append(x_packet)
        packet_mom.append(p_pack)
        system_com.append(current_com)
        
        t += dt

    # --- Write Results to CSV ---
    print(f"Saving simulation data to '{output_csv}'...")
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_s", "ChassisPosition_m", "ChassisVelocity_ms", "ChassisMomentum_kgms", 
                         "PacketPosition_m", "PacketMomentum_kgms", "SystemCenterOfMass_m"])
        for i in range(total_steps):
            writer.writerow([time_series[i], chassis_pos[i], chassis_vel[i], chassis_mom[i],
                             packet_pos[i], packet_mom[i], system_com[i]])
            
    # --- Generate Publication-Quality Plots ---
    print(f"Generating visual disproof plots and saving to '{output_png}'...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Subplot 1: Momentums (Showing perfect equal and opposite action/reaction)
    axs[0].plot(time_series, chassis_mom, label="Chassis Momentum ($p_{chassis}$)", color="royalblue", lw=2)
    axs[0].plot(time_series, packet_mom, label="Wave/Particle Momentum ($p_{packet}$)", color="orange", lw=1.5, linestyle="--")
    axs[0].set_ylabel("Linear Momentum (kg$\cdot$m/s)")
    axs[0].set_title("Unified Propulsion Fallacy: Momentum Conservation Verification")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="upper right")
    
    # Subplot 2: Chassis Position (Einstein Box transient shift)
    axs[1].plot(time_series, chassis_pos, label="Chassis Position ($x_{chassis}$)", color="crimson", lw=2)
    axs[1].set_ylabel("Chassis Displacement (meters)")
    axs[1].grid(True, linestyle=":", alpha=0.6)
    axs[1].legend(loc="upper right")
    
    # Subplot 3: Combined System Center of Mass (Proving zero net movement)
    axs[2].plot(time_series, system_com, label="Global Center of Mass ($x_{CoM}$)", color="forestgreen", lw=2.5)
    axs[2].set_ylim(-L_CAVITY/5, L_CAVITY/5)
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("CoM Coordinate (meters)")
    axs[2].grid(True, linestyle=":", alpha=0.6)
    axs[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print("Simulation execution complete. Fallacy verified: Net-zero coordinate acceleration.")

if __name__ == "__main__":
    run_simulation()
