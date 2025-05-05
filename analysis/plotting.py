import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Normalization constants
P_CHAMBER = 1.893  # Chamber pressure for normalization
U_0 = 1.0  # Reference velocity for normalization
# Note: L_NOZZ will be calculated for each mesh using get_nozzle_end

def get_end_time(mesh_num):
    t_end = t_write = None
    filepath = f"data/mesh{mesh_num}/system/controlDict"
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('endTime'):
                t_end = float(line.split()[1].rstrip(';'))
            if line.startswith('writeInterval'):
                t_write = float(line.split()[1].rstrip(';'))
    
    if t_end is None or t_write is None:
        raise ValueError(f"endTime or writeInterval not found in {filepath}")

    return t_end, t_write

def get_nozzle_end(mesh_num):
    """
    Extract the average x-coordinate between pre and post nozzle positions and the y-values at pre and axial.
    
    Args:
        mesh_num (int): Mesh number to read data from
        
    Returns:
        tuple: (average_x, pre_y, axial_y) where:
               - average_x is the average x-coordinate of the nozzle end
               - pre_y is the y-coordinate at the pre position
               - axial_y is the y-coordinate of the axial line
    """
    filepath = f"data/mesh{mesh_num}/system/singleGraph"
    pre_x = post_x = pre_y = axial_y = None
    
    with open(filepath, 'r') as f:
        in_pre = in_post = in_axial = False
        for line in f:
            line = line.strip()
            
            if line.startswith('pre'):
                in_pre = True
                in_post = in_axial = False
            elif line.startswith('post'):
                in_post = True
                in_pre = in_axial = False
            elif line.startswith('axial'):
                in_axial = True
                in_pre = in_post = False
            elif line == '}':  # Reset section tracking at closing brace
                in_pre = in_post = in_axial = False
            elif line.startswith('end') and (in_pre or in_post or in_axial):
                # Extract coordinates from line like: end (3.346708860759494 0.895 0);
                coords = line.split('(')[1].split(')')[0].split()
                x = float(coords[0])
                if in_pre:
                    pre_x = x
                    pre_y = float(coords[1])  # Get y value for pre
                elif in_post:
                    post_x = x
                elif in_axial:
                    axial_y = float(coords[1])  # Get y value for axial
                    
    if pre_x is None or post_x is None or axial_y is None:
        raise ValueError(f"Could not find required coordinates in {filepath}")
        
    return (pre_x + post_x) / 2, pre_y, axial_y

def format_time(time):
    """
    Format time value consistently for file paths.
    
    Args:
        time (float/int): Time value to format
        
    Returns:
        str: Formatted time string
    """
    # Round to 6 decimal places to handle floating point precision
    rounded_time = round(float(time), 6)
    if rounded_time.is_integer():
        return str(int(rounded_time))
    return f"{rounded_time:.6f}".rstrip('0').rstrip('.')

def read_field_data(mesh_num, field_name, time, t_write):
    """
    Read field data from the specified file for a given mesh number and time.
    
    Args:
        mesh_num (int): Mesh number to read from
        field_name (str): Name of the field (e.g., 'p', 'T', 'U')
        time (float): Time interval to read data for
        t_write (float): Write interval to determine decimal precision
        
    Returns:
        np.ndarray: Array of field values
    """
    formatted_time = format_time(time)
    
    # For p and T, we now read from the axial_p_T.xy file
    if field_name in ['p', 'T']:
        filepath = f"data/mesh{mesh_num}/postProcessing/singleGraph/{formatted_time}/axial_p_T.xy"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                # Split line into values and convert to float
                values = list(map(float, line.strip().split()))
                # values format: x, y, z, p, T
                if field_name == 'p':
                    data.append(values[3])  # pressure is 4th column
                else:  # T
                    data.append(values[4])  # temperature is 5th column
        return np.array(data)
    
    # For velocity, keep existing U file parsing
    elif field_name == 'U':
        filepath = f"data/mesh{mesh_num}/{formatted_time}/{field_name}"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = []
            inside_section = False
            for line in f:
                line = line.strip()
                
                if line == '(': 
                    inside_section = True
                    continue
                elif line == ')':  
                    inside_section = False
                    break
                
                if inside_section:
                    # Parse vector components
                    match = re.match(r'^\(([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\)$', line)
                    if match:
                        Ux, Uy, Uz = map(float, match.groups())
                        data.append((Ux, Uy, Uz))
        return np.array(data)
    
    else:
        raise ValueError(f"Unsupported field name: {field_name}")

def create_dataframe(mesh_num, t, t_write):
    """
    Create a DataFrame for the fields p, T, U for each time interval.
    
    Args:
        mesh_num (int): Mesh number to read from
        t (np.ndarray): Array of time intervals
        t_write (float): Write interval
        
    Returns:
        pd.DataFrame: DataFrame containing the field values
    """
    data = {
        'time': [],
        'p': [],
        'T': [],
        'Ux': [],
        'Uy': [],
        'Uz': []
    }
    
    for time in t[1:]:
        try:
            data['time'].append(time)
            data['p'].append(read_field_data(mesh_num, 'p', time, t_write))
            data['T'].append(read_field_data(mesh_num, 'T', time, t_write))
            U_values = read_field_data(mesh_num, 'U', time, t_write)
            U_array = np.array(U_values)
            data['Ux'].append(U_array[:, 0])  # All x components
            data['Uy'].append(U_array[:, 1])  # All y components
            data['Uz'].append(U_array[:, 2])  # All z components
        except Exception as e:
            print(f"Error reading data for time {time}: {e}")
            data['p'].append(None)
            data['T'].append(None)
            data['Ux'].append(None)
            data['Uy'].append(None)
            data['Uz'].append(None)

    return pd.DataFrame(data)

def load_data(mesh_num, time, label):
    """
    Load data from postProcessing files for a specific label (axial, post, pre).
    
    Args:
        mesh_num (int): Mesh number
        time (float/int): Time to read data for
        label (str): Label (axial, post, or pre)
        
    Returns:
        tuple: (p_data, U_data) where each is a dictionary containing coordinates and values
    """
    # Format the path with consistent time formatting
    formatted_time = format_time(time)
    base_path = f"data/mesh{mesh_num}/postProcessing/singleGraph/{formatted_time}"
    
    # Read pressure and temperature data
    p_data = {'x': [], 'y': [], 'z': [], 'p': [], 'T': []}
    p_file = f"{base_path}/{label}_p_T.xy"
    
    with open(p_file, 'r') as f:
        for line in f:
            x, y, z, p, T = map(float, line.strip().split())
            p_data['x'].append(x)
            p_data['y'].append(y)
            p_data['z'].append(z)
            p_data['p'].append(p)
            p_data['T'].append(T)
            
    # Read velocity data
    U_data = {'x': [], 'y': [], 'z': [], 'Ux': [], 'Uy': [], 'Uz': []}
    U_file = f"{base_path}/{label}_U.xy"
    
    with open(U_file, 'r') as f:
        for line in f:
            x, y, z, Ux, Uy, Uz = map(float, line.strip().split())
            U_data['x'].append(x)
            U_data['y'].append(y)
            U_data['z'].append(z)
            U_data['Ux'].append(Ux)
            U_data['Uy'].append(Uy)
            U_data['Uz'].append(Uz)
            
    return p_data, U_data

def plot_data(mesh_num, time):
    """
    Plot normalized pressure and velocity data in two separate figures with shared axes and grids.
    """
    # Get nozzle length for normalization
    L_nozz, _, _ = get_nozzle_end(mesh_num)
    
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12  # Base font size
    
    # Figure 1: Axial data (vs x)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    p_data, U_data = load_data(mesh_num, time, 'axial')
    # Normalize x coordinates and pressure
    x_norm = np.array(p_data['x']) / L_nozz
    p_norm = np.array(p_data['p']) / P_CHAMBER
    
    ax1.plot(x_norm, p_norm, label='Axial')
    ax1.grid()
    ax1.set_ylabel('Pressure (p/p₀)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                    np.array(U_data['Uy'])**2 + 
                    np.array(U_data['Uz'])**2)
    ax2.plot(x_norm, U_mag, label='Axial')
    ax2.grid()
    ax2.set_xlabel('x/L', fontsize=14)
    ax2.set_ylabel('Velocity Magnitude (U/U₀)', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add a main title for the figure with adjusted position
    plt.tight_layout()
    fig1.subplots_adjust(top=0.9)  # Make room for the title
    fig1.suptitle(f'Normalized Axial Distribution at t = {time}', fontsize=16, y=0.95)
    
    # Figure 2: Pre/Post data (vs y)
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    for label in ['pre', 'post']:
        p_data, U_data = load_data(mesh_num, time, label)
        # Normalize y coordinates and pressure
        y_norm = np.array(p_data['y']) / L_nozz
        p_norm = np.array(p_data['p']) / P_CHAMBER
        
        # Plot pressure
        ax3.plot(y_norm, p_norm, label=f'{label.capitalize()}-Nozzle Exit')
        ax3.grid()
        ax3.set_ylabel('Pressure (p/p₀)', fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.legend(fontsize=12)
        
        # Plot velocity magnitude
        U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                       np.array(U_data['Uy'])**2 + 
                       np.array(U_data['Uz'])**2)
        ax4.plot(y_norm, U_mag, label=f'{label.capitalize()}-Nozzle Exit')
        ax4.grid()
        ax4.set_xlabel('y/L', fontsize=14)
        ax4.set_ylabel('Velocity Magnitude (U/U₀)', fontsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        ax4.legend(fontsize=12)
    
    # Add a main title for the figure with adjusted position
    plt.tight_layout()
    fig2.subplots_adjust(top=0.9)  # Make room for the title
    fig2.suptitle(f'Normalized Nozzle Exit Distribution at t = {time}', fontsize=16, y=0.95)
    
    plt.show()

def plot_mesh_comparison(mesh_nums, time, show_nozzle_lines=True, mesh_names=None):
    """
    Create comparison plots across different mesh numbers for axial, pre-nozzle, and post-nozzle data.
    """
    if mesh_names is not None and len(mesh_names) != len(mesh_nums):
        raise ValueError("mesh_names must have the same length as mesh_nums")
    
    # Set global font settings
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    
    # Create a colormap for different meshes
    colors = [plt.cm.tab10(i) for i in range(len(mesh_nums))]
    
    # Create separate figures for axial (x-axis) and nozzle exits (y-axis)
    fig_axial, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Add dummy line for nozzle end legend if showing nozzle lines
    if show_nozzle_lines:
        ax1.plot([], [], '--', color='gray', label='Nozzle End', alpha=0.5)
    
    for i, mesh_num in enumerate(mesh_nums):
        mesh_label = mesh_names[i] if mesh_names is not None else f'Mesh {mesh_num}'
        
        # Get nozzle length for normalization
        L_nozz, _, _ = get_nozzle_end(mesh_num)
        
        p_data, U_data = load_data(mesh_num, time, 'axial')
        # Normalize coordinates and pressure
        x_norm = np.array(p_data['x']) / L_nozz
        p_norm = np.array(p_data['p']) / P_CHAMBER
        
        ax1.plot(x_norm, p_norm, label=mesh_label, color=colors[i])
        
        U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                       np.array(U_data['Uy'])**2 + 
                       np.array(U_data['Uz'])**2)
        ax2.plot(x_norm, U_mag, label=mesh_label, color=colors[i])
        
        if show_nozzle_lines:
            nozzle_x, _, _ = get_nozzle_end(mesh_num)
            ax1.axvline(x=nozzle_x/L_nozz, color=colors[i], linestyle='--', alpha=0.5)
            ax2.axvline(x=nozzle_x/L_nozz, color=colors[i], linestyle='--', alpha=0.5)
    
    ax1.grid()
    ax1.set_ylabel('Pressure (p/p₀)')
    ax1.legend()
    
    ax2.grid()
    ax2.set_xlabel('x/L')
    ax2.set_ylabel('Velocity Magnitude (U/U₀)')
    
    fig_axial.suptitle(f'Normalized Axial Distribution Comparison at t = {time}', fontsize=20, y=0.98)
    
    # Pre-nozzle plot (varies over y)
    fig_pre, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Add style indicators to legend first
    ax1.plot([], [], 'k-', label='Post-nozzle')
    ax1.plot([], [], 'k:', label='Pre-nozzle')
    
    for i, mesh_num in enumerate(mesh_nums):
        mesh_label = mesh_names[i] if mesh_names is not None else f'Mesh {mesh_num}'
        
        # Get nozzle length for normalization
        L_nozz, nozzle_y, _ = get_nozzle_end(mesh_num)
        
        # Plot pre data with dotted line
        p_data, U_data = load_data(mesh_num, time, 'pre')
        # Normalize coordinates and pressure
        y_norm = np.array(p_data['y']) / L_nozz
        p_norm = np.array(p_data['p']) / P_CHAMBER
        
        ax1.plot(y_norm, p_norm, ':', color=colors[i], label=mesh_label)
        if show_nozzle_lines:
            ax1.axvline(x=nozzle_y/L_nozz, color=colors[i], linestyle=':', alpha=0.5)
        
        U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                       np.array(U_data['Uy'])**2 + 
                       np.array(U_data['Uz'])**2)
        ax2.plot(y_norm, U_mag, ':', color=colors[i], label=mesh_label)
        if show_nozzle_lines:
            ax2.axvline(x=nozzle_y/L_nozz, color=colors[i], linestyle=':', alpha=0.5)
        
        # Plot post data with solid line
        p_data, U_data = load_data(mesh_num, time, 'post')
        # Normalize coordinates and pressure
        y_norm = np.array(p_data['y']) / L_nozz
        p_norm = np.array(p_data['p']) / P_CHAMBER
        
        ax1.plot(y_norm, p_norm, '-', color=colors[i])
        
        U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                       np.array(U_data['Uy'])**2 + 
                       np.array(U_data['Uz'])**2)
        ax2.plot(y_norm, U_mag, '-', color=colors[i])
    
    ax1.grid()
    ax1.set_ylabel('Pressure (p/p₀)')
    ax1.legend()
    
    ax2.grid()
    ax2.set_xlabel('y/L')
    ax2.set_ylabel('Velocity Magnitude (U/U₀)')
    
    fig_pre.suptitle(f'Normalized Nozzle Exit Distribution Comparison at t = {time}', fontsize=20, y=0.98)
    
    # Adjust layout for all figures
    for fig in [fig_axial, fig_pre]:
        plt.figure(fig.number)
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
    
    plt.show()

def plot_3d_evolution(mesh_num, times):
    """
    Create 3D plots showing the evolution of normalized pressure and velocity over time.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get nozzle length for normalization
    L_nozz, _, _ = get_nozzle_end(mesh_num)
    
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # Create figure for axial data
    fig_axial = plt.figure(figsize=(15, 10))
    ax_p_axial = fig_axial.add_subplot(121, projection='3d')
    ax_u_axial = fig_axial.add_subplot(122, projection='3d')
    
    # Create figures for pre/post data
    fig_nozzle = plt.figure(figsize=(15, 10))
    ax_p_nozzle = fig_nozzle.add_subplot(121, projection='3d')
    ax_u_nozzle = fig_nozzle.add_subplot(122, projection='3d')
    
    # Plot axial data (varies over x)
    for t in times:
        p_data, U_data = load_data(mesh_num, t, 'axial')
        # Normalize coordinates and pressure
        x_norm = np.array(p_data['x']) / L_nozz
        p_norm = np.array(p_data['p']) / P_CHAMBER
        
        # Create a mesh for time coordinate
        t_coords = np.full_like(x_norm, t)
        
        # Plot pressure (x, time, pressure)
        ax_p_axial.plot(x_norm, t_coords, p_norm, label=f't={t}')
        
        # Plot velocity magnitude (x, time, velocity)
        U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                       np.array(U_data['Uy'])**2 + 
                       np.array(U_data['Uz'])**2)
        ax_u_axial.plot(x_norm, t_coords, U_mag, label=f't={t}')
    
    # Set labels and titles for axial plots
    ax_p_axial.set_xlabel('x/L', fontsize=14)
    ax_p_axial.set_ylabel('Time', fontsize=14)
    ax_p_axial.set_zlabel('Pressure (p/p₀)', fontsize=14)
    ax_p_axial.set_title('Normalized Axial Pressure Evolution', fontsize=14)
    ax_p_axial.tick_params(axis='both', which='major', labelsize=12)
    
    ax_u_axial.set_xlabel('x/L', fontsize=14)
    ax_u_axial.set_ylabel('Time', fontsize=14)
    ax_u_axial.set_zlabel('Velocity Magnitude (U/U₀)', fontsize=14)
    ax_u_axial.set_title('Axial Velocity Evolution', fontsize=14)
    ax_u_axial.tick_params(axis='both', which='major', labelsize=12)
    
    fig_axial.suptitle(f'Normalized Axial Distribution Evolution (Mesh {mesh_num})', fontsize=16, y=0.95)
    
    # Plot pre/post nozzle data (varies over y)
    for location in ['pre', 'post']:
        for t in times:
            p_data, U_data = load_data(mesh_num, t, location)
            # Normalize coordinates and pressure
            y_norm = np.array(p_data['y']) / L_nozz
            p_norm = np.array(p_data['p']) / P_CHAMBER
            
            t_coords = np.full_like(y_norm, t)
            
            # Plot with different colors for pre/post
            color = 'blue' if location == 'pre' else 'red'
            
            # Plot pressure (y, time, pressure)
            ax_p_nozzle.plot(y_norm, t_coords, p_norm, 
                           label=f'{location.capitalize()}-t={t}',
                           color=color, alpha=0.6)
            
            # Plot velocity magnitude (y, time, velocity)
            U_mag = np.sqrt(np.array(U_data['Ux'])**2 + 
                          np.array(U_data['Uy'])**2 + 
                          np.array(U_data['Uz'])**2)
            ax_u_nozzle.plot(y_norm, t_coords, U_mag,
                           label=f'{location.capitalize()}-t={t}',
                           color=color, alpha=0.6)
    
    # Set labels and titles for nozzle plots
    ax_p_nozzle.set_xlabel('y/L', fontsize=14)
    ax_p_nozzle.set_ylabel('Time', fontsize=14)
    ax_p_nozzle.set_zlabel('Pressure (p/p₀)', fontsize=14)
    ax_p_nozzle.set_title('Normalized Nozzle Exit Pressure Evolution', fontsize=14)
    ax_p_nozzle.tick_params(axis='both', which='major', labelsize=12)
    
    ax_u_nozzle.set_xlabel('y/L', fontsize=14)
    ax_u_nozzle.set_ylabel('Time', fontsize=14)
    ax_u_nozzle.set_zlabel('Velocity Magnitude (U/U₀)', fontsize=14)
    ax_u_nozzle.set_title('Nozzle Exit Velocity Evolution', fontsize=14)
    ax_u_nozzle.tick_params(axis='both', which='major', labelsize=12)
    
    fig_nozzle.suptitle(f'Normalized Nozzle Exit Distribution Evolution (Mesh {mesh_num})', fontsize=16, y=0.95)
    
    # Adjust the view angles for better visualization
    for ax in [ax_p_axial, ax_u_axial, ax_p_nozzle, ax_u_nozzle]:
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

def calculate_thrust(mesh_num, times):
    """
    Calculate thrust force at each time step using normalized pressure values and ideal gas law for density.
    
    The ideal gas law is used to calculate density: rho = p / (RT)
    where:
    - R is the specific gas constant for air (287 J/kg·K)
    - p is the absolute pressure (normalized pressure * P_CHAMBER)
    - T is the temperature in Kelvin
    """
    # Get the axial_y value for area calculations and nozzle length
    L_nozz, _, axial_y = get_nozzle_end(mesh_num)
    dA = axial_y  # Area element
    
    # Constants
    R = 287.0  # Specific gas constant for air [J/kg·K]
    v_in = 1.0  # Inlet velocity (normalized)
    
    thrust_values = []
    
    for t in times:
        # Load pre and post data
        pre_p_data, pre_U_data = load_data(mesh_num, t, 'pre')
        post_p_data, post_U_data = load_data(mesh_num, t, 'post')
        
        # Calculate absolute pressures (convert from normalized)
        p_post_abs = np.array(post_p_data['p']) * P_CHAMBER
        p_pre_abs = np.array(pre_p_data['p']) * P_CHAMBER
        
        # Get temperatures
        T_post = np.array(post_p_data['T'])
        
        # Calculate inlet mass flow using the temperature at the first point
        mdot_in = 0.011 * (P_CHAMBER / (R * T_post[0]))
        Fmom_in = mdot_in * v_in
        
        # Calculate density using ideal gas law at each point
        rho_post = p_post_abs / (R * T_post)
        
        # Calculate momentum force at exit (using post data)
        # Fmom = sum(rho * Ux^2 * dA)
        Ux_post = np.array(post_U_data['Ux'])
        Fmom = np.sum(rho_post * Ux_post**2) * dA
        
        # Calculate pressure force
        # Fpres = sum((p_post - p_pre) * dA)
        Fpres = np.sum((p_pre_abs - p_post_abs)) * dA  # Note the reversed order
        
        # Calculate total thrust with small constant offset
        thrust = Fmom + Fpres - Fmom_in + 0.0005  # Add small constant offset
        thrust_values.append(thrust)
    
    return np.array(times), np.array(thrust_values)

def plot_thrust(mesh_num, times):
    """
    Plot thrust force evolution over time.
    
    Args:
        mesh_num (int): Mesh number to analyze
        times (list/array): Time points to plot thrust for
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    
    # Calculate thrust values
    t, thrust = calculate_thrust(mesh_num, times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot thrust vs time
    ax.plot(t, thrust, 'b-', linewidth=2)
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Thrust Force')
    ax.set_title(f'Thrust Evolution (Mesh {mesh_num})')
    
    plt.tight_layout()
    plt.show()
    
    return t, thrust

def get_expansion_ratio(mesh_num, time):
    """
    Calculate the expansion ratio (exit area / throat area) for a given mesh at a specific time.
    The area is proportional to the square of the y-coordinate (like a circular nozzle where A = πr²).
    
    Args:
        mesh_num (int): Mesh number
        time (float): Time point to analyze
        
    Returns:
        float: Expansion ratio (Ae/At)
    """
    expansion_dict = {3: 1, 4: .85, 5: .9, 6: 1.1, 7: 1.2}
    expansion_scale = expansion_dict[mesh_num]
    width_scale = 5.  # Scale factor for all dimensions (1.0 is original size)
    
    # Base nozzle coordinates
    x_nozz = np.array([0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050, 0.0054,
                    0.0059, 0.0065, 0.0070, 0.0907, 0.1181, 0.1433, 0.1688, 0.1958, 0.2248, 0.2563, 0.2909,
                    0.3291, 0.3715, 0.4185, 0.4710, 0.5297, 0.5956, 0.6695])
    
    y_nozz_base = np.array([0.0500, 0.0500, 0.0500, 0.0500, 0.0501, 0.0501, 0.0502, 0.0503, 0.0504, 0.0505, 0.0507,
                    0.0508, 0.0510, 0.0512, 0.0513, 0.0858, 0.0962, 0.1051, 0.1133, 0.1212, 0.1290, 0.1365,
                    0.1438, 0.1508, 0.1575, 0.1636, 0.1691, 0.1737, 0.1771, 0.1790])

    # Apply width scaling to base coordinates
    x_nozz = x_nozz * width_scale
    y_nozz_base = y_nozz_base * width_scale

    # Find throat index (minimum y value in first few points)
    throat_idx = np.argmin(y_nozz_base[:15])
    throat_y = y_nozz_base[throat_idx]
    
    # Scale y coordinates after throat while keeping throat and inlet constant
    y_nozz = y_nozz_base.copy()
    y_nozz[throat_idx:] = throat_y + (y_nozz_base[throat_idx:] - throat_y) * expansion_scale
    
    # Calculate and print expansion ratio (exit area / throat area)
    expansion_ratio = y_nozz[-1] ** 2 / throat_y ** 2
    return expansion_ratio
    

def plot_thrust_vs_expansion(mesh_nums, time):
    """
    Plot thrust force versus expansion ratio at a given time.
    
    Args:
        mesh_nums (list): List of mesh numbers to analyze
        time (float): Time point to analyze thrust at
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    
    # Calculate thrust and expansion ratio for each mesh
    thrusts = []
    ratios = []
    
    for mesh_num in mesh_nums:
        try:
            # Calculate expansion ratio
            expansion_ratio = get_expansion_ratio(mesh_num, time)
            ratios.append(expansion_ratio)
            
            # Calculate thrust at the specific time
            t, thrust = calculate_thrust(mesh_num, [time])
            thrusts.append(thrust[0])  # Take first (and only) thrust value
            print(f"Mesh {mesh_num}: ε = {expansion_ratio:.3f}, Thrust = {thrust[0]:.3f}")
        except Exception as e:
            print(f"Error processing mesh {mesh_num}: {e}")
            continue
    
    if not thrusts:  # If no data was successfully processed
        print("No valid data to plot")
        return [], []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by expansion ratio for proper line plotting
    sorted_indices = np.argsort(ratios)
    ratios = np.array(ratios)[sorted_indices]
    thrusts = np.array(thrusts)[sorted_indices]
    mesh_nums = np.array(mesh_nums)[sorted_indices]
    
    # Plot thrust vs expansion ratio
    ax.plot(ratios, thrusts, 'b-', marker='o', linewidth=2, markersize=8)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('Expansion Ratio (ε = A$_e$/A$_t$)')
    ax.set_ylabel('Thrust Force')
    ax.set_title(f'Thrust vs Expansion Ratio at t = {time}')
    
    # Add mesh numbers as annotations
    for i, mesh_num in enumerate(mesh_nums):
        ax.annotate(f'Mesh {mesh_num}\nε = {ratios[i]:.3f}', 
                   (ratios[i], thrusts[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return ratios, thrusts

def main():
    mesh_num = 3
    mesh_nums = [3, 4, 5, 6, 7]
    time = 10
    times = [1, 2.5, 5, 7.5, 10]
    
    # Plot thrust vs expansion ratio at different times
    # for time in times:
    plot_thrust_vs_expansion(mesh_nums, time)
    
    # Example with default mesh names
    # plot_mesh_comparison(mesh_nums, time, show_nozzle_lines=True)
    
    # plot_data_time_series(mesh_num, times, plot_type='nozzle')
    # plot_data_time_series(mesh_num, times, plot_type='axial')
    
    # Example with custom mesh names
    # mesh_names = ['Coarse', 'Medium', 'Fine']
    # plot_mesh_comparison([1, 2, 3], time, show_nozzle_lines=False, mesh_names=['Mesh 1 (CV=24964)', 'Mesh 2 (CV=69432)', 'Mesh 3 (CV=99856)'])

if __name__ == "__main__":
    main()
