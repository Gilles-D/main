import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les données
spike_times_data = pd.read_excel(r"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/spike_times.xlsx")
session_files = [
    # "0022_01_3_mocap.xlsx", "0022_01_4_mocap.xlsx", "0022_01_5_mocap.xlsx",
    #              "0022_01_6_mocap.xlsx", "0022_01_7_mocap.xlsx", "0022_01_8_mocap.xlsx",
    #              "0022_01_9_mocap.xlsx", "0022_01_10_mocap.xlsx", "0022_01_11_mocap.xlsx",
    #               "0022_01_12_mocap.xlsx", "0022_01_13_mocap.xlsx", 
                   "0022_01_14_mocap.xlsx",
                   "0022_01_15_mocap.xlsx", "0022_01_16_mocap.xlsx", "0022_01_17_mocap.xlsx",
                   "0022_01_18_mocap.xlsx", "0022_01_19_mocap.xlsx", "0022_01_20_mocap.xlsx",
                   "0022_01_21_mocap.xlsx", "0022_01_22_mocap.xlsx"
                 ]
mocap_sessions = {}

plot_path = r"\\equipe2-nas1\Public\DATA\Gilles\Spikesorting_August_2023\SI_Data\spikesorting_results\0022_01_08\kilosort3\curated\processing_data\plots\spikes_location_on_setup"

for file in session_files:
    print(file)
    session_name = "_".join(file.split('_')[1:3])
    mocap_sessions[session_name] = pd.read_excel(rf"//equipe2-nas1/Public/DATA/Gilles/Spikesorting_August_2023/SI_Data/spikesorting_results/0022_01_08/kilosort3/curated/processing_data/sync_data_rate_sigma_20.0 msms_Gaussian/{file}")

# 2. Fonctions utiles
def find_closest_times(spike, time_axis):
    """Trouve les points temporels avant et après le spike en utilisant la recherche binaire."""
    index = time_axis.searchsorted(spike)
    before_time = time_axis.iloc[index - 1]
    after_time = time_axis.iloc[index]
    return before_time, after_time

def plot_interpolated_positions_with_equal_axes(unit_name, positions_by_session,plot_path):
    plt.figure(figsize=(10, 6))
    
    # Tracer le rectangle
    rect = plt.Rectangle((-370, -15), 456, 30, fill=False, edgecolor='black', linewidth=1.5)
    plt.gca().add_patch(rect)
    
    for session, (x_positions, y_positions) in positions_by_session.items():
        plt.scatter(x_positions, y_positions, label=session, alpha=0.7, color='red', s=1)
    plt.title(f"Positions interpolées (x,y) de back1 pour {unit_name}")
    
    plt.xlim()
    plt.gca().invert_xaxis()
    
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    # plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Mettre la même échelle pour les deux axes
    
    plt.show()
    plt.savefig(rf'{plot_path}/obstacle_unit_{unit_name}.svg')
    print(rf'{plot_path}/obstacle_unit_{unit_name}.svg')

# 3. Traitement des données
positions_by_session_dict = {unit: {} for unit in spike_times_data.columns[1:]}
for session, data in mocap_sessions.items():
    time_axis_session = data['time_axis']
    start_time_session = time_axis_session.min()
    end_time_session = time_axis_session.max()
    
    for unit in spike_times_data.columns[1:]:
        x_positions = []
        y_positions = []
        spikes = spike_times_data[unit].dropna()
        valid_spikes = spikes[(spikes >= start_time_session) & (spikes <= end_time_session)]
        
        for spike in valid_spikes:
            if spike in time_axis_session.values:
                x_pos = data.loc[data['time_axis'] == spike, 'back1_x'].values[0]
                y_pos = data.loc[data['time_axis'] == spike, 'back1_y'].values[0]
            else:
                before_time, after_time = find_closest_times(spike, time_axis_session)
                x_before = data.loc[data['time_axis'] == before_time, 'back1_x'].values[0]
                x_after = data.loc[data['time_axis'] == after_time, 'back1_x'].values[0]
                x_pos = x_before + (x_after - x_before) * (spike - before_time) / (after_time - before_time)
                
                y_before = data.loc[data['time_axis'] == before_time, 'back1_y'].values[0]
                y_after = data.loc[data['time_axis'] == after_time, 'back1_y'].values[0]
                y_pos = y_before + (y_after - y_before) * (spike - before_time) / (after_time - before_time)
            
            x_positions.append(x_pos)
            y_positions.append(y_pos)
        
        positions_by_session_dict[unit][session] = (x_positions, y_positions)

# 4. Visualisation
for unit in spike_times_data.columns[1:]:
    plot_interpolated_positions_with_equal_axes(unit, positions_by_session_dict[unit],plot_path)

