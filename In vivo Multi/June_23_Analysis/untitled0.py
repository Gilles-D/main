import spikeinterface as si
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw

import probeinterface as pi
from probeinterface.plotting import plot_probe

#%%Parameters

probe_path=r'D:/ePhy/SI_Data/A1x16-Poly2-5mm-50s-177.json'   #INTAN Optrode
# probe_path = 'D:/ePhy/SI_Data/Buzsaki16.json'              #INTAN Buzsaki16

# Saving Folder path
saving_dir=r"D:/ePhy/SI_Data/concatenated_signals"
saving_name="0012_session_3_allchan_baseline"

spikesorting_results_folder='D:\ePhy\SI_Data\spikesorting_results'

recordings=[

'D:/ePhy/Intan_Data/0012/07_12/0012_12_07_230712_182326/0012_12_07_230712_182326.rhd',

    ]

excluded_sites = ['1','2','3','12','13']
excluded_sites = ['4','5','14','15']
excluded_sites = []
freq_min=300
freq_max=6000

window = [83,84]


#%%

"""------------------Concatenation------------------"""
recordings_list=[]
for recording_file in recordings:
    recording = se.read_intan(recording_file,stream_id='0')
    recording.annotate(is_filtered=False)
    recordings_list.append(recording)

multirecording = si.concatenate_recordings(recordings_list)

"""------------------Set the probe------------------"""
probe = pi.io.read_probeinterface(probe_path)
probe = probe.probes[0]
multirecording = multirecording.set_probe(probe)


"""------------------Defective sites exclusion------------------"""

multirecording.set_channel_groups(1, excluded_sites)
multirecording = multirecording.split_by('group')[0]

sw.plot_timeseries(multirecording, channel_ids=multirecording.get_channel_ids(),time_range=window)


"""------------------Pre Processing------------------"""
#Bandpass filter
recording_f = spre.bandpass_filter(multirecording, freq_min=freq_min, freq_max=freq_max)

w = sw.plot_timeseries(recording_f,time_range=window, segment_index=0)


#Median common ref
recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')

w = sw.plot_timeseries(recording_cmr,time_range=window, segment_index=0)

