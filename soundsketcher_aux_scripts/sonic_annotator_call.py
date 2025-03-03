import subprocess
import os

def run_sonic_annotator(input_wav_path, output_csv_dir):
    # Change directory to where sonic-annotator executable is located
    os.chdir("/media/datadisk/velenisrepos/soundsketcher/sonic-annotator-1.6-linux64-static")
    # Construct the Sonic Annotator command
    command = [
        './sonic-annotator',
        '-t', 'periodicity.n3', '-t', 'f0Candidates.n3',
        input_wav_path, 
        '-w', 'csv',
        '--csv-basedir', output_csv_dir
    ]
#'-t', 'spectral-kurtosis.n3','-t', 'yin-f0.n3',
    try:
        # Run the Sonic Annotator command
        subprocess.run(command, check=True)
        print('Sonic Annotator completed successfully.')
        os.chdir("/media/datadisk/velenisrepos/soundsketcher")
    except subprocess.CalledProcessError as e:
        print(f'Error running Sonic Annotator: {e}')


# input_wav_path = '../input_wav/synth_sound_2.wav'
# output_csv_dir = '../output_csv'
# audio_feature = 'vamp:bbc-vamp-plugins:bbc-rhythm:onset'
# run_sonic_annotator(audio_feature, input_wav_path, output_csv_dir)
#test