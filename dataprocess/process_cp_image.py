import os
import numpy as np
import tifffile as tiff
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
with open('data_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

data_dir =  config['raw_data_root']
output_dir = config['data_root']
os.makedirs(output_dir, exist_ok=True)

channels = ['ERSyto', 'ERSytoBleed', 'Hoechst', 'Mito', 'Ph_golgi']
plate_folders = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) and '24609' in name]
plates = set(folder.split('-')[0] for folder in plate_folders)

progress_file = os.path.join(output_dir, 'processing_progress.txt')

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        processed = {line.strip() for line in f if '24609' in line}
else:
    processed = set()

def process_well_view(plate_well_view):
    plate, well_view = plate_well_view
    output_filename = f"{well_view.replace('_', '-')}.npy"
    output_filepath = os.path.join(output_dir, plate, output_filename)
    if output_filepath in processed:
        return f"Skipped {plate}-{well_view}"
    
    images = []
    for channel in channels:
        channel_dir = os.path.join(data_dir, f'{plate}-{channel}')
        matching_files = glob(os.path.join(channel_dir, f'*{well_view}_*.tif'))
        if matching_files:
            image_path = matching_files[0]
            image = tiff.imread(image_path)
            image_8bit = (image >> 8).astype('uint8')
            images.append(image_8bit)

    if len(images) == 5:
        combined_image = np.stack(images, axis=-1)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        np.save(output_filepath, combined_image)
        with open(progress_file, 'a') as f:
            f.write(output_filepath + '\n')
        return f"Processed {plate}-{well_view}"
    else:
        return f"Skipped {plate}-{well_view} due to missing channels"

num_cores = 16  
futures = []
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    for plate in plates:
        plate_output_dir = os.path.join(output_dir, plate)
        os.makedirs(plate_output_dir, exist_ok=True)
        channel_dir = os.path.join(data_dir, f'{plate}-ERSyto')
        if os.path.isdir(channel_dir):
            channel_files = os.listdir(channel_dir)
            well_view_combinations = ['_'.join(f.split('_')[1:3]) for f in channel_files]
            for well_view in well_view_combinations:
                if os.path.join(plate_output_dir, f"{well_view.replace('_', '-')}.npy") not in processed:
                    future = executor.submit(process_well_view, (plate, well_view))
                    futures.append(future)

    for future in as_completed(futures):
        print(future.result())

print("Done!")
