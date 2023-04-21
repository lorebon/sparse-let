from pathlib import Path
import numpy as np
import pandas as pd
import os

def read_sequences(file_path, cast_to_int=False):
    with open(file_path, "r") as f:
        lines = f.readlines()

    sequences = []
    for l in lines:
        interval = "".join(l.split("\n")).split(" ")
        seqence_id = int(interval[0])
        while seqence_id >= len(sequences):
            sequences.append([])
        event_id = int(interval[1]) - 1
        while event_id >= len(sequences[seqence_id]):
            sequences[seqence_id].append([])
        start_timestamp = int(interval[2]) if cast_to_int else float(interval[2])
        end_timestamp = int(interval[3]) if cast_to_int else float(interval[3])
        if len(interval) == 4:
            value = 1
        else:
            value = float(interval[4])
        sequences[seqence_id][event_id].append([start_timestamp, end_timestamp, value])
    
    max_events = max([len(evts) for evts in sequences])
    for evts in sequences:
        while len(evts) < max_events:
            evts.append([])

    return sequences

def read_classes(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        classes = []
        for i in range(len(lines)):
            classes.append(int(lines[i]))
    return classes

def load_auslan2(cast_to_int=False):
    sequences = read_sequences(Path('data/AUSLAN2/data.txt'), cast_to_int)
    ys = read_classes(Path('data/AUSLAN2/classes.txt'))
    return sequences, ys

def load_fedegari(cast_to_int=False):
    sequences = read_sequences(Path('data/FEDEGARI/data.txt'), cast_to_int)
    ys = read_classes(Path('data/FEDEGARI/classes.txt'))
    return sequences, ys

def load_blocks(cast_to_int=False):
    sequences = read_sequences(Path('data/BLOCKS/data.txt'), cast_to_int)
    ys = read_classes(Path('data/BLOCKS/classes.txt'))
    return sequences, ys

def load_context(cast_to_int=False):
    sequences = read_sequences(Path('data/CONTEXT/data.txt'), cast_to_int)
    ys = read_classes(Path('data/CONTEXT/classes.txt'))
    return sequences, ys

def load_hepatitis(cast_to_int=False):
    sequences = read_sequences(Path('data/HEPATITIS/data.txt'), cast_to_int)
    ys = read_classes(Path('data/HEPATITIS/classes.txt'))
    return sequences, ys

def load_pioneer(cast_to_int=False):
    sequences = read_sequences(Path('data/PIONEER/data.txt'), cast_to_int)
    ys = read_classes(Path('data/PIONEER/classes.txt'))
    return sequences, ys

def load_skating(cast_to_int=False):
    sequences = read_sequences(Path('data/SKATING/data.txt'), cast_to_int)
    ys = read_classes(Path('data/SKATING/classes.txt'))
    return sequences, ys

def get_all_old_methods():
    return [load_auslan2, load_blocks, load_context, load_hepatitis, load_pioneer, load_skating]

def load_musekey_with_intensity(cast_to_int=False):
    sequences = read_sequences(Path('data/maestro/data_intensities.txt'), cast_to_int)
    ys = read_classes(Path('data/maestro/classes_key.txt'))
    return sequences, ys

def load_musekey(cast_to_int=False):
    sequences = read_sequences(Path('data/maestro/data_20.txt'), cast_to_int)
    ys = read_classes(Path('data/maestro/classes_key_20.txt'))
    return sequences, ys

def get_piece_titles():
    with open(Path('data/maestro/piece_names.txt'), "r") as f:
        lines = f.readlines()
    return lines

def get_all_music_methods():
    return [load_musekey]

def load_weather(cast_to_int=False):
    sequences = read_sequences(Path('data/weather/data.txt'), cast_to_int)
    ys = read_classes(Path('data/weather/classes.txt'))
    return sequences, ys

def load_weather_with_intensity(cast_to_int=False):
    sequences = read_sequences(Path('data/weather/data_intensities.txt'), cast_to_int)
    ys = read_classes(Path('data/weather/classes.txt'))
    return sequences, ys

def get_all_weather_methods():
    return [load_weather]

def get_all_intensity_methods():
    return [load_weather_with_intensity, load_musekey_with_intensity]
