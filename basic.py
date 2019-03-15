import glob
import numpy as np

from keras.utils import np_utils
from music21 import converter, chord, note, instrument

def get_notes():
    notes = []
    for midi_file in glob.glob('./data/*.mid'):
        print("Processing %s" % midi_file)
        midi = converter.parse(midi_file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def create_datset(notes, sequence_length=100):
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(pitchnames)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i+sequence_length]
        sequence_out = notes[i+sequence_length]

        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    num_patterns = len(network_input)

    network_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
    # normalize network
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    return (network_input, network_output)


if __name__ == "__main__":
    notes = get_notes()
    network_input, network_output = create_datset(notes)
    print(network_input.shape, network_output.shape)
