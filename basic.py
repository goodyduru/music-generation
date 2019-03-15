import glob
import numpy as np

from keras.utils import np_utils
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
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

def create_datset(notes, n_vocab, sequence_length=100):
    pitchnames = sorted(set(item for item in notes))

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

def create_model(input_shape, n_vocab):
    x = Input(shape=input_shape)
    out = LSTM(256, return_sequences=True)(x)
    out = Dropout(0.3)(out)
    out = LSTM(512, return_sequences=True)(out)
    out = Dropout(0.3)(out)
    out = LSTM(256)(out)
    out = Dense(256)(out)
    out = Dropout(0.3)(out)
    out = Dense(n_vocab, activation='softmax')(out)
    model = Model(inputs=x, outputs=out)
    return model

if __name__ == "__main__":
    notes = get_notes()
    n_vocab = len(set(notes))
    print(n_vocab)
    network_input, network_output = create_datset(notes, n_vocab)
    print(network_input.shape, network_output.shape)
    sequence, end = network_input.shape[1], network_input.shape[2]
    model = create_model((sequence, end), n_vocab)
    print(model.summary())
