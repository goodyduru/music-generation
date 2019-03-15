import glob
import numpy as np

from collections import Counter
from keras.utils import np_utils
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from music21 import converter, chord, note, instrument, stream

def get_notes():
    notes = []
    notes_offset = []
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
                notes_offset.append(element.offset)
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                notes_offset.append(element.offset)
    return notes, notes_offset

def create_datset(notes, n_vocab, sequence_length=100):
    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((element, number) for number, element in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i+sequence_length]
        sequence_out = notes[i+sequence_length]

        network_input.append([note_to_int[element] for element in sequence_in])
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

def predict(model, network_input, notes, n_vocab, num_notes = 500):
    pitchnames = sorted(set(item for item in notes))

    int_to_note = dict((number, element) for number, element in enumerate(pitchnames))

    start = np.random.randint(0, network_input.shape[0] - 1)
    pattern = network_input[start].aslist()
    pattern_length = len(pattern)

    prediction_output = []

    for _ in range(num_notes):
        prediction_input = np.reshape(pattern, (1, pattern_length, 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input)

        predicted_note = np.argmax(prediction)
        result = int_to_note[predicted_note]
        prediction_output.append(result)

        pattern.append(result)
        pattern = pattern[-pattern_length:]
    return prediction_output

def generate_mid(prediction_output):
    offset = 0
    output = []

    for element in prediction_output:
        # element is a chord
        if element.find('.') > -1 or element.isdigit():
            notes_in_chord = element.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output.append(new_chord)
        # element is a note
        else:
            new_note = note.Note(element)
            new_note.storedInstrument = instrument.Piano()
            new_note.offset = offset
            output.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output)
    midi_stream.write('midi', fp='generated.mid')


if __name__ == "__main__":
    notes, notes_offset = get_notes()
    n_vocab = len(set(notes))
    print(n_vocab)
    network_input, network_output = create_datset(notes, n_vocab)
    print(network_input.shape, network_output.shape)
    sequence, end = network_input.shape[1], network_input.shape[2]
    model = create_model((sequence, end), n_vocab)
    print(model.summary())
    analyze = []
    for i, n in enumerate(notes_offset):
        if i == 0 or n < notes_offset[i-1]:
            continue
        analyze.append(n - notes_offset[i-1])
    counter = Counter(analyze)
    print(counter.most_common(5))

