import glob
from music21 import converter, chord, note, instrument

def prepare_dataset():
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

if __name__ == "__main__":
    notes = prepare_dataset()
    print(len(notes))
