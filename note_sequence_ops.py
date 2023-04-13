import itertools
import numpy as np

QUANTIZE_CUTOFF = 0.5
PITCH_CUTOFF = 20
STEP_CUTOFF = 16


class NegativeTimeError(Exception):
    pass


# Public ops
def quantize_note_sequence(note_sequence, steps_per_quarter):
    note_sequence.quantization_info.steps_per_quarter = steps_per_quarter
    steps_per_second = steps_per_quarter_to_steps_per_second(
        steps_per_quarter, note_sequence.tempos[0].qpm)
    note_sequence.total_quantized_steps = quantize_to_step(note_sequence.total_time, steps_per_second)
    return _quantize_notes(note_sequence, steps_per_second)


def steps_per_bar_in_quantized_sequence(note_sequence):
    """Calculates steps per bar in a NoteSequence that has been quantized.
    Args:
      note_sequence: The NoteSequence to examine.
    Returns:
      Steps per bar as a floating point number.
    """
    assert note_sequence.quantization_info.steps_per_quarter > 0

    quarters_per_beat = 4.0 / note_sequence.time_signatures[0].denominator
    quarters_per_bar = (
        quarters_per_beat * note_sequence.time_signatures[0].numerator)
    steps_per_bar_float = (
        note_sequence.quantization_info.steps_per_quarter * quarters_per_bar)
    return steps_per_bar_float


# Private ops

def steps_per_quarter_to_steps_per_second(steps_per_quarter, qpm):
    """Calculates steps per second given steps_per_quarter and a qpm."""
    return steps_per_quarter * qpm / 60.0


def steps_per_quarter_to_seconds_per_step(steps_per_quarter, qpm):
    return 60.0 / steps_per_quarter / qpm


def quantize_to_step(unquantized_seconds,
                     steps_per_second,
                     quantize_cutoff=QUANTIZE_CUTOFF):
    """Quantizes seconds to the nearest step, given steps_per_second.
    Args:
      unquantized_seconds: Seconds to quantize.
      steps_per_second: Quantizing resolution.
      quantize_cutoff: Value to use for quantizing cutoff.
    Returns:
      The input value quantized to the nearest step.
    """
    unquantized_steps = unquantized_seconds * steps_per_second
    return int(unquantized_steps + (1 - quantize_cutoff))


def _quantize_to_step(unquantized_seconds,
                      steps_per_second,
                      quantize_cutoff=QUANTIZE_CUTOFF):
    """Quantizes seconds to the nearest step, given steps_per_second.
    See the comments above `QUANTIZE_CUTOFF` for details on how the quantizing
    algorithm works.
    Args:
      unquantized_seconds: Seconds to quantize.
      steps_per_second: Quantizing resolution.
      quantize_cutoff: Value to use for quantizing cutoff.
    Returns:
      The input value quantized to the nearest step.
    """
    unquantized_steps = unquantized_seconds * steps_per_second
    return int(unquantized_steps + (1 - quantize_cutoff))


def _quantize_notes(note_sequence, steps_per_second):
    """Quantize the notes and chords of a NoteSequence proto in place.
    Note start and end times, and chord times are snapped to a nearby quantized
    step, and the resulting times are stored in a separate field (e.g.,
    quantized_start_step).
    Args:
      note_sequence: A music_pb2.NoteSequence protocol buffer. Will be modified in
        place.
      steps_per_second: Each second will be divided into this many quantized time
        steps.
    Raises:
      NegativeTimeError: If a note or chord occurs at a negative time.
    """
    for note in note_sequence.notes:
        # Quantize the start and end times of the note.
        note.quantized_start_step = _quantize_to_step(note.start_time,
                                                      steps_per_second)
        note.quantized_end_step = _quantize_to_step(
            note.end_time, steps_per_second)
        if note.quantized_end_step == note.quantized_start_step:
            note.quantized_end_step += 1

        # Do not allow notes to start or end in negative time.
        if note.quantized_start_step < 0 or note.quantized_end_step < 0:
            raise NegativeTimeError(
                'Got negative note time: start_step = %s, end_step = %s' %
                (note.quantized_start_step, note.quantized_end_step))

        # Extend quantized sequence if necessary.
        if note.quantized_end_step > note_sequence.total_quantized_steps:
            note_sequence.total_quantized_steps = note.quantized_end_step

    # Also quantize control changes and text annotations.
    for event in note_sequence.control_changes:
        # Quantize the event time, disallowing negative time.
        event.quantized_step = _quantize_to_step(event.time, steps_per_second)
        if event.quantized_step < 0:
            raise NegativeTimeError(
                'Got negative event time: step = %s' % event.quantized_step)

    return note_sequence


def find_start_time(ns):
    ns.notes = sorted(ns.notes, key=lambda note: note.start_time)
    for note in ns.notes:
            return note.start_time
    print('cant find')
    return 0


def delete_auftakt(ns):
    start_time = find_start_time(ns)
    filtered_notes = []
    for note in ns.notes:
        note.start_time = max(note.start_time - start_time, 0.)
        note.end_time = max(note.end_time - start_time, 0.)
        if note.end_time - note.start_time > 0.00001:
            filtered_notes.append(note)
    ns.notes = filtered_notes
    return ns
