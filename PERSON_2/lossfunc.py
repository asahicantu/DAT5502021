from helpers import mid2arry
import numpy as np


def midiloss(reference, generated):
    reference_arr = mid2arry(reference)
    generated_arr = mid2arry(generated)

    len_reference = len(reference_arr)
    len_generated = len(generated_arr)

    if len_reference > len_generated:
        generated_arr = np.pad(generated_arr, (0,len_reference-len_generated), 'constant', constant_values=(0))

    if len_reference < len_generated:
        reference_arr = np.pad(reference_arr, (0,len_generated-len_reference), 'constant', constant_values=(0))

    loss = np.sum(np.square(reference_arr - generated_arr)) / len_generated

    return loss