import numpy as np

ref_string = "0000001010101100110010101110010011010010110011001101001011000110110000101110100011010010110111101101110001000000100101101100101011110010011101110101111011110001001110011111011011101000100011111110110"


def bit_str2arr(string: str):
    return np.array(list(string)).astype(int)


def eval_ber_for(bit_arr: np.ndarray):
    ref_arr = bit_str2arr(ref_string)

    length_factor = bit_arr.size // ref_arr.size + 3
    ref_arr = np.tile(ref_arr, length_factor)

    bit_matches = np.empty((199, bit_arr.size), dtype=int)
    ber = np.ones(199)

    for i in range(0, 199 * 2, 2):
        bit_matches[i // 2] = np.equal(bit_arr, ref_arr[i:i + bit_arr.size])
        ber[i // 2] = 1 - bit_matches[i // 2].sum() / bit_arr.size
        # print(f'i={i:03d} ber={ber[i]:.03f}')

    channel_offset_left = 2 * find_channel_offset(bit_matches[:, 0::2], filter_size=200)
    channel_offset_right = 2 * find_channel_offset(bit_matches[:, 1::2], filter_size=200)
    # channel_offset = find_channel_offset(bit_matches)
    channel_offset = np.zeros((len(channel_offset_left) + len(channel_offset_right)), dtype=int)
    channel_offset[::2] = channel_offset_left
    channel_offset[1::2] = channel_offset_right

    bit_matches_with_offset = bit_matches[channel_offset // 2, range(bit_arr.size)]
    ber_with_offset = 1 - bit_matches_with_offset.sum() / bit_arr.size
    moving_ber = calc_moving_ber(bit_matches_with_offset, size=300)  # Size - 30FPS: 300, 240FPS: 2400 for 5 Seconds

    return bit_matches_with_offset, channel_offset, moving_ber, ber_with_offset


def find_channel_offset(bit_matches, window_size=200, filter_size=50):
    channel_offset = np.zeros(bit_matches.shape[1] - window_size, dtype=int)

    for i in range(bit_matches.shape[1] - window_size):
        bit_matches_window = bit_matches[:, i: i + window_size]
        bit_matches_sum = np.sum(bit_matches_window, axis=1)
        channel_offset[i] = np.argmax(bit_matches_sum)

    channel_offset_filtered = np.zeros(bit_matches.shape[1], dtype=int)
    channel_offset_filtered[:] = -1

    if channel_offset.size <= filter_size:
        filter_size = channel_offset.size

    for i in range(channel_offset.size - filter_size):
        offset_window = channel_offset[i: i + filter_size]
        if np.min(offset_window) == np.max(offset_window):
            channel_offset_filtered[window_size // 2 + i] = channel_offset[i]

    first_correct_offset = np.argmax(channel_offset_filtered >= 0)
    channel_offset_filtered[:first_correct_offset] = channel_offset_filtered[first_correct_offset]

    for i in range(channel_offset_filtered.size):
        if channel_offset_filtered[i] < 0:
            channel_offset_filtered[i] = channel_offset_filtered[i - 1]

    return channel_offset_filtered


def calc_moving_ber(bit_matches, size=300):
    moving_ber = np.zeros_like(bit_matches, dtype=float)
    for i in range(bit_matches.size):
        start = max(0, i - size // 2)
        end = min(bit_matches.size, i + size // 2)
        bit_matches_part = bit_matches[start:end]
        moving_ber[i] = 1 - bit_matches_part.sum() / bit_matches_part.size
    return moving_ber
