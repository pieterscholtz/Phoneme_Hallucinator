import os
# from os import PathLike
from typing import Any, BinaryIO, Optional, Union
import struct
import json
import torch
from functools import reduce
from operator import mul
import pathlib

# Dictionary mapping data types to their corresponding torch data type and byte size
DTYPE_BYTE_MAPPING = {
    'F16': [torch.float16, 2],
    'F32': [torch.float32, 4],
    'I16': [torch.int16, 2],
}

def read_safetensors_file(
        file_path: Union[str, int, pathlib.PurePath, BinaryIO],
        key: Optional[str] = None,
        offset: int = 0,
        length: Optional[int] = None,
        length_index: Optional[int] = None) -> Union[Any, torch.Tensor]:
    """ reads `key` from `file_path` safetensors file that has arbitrary shape
    The file is read from `offset` (inclusive) to `offset + length` (exclusive).

    Args:
        file_path (Union[str, int, os.PathLike[Any], BinaryIO]): file path or file-like object
        key (Optional[str], optional): if key is provided read tensor with that key from file, otherwise return header only
        offset (int, optional): offset to start reading from. Defaults to 0.
        length (Optional[int], optional): sequence length to read from the file. Defaults to None.
        length_index (int, optional): index of the length dimension in the tensor shape. Defaults to None.
    Source: Gyanendra
    Modified: Pieter
    """
    if isinstance(file_path, (str, pathlib.PurePath)):
        file = open(file_path, 'rb')
    else:
        file = file_path
        file.seek(0) # ensures we read from the beginning of the file

    # Read the size of the header
    header_size_bytes = file.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]

    # Read and decode the JSON header
    header_json = file.read(header_size).decode('utf-8').strip()
    header = json.loads(header_json)

    # if no key is specified, just return the header
    if key is None:
        return header

    # Get tensor information from the header
    tensor_info = header[key]
    offsets = tensor_info['data_offsets']
    tensor_shape = tensor_info['shape']
    dtype, byte = DTYPE_BYTE_MAPPING[tensor_info['dtype']]

    # for anything scalar we won't even look at length_index, offset or duration
    if len(tensor_shape) == 0 and len(offsets) and offsets[0] < offsets[1]:
        # ignore start and end indexes when we're reading scalars
        slice_length = offsets[1] - offsets[0]
        assert slice_length % byte == 0, f"slice length {slice_length} must be a multiple of the byte size {byte}"
        slice_length = slice_length // byte
        start_pos = 8 + header_size + offsets[0]
        file.seek(start_pos)
        # Calculate the number of bytes to read for this iteration
        end_pos = slice_length * byte
        # Read the data from the file
        final_data = torch.frombuffer(file.read(end_pos), dtype=dtype).clone().detach()
        return final_data

    if len(tensor_shape) == 1:
        # special case where we can infer value of length_index
        length_index = 0

    assert length_index in [-2, -1, 0], 'length_index of -2, -1 or 0 must be provided for non-scalar tensors'

    if len(tensor_shape) == 2:
        if length_index == 0:
            length_index = -2 # interface accepts both 0 and -2 in this, but -2 is canonical
    elif len(tensor_shape) > 2:
        # assert that all dimensions prior to length_index are all 1 if they exist
        for i in range(len(tensor_shape) + length_index):
            assert tensor_shape[i] == 1, f"length_index {length_index} must be 1 for all dimensions prior to length_index"

    # handle if length is not provided
    if length is None:
        length = tensor_shape[length_index]

    # handle if offset is negative
    if offset < 0:
        offset = tensor_shape[length_index] + offset

    # handle if offset is greater than length
    offset = min(offset, length)

    start_index = offset
    end_index = offset + length

    #handle if end_index is greater than max length
    if end_index > tensor_shape[length_index]:
        end_index = tensor_shape[length_index]

    # Adjust the final shape to include the sliced dimension
    final_data_shape = tensor_shape.copy()
    final_data_shape[length_index] = (end_index - start_index)
    final_data = torch.empty(reduce(mul, final_data_shape), dtype=dtype)

    # Calculate the absolute starting position of the data within the file
    absolute_start_pos = 8 + header_size + offsets[0]
    slice_length = end_index - start_index

    if len(tensor_shape) == 1:
        # Calculate the byte position where this chunk's data starts in the file
        start_pos = absolute_start_pos + start_index * byte
        file.seek(start_pos)
        # Calculate the number of bytes to read for this chunk
        end_pos = slice_length * byte
        # Read the data from the file
        final_data = torch.frombuffer(file.read(end_pos), dtype=dtype)
        return final_data.reshape(final_data_shape)

    if length_index == -2:
        start_index_iter = start_index * reduce(mul, tensor_shape[1:])
        # Calculate the byte position where this iteration's data starts in the file
        start_pos = absolute_start_pos + start_index_iter * byte
        file.seek(start_pos)
        # Calculate the number of bytes to read for this iteration
        end_pos = slice_length * byte * reduce(mul, tensor_shape[1:])
        # Read the data from the file
        final_data = torch.frombuffer(file.read(end_pos), dtype=dtype) #.clone().detach()
        return final_data.reshape(final_data_shape)

    if length_index == -1:
        # Calculate the number of iterations needed to cover the entire data
        iterations = reduce(mul, tensor_shape[:-1])

        for i in range(iterations):
            # Calculate the start index for this iteration
            start_index_iter = start_index + tensor_shape[-1] * i
            # Calculate the byte position where this iteration's data starts in the file
            start_pos = absolute_start_pos + start_index_iter * byte
            file.seek(start_pos)
            # Calculate the number of bytes to read for this iteration
            end_pos = slice_length * byte
            # Read the data from the file
            data = torch.frombuffer(file.read(end_pos), dtype=dtype) #.clone().detach()

            # Assign the read data to the appropriate position in the final tensor
            final_data[i * slice_length: (i + 1) * slice_length] = data

    # Reshape the final tensor to its original shape
    return final_data.reshape(final_data_shape)
