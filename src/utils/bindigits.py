import random
from typing import List

import numpy as np


def decimal2binary(decimal: int, bit_length: int):
    """
    Converts given integer to array of binaries with prefix of 1 if negative.
    :param decimal:
    :param bit_length: Length of the resulting array
    :return:
    """
    binary = [0] * bit_length
    _sign = np.sign(decimal)
    # convert -1, 1 to 0 -> positive, 1 -> negative
    binary.insert(0, int((abs((_sign - 1)) / 2)))
    # Cut the prefix
    _value = list(bin(abs(decimal)))[2:]
    # Convert to int and add to end of result array.
    binary[bit_length - len(_value):] = [int(x) for x in _value]
    return binary


def binary2decimal(binary_value: List[int]):
    """
    Decode a binary array into decimal.
    :param binary_value:
    :return:
    """
    _sign = (-1) ** int(binary_value[0] + 1)
    result: int = 0
    for index, x in enumerate(reversed(binary_value[1:])):
        result += x * 2 ** index
    return _sign * result


def createChromosome(qtyFeatures, lenFeature):
    """
    Creates an array with a subarray of features
    :param qtyFeatures: Number of features.
    :param lenFeature: Length of the feature.
    :return:
    """
    chromosome = []
    for i in range(0, qtyFeatures):
        feature = []
        for j in range(0, lenFeature):
            feature.append(random.randint(0, 1))
        chromosome = np.concatenate((chromosome, feature))
    return chromosome


def crossOver(chromosome1, chromosome2):
    """
    Mix the chromosomes randomly.
    :param chromosome1:
    :param chromosome2:
    :return:
    """
    _th = np.random.randint(1, len(chromosome1))
    chrom1 = np.concatenate((chromosome1[:_th], chromosome2[_th:]))
    chrom2 = np.concatenate((chromosome2[:_th], chromosome1[_th:]))
    return chrom1, chrom2


if __name__ == "__main__":
    number = 3
    print(f"Decode {number} in binary: ", decimal2binary(number, 5))
    print(f"Decode {number} back to decimal: ", binary2decimal(decimal2binary(number, 5)))
    # print(str(createChromosome(4, 3)))
