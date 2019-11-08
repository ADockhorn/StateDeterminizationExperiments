__author__ = 'Alex'
import random


def RandSelect(A,k,length):
    """ Return the k-th smallest element of list A

    :param "A": list of numbers
    :param "k" : nr of searched k-th nearest neighbor
    :param "length": length of list A, needed for recursive call

    :return: returns the k smallest element
    """

    #let r be chosen uniformly at random in the range 1 to length(A)
    n = length-1
    r = random.randint(0, length-1)
    A1 = []
    A2 = []
    pivot = A[r]
    # lesser and bigger array
    for i in range ( 0 , n+1):
        if A[i] < pivot :
                A1.append(A[i])
        if A[i] > pivot :
                A2.append(A[i])
    if k <= len(A1):
        # search in list of small elements
        return RandSelect(A1, k ,len(A1))
    if k > len(A) - len(A2):
        # search in the pile of big elements
        return RandSelect(A2, k - (len(A) - len(A2)) , len(A2))
    else :
        return pivot
