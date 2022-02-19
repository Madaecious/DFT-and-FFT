#############################################################################################################
# Implementations of the Discrete Fourier Transform and Fast Fourier Transform
# Mark Barros - BID 013884117
# CS3310 - Design and Analysis of Algorithms
# Cal Poly Pomona: Spring 2021
# Project Description:
#       Computes the DFT using native Python lists
#       Computes the DFT using NumPy arrays
#       Computes the FFT using recursion and native Python lists
#       Computes the FFT using recursion and native NumPy arrays
#       Outputs respective run times for comparison purposes
#############################################################################################################

# These are needed imports. ---------------------------------------------------------------------------------
from numpy import *                            # Provides functionality for arrays
from time import perf_counter                  # Provides functionality for timing

# These are constants. --------------------------------------------------------------------------------------
ITERATIONS = 10                                # Specifies number of executions to perform


# This permits for measuring the time a method takes to execute. --------------------------------------------
# f is the function to be called, size is the size of the vector, repeat is the number of executions.
# It returns the average execution time for that function with a vector of the given size.
def time_it(f, size, repeat=ITERATIONS):
    x = arange(size, dtype=complex)            # Generate a vector
    t0 = perf_counter()                        # Start a timer
    for j in range(0, repeat):                 # Repeated calls
        f(x)
    return (perf_counter() - t0) / repeat      # Compute average


# This implementation uses native Python lists to compute the DFT. ------------------------------------------
# x is the vector for which the DFT will be computed.
# It returns a complex-number vector of the same size, with the coefficients of the DFT.
def direct_ft(x):
    N = len(x)                                 # Length of the vector
    X = [ 0+0j ] * N                           # Accumulate the results
    W = exp(-2j*pi/N)                          # Coefficients
    Wk = 1.
    for k in range(0, N):                      # Compute the kth coefficient
        Wkn = 1.
        for n in range(0, N):                  # Operate the summation
            X[k] = X[k] + x[n]*Wkn             # Compute every term
            Wkn = Wkn * Wk                     # Update coefficients 
        Wk = Wk * W
    return X


# This implementation uses NumPy arrays to compute the DFT. -------------------------------------------------
# x is the vector for which the DFT will be computed.
# It returns a complex-number vector of the same size, with the coefficients of the DFT.
def array_direct_ft(x):
    N = len(x)                                 # Length of the vector
    X = zeros(x.shape, dtype=complex)          # Accumulate the results
    W = exp(-2j*pi/N)                          # Coefficients
    Wk = 1.
    for k in range(0, N):                      # Compute the kth coefficient
        Wkn = 1.
        for n in range(0, N):                  # Operate the summation
            X[k] = X[k] + x[n]*Wkn             # Compute every term
            Wkn = Wkn * Wk                     # Update coefficients
        Wk = Wk * W
    return X


# This is used to compute the smallest prime factor of a given number. --------------------------------------
# n is the number for which the prime is sought.
# It returns the smallest prime factor (or the number itself it is already prime).
def __factor(n):
    rn = int(ceil(sqrt(n)))                    # Search up to the square root of the number
    for i in range(2, rn+1):
        if n%i == 0:                           # When remainder is zero, factor is found
            return i
    return n


# This computes the FFT using recursion and native Python lists. --------------------------------------------
# x is the vector of which the FFT will be computed.
# It returns a complex-number vector of the same size, with the coefficents of the DFT.
def recursive_fft(x):
    N = len(x)                                 # Length of the vector
    N1 = __factor(N)                           # Find the smallest factor of the vector length
    if N1 == N:                                # If the length is prime itself,
        return direct_ft(x)                    # the transform is given by the direct form
    else:
        N2 = N // N1                           # Decompose in two factors, N1 being prime
        X = [ 0+0j ] * N                       # Accumulate the results
        W = exp(-2j*pi/N)                      # Coefficients
        Wj = 1.
        for j in range(N1):                    # Compute every subsequence of size N2
            Xj = recursive_fft(x[j::N1])
            Wkj = 1.
            for k in range(N):
                X[k] = X[k] + Xj[k%N2] * Wkj   # Recombine results
                Wkj = Wkj * Wj                 # Update coefficients
            Wj = Wj * W
        return X


# This computes the FFT using recursion and native NumPy arrays. --------------------------------------------
# x is the vector for which the FFT will be computed.
# It returns a complex-number vector of the same size, with the coefficients of the DFT.
def array_recursive_fft(x):
    N = len(x)                                 # Length of the vector
    N1 = __factor(N)                           # Find the smallest factor of the vector length
    if N1 == N:                                # If the length is prime itself,
        return array_direct_ft(x)              # the transform is given by the direct form
    else:
        N2 = N // N1                           # Decompose in two factors, N1 being prime
        X = zeros((N, ), dtype=complex)        # Accumulate the results
        W = exp(-2j*pi/N)                      # Coefficients
        Wj = 1.
        for j in range(N1):                    # Compute every subsequence of size N2
            Xj = array_recursive_fft(x[j::N1])
            Wkj = 1.
            for k in range(N):
                X[k] = X[k] + Xj[k%N2] * Wkj   # Recombine results
                Wkj = Wkj * Wj                 # Update coefficients
            Wj = Wj * W
        return X


# This is where program exection begins. --------------------------------------------------------------------
if __name__ == "__main__":

    # This section performs the DFT and the FFT using lists on the same sequence of numbers.

    # This is the sequence that will be input.
    sequence = [15, 21, 13, 44]                 # [3*5, 3*7, 13, 2*2*11]

    print()
    print()
    print("############################################ The DFT and FFT of a Sequence "
          "of Numbers ##############################################")
    print()
    print("\t\t DFT-L:  Computes the DFT using native Python lists")
    print("\t\t FFT-L:  Computes the FFT using recursion and native Python lists")
    print()
    print()
    print("\t\t This is a sequence: ", sequence)
    print()
    print()
    print("\t\t These are the results of the two implementations:")
    print()
    dft_l = direct_ft(sequence)
    print("\t\t DFT-L: ", dft_l)
    fft_l = recursive_fft(sequence)
    print("\t\t FFT-L: ", fft_l)
    print()
    print()


    # This section runs the four different implementations for purposes of comparing the
    # times taken to execute.

    # These are the sizes of the vectors that each implementation will run.
    # [ 2*3, 2*2*3, 2*3*3, 2*3*5, 2*2*3*3, 2*2*5*5, 2*3*5*7, 2*2*3*3*5*5 ]
    SIZES = [6, 12, 18, 30, 36, 100, 210, 900]

    # This is the output header.
    print("######################################### Computation Times for the "
          "Various Implementations ########################################")

    # This is the legend for the different implementations.
    print()
    print("\t\t DFT-L:  Computes the DFT using native Python lists")
    print("\t\t DFT-A:  Computes the DFT using NumPy arrays")
    print("\t\t FFT-L:  Computes the FFT using recursion and native Python lists")
    print("\t\t FFT-A:  Computes the FFT using recursion and native NumPy arrays")
    print()

    # This is the header of the table that will contain the time comparisons.
    print("\t\t", "-----------"*6 + "-")
    print("\t\t", "|    N     |   N^2    |  DFT-L   |  DFT-A   |  FFT-L   |  FFT-A   |")
    print("\t\t", "-----------"*6 + "-")

    # This calculates the time it takes to execute each implementation.
    for n in SIZES:

        # This computes the average execution times for each of the implementations.
        dtime  = time_it(direct_ft, n, ITERATIONS)
        adtime = time_it(array_direct_ft, n, ITERATIONS)
        rtime  = time_it(recursive_fft, n, ITERATIONS)
        artime = time_it(array_recursive_fft, n, ITERATIONS)

        # This formats and prints the results.
        print("\t\t", f'| {n:8} | {n**2:8} 'f'| {dtime:8.6f} | '
                      f'{adtime:8.6f} | {rtime:8.6f} | {artime:8.6f} | ')

    # This is the end of the table that will contain the time comparisons.
    print("\t\t", "-----------"*6 + "-")
    print()
    print("#####################################################################"
          "##############################################################")
    print()
    # This is the end of the Discrete Fourier Tranform and Fast Fourier Transform----------------------------