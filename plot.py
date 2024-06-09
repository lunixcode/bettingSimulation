# Correcting the oversight and importing matplotlib for plotting

import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Re-defining the function to generate the Fibonacci sequence starting with 10
def fibonacci(n):
    fib_sequence = [10, 10]  # Starting the sequence with two 10s
    for _ in range(2, n):
        next_value = fib_sequence[-1] + fib_sequence[-2]
        fib_sequence.append(next_value)
    return fib_sequence

# Generating the Fibonacci sequence
n = 10  # Number of terms
fib_sequence = fibonacci(n)

# Generating an exponential sequence for comparison, starting with 10
# Using a base that provides a visually comparable growth to the Fibonacci sequence
base = 1.6  # Adjust the base as needed for visual comparison
exp_sequence = [10 * (base ** i) for i in range(n)]

# Plotting the sequences
plt.figure(figsize=(100, 6))
plt.plot(fib_sequence, label='Fibonacci Sequence', marker='o')
plt.plot(exp_sequence, label='Exponential Growth', linestyle='--')
plt.title('Fibonacci Sequence vs. Exponential Growth')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
