from matplotlib import pyplot as plt

array = [] # Initialize an empty list to store epsilon values.
for i in range(1000): # Loop through 1000 epochs (iterations).
    epsion = max(0.01, 0.9 - 0.01 * (i / 10))
    # Calculate epsilon for each epoch:
    # - Epsilon starts at 0.9 and decreases by 0.01 every 10 epochs.
    # - The minimum value of epsilon is capped at 0.01 to ensure exploration never completely stops.
    array.append(epsion)
    # Append the calculated epsilon value to the array.
y = array # Set y-axis values as the calculated epsilon values.
x = list(range(len(array))) # Set y-axis values as the calculated epsilon values.

plt.plot(x, y)
# Plot the x and y values to create a line graph.

plt.title('epsion')
plt.xlabel('epoch')
plt.ylabel('epsion')

plt.show()
