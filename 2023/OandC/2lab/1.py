import matplotlib.pyplot as plt
import json
coordinates=[]
with open('qa194.txt', 'r') as file:
        for line in file:
            z,x, y = map(float, line.split())
            coordinates.append((x, y))

with open('Optimal_bestT.txt', 'r') as file:
        for line in file:

            
            # # print()
            # L=line[1:862]
            # path_order=L.split(' ,')
            path_order=json.loads(line)

                       # for i in line.split(' ,')[0]:
            # print(len(line))

# Extract x and y coordinates
x_values, y_values = zip(*coordinates)
print(path_order)
# Plot the dots
plt.scatter(x_values, y_values, color='red', marker='o')
for i in range(len(path_order) - 1):
    start_index = int(path_order[i])
    end_index = int(path_order[i + 1])
    
    plt.plot([x_values[start_index], x_values[end_index]],
             [y_values[start_index], y_values[end_index]], color='blue')


# Add labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Dots by Coordinates')

# Show the plot
plt.show()


