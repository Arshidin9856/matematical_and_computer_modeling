def triangle_type(a, b, c):
    if a == b == c:
        return "Equilateral"
    elif a == b or b == c or c == a:
        return "Isosceles"
    else:
        return "Versatile"

# Input
a = int(input())
b = int(input())
c = int(input())

# Check if the entered values form a valid triangle
if a + b > c and a + c > b and b + c > a:
    # Determine and print the type of triangle
    result = triangle_type(a, b, c)
    print(result)
else:
    print("The entered side lengths do not form a valid triangle.")

# 1
# 1
# 1
# Equilateral
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\5.py
# 2
# 2
# 5
# The entered side lengths do not form a valid triangle.
# PS C:\Users\Дом\Desktop\2023\lab6_ICT\task2> py .\5.py
# 2
# 2
# 1
# Isosceles