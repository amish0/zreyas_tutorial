
def sum_two_numbers(a, b):
    return a + b

# x = sum_two_numbers(3, 4)
# print(x)
# print(sum_two_numbers(3, 7))
import argparse
parser = argparse.ArgumentParser(
        prog='add two numbers',
        description='test argparse parameters',
        epilog='Text at the bottom of help')
parser.add_argument('--x', type=int, default=2, help='First number', required=False)
parser.add_argument('--y', type=int, default=3, help='Second number', required=False)
parser.add_argument('--bool_test', action='store_false', help='Third number', required=False)

args = parser.parse_args()
x = args.x
y = args.y
bool_test = args.bool_test
print(f"x: {x}, y: {y}")
print(f"bool_test: {bool_test}")
if bool_test:
    print("bool_test is true")
    print(sum_two_numbers(x, y))
