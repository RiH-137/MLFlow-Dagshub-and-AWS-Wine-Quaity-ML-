import sys

## this code is use to provide different values of alpha and l1_ratio from command line
## for example: python exp.py 0.1 0.2 else default values will be used (0.5, 0.5)
## cli--> python exp.py 0.1 0.2 or python app.py 0.1 0.2
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

print(alpha, l1_ratio)