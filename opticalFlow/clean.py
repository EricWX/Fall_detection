import sys

with open("dev.txt") as f:
	for line in f:
		if not line.isspace():
			sys.stdout.write(line)
