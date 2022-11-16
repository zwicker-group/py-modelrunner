import sys

if __name__ == "__main__":
    for value in sys.argv[1:]:
        sys.stdout.write(value)
    sys.stdout.flush()
