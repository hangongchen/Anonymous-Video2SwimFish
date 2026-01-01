#!/usr/bin/env python3
import collections
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: log_tail.py <output_path> <max_lines>", file=sys.stderr)
        return 1
    output_path = sys.argv[1]
    try:
        max_lines = int(sys.argv[2])
    except ValueError:
        print("max_lines must be an integer", file=sys.stderr)
        return 1
    if max_lines <= 0:
        print("max_lines must be > 0", file=sys.stderr)
        return 1

    buffer = collections.deque(maxlen=max_lines)
    flush_every = 200
    counter = 0

    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        buffer.append(line)
        counter += 1
        if counter % flush_every == 0:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.writelines(buffer)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
