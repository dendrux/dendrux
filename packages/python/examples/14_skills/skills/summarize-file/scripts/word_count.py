"""Count the words in the text given as arguments."""

import sys

text = " ".join(sys.argv[1:])
words = len(text.split())
print(f"{words} words" + (" (over the 200 word limit)" if words > 200 else ""))
