import sys

vocab = set([])
with open(sys.argv[1]) as file:
    for line in file:
        words = line.strip().split()
        for word in words:
            vocab.add(word)
print(f'Vocab size: {len(vocab)}')
