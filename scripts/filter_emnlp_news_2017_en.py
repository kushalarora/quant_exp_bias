import sys
min_len=10
max_len=50
min_word_freq=500

word2count={}
word2lineidx = {}
lines=[]
lineidxs=set([])
with open(sys.argv[1]) as infile, \
        open(sys.argv[2], 'w') as outfile:
    for i, line in enumerate(infile):

        words = line.strip().split()
        for word in words:
            if word not in word2count:
                word2count[word] = 0
            word2count[word] += 1
            
            if word not in word2lineidx:
                word2lineidx[word] = set([])
            word2lineidx[word].add(i)
        lines.append(line)
        lineidxs.add(i)
        print(f'Lines Processed: {i}', end='\r')
    print('#'*10)
    
    i = 0
    idxs_to_exclude=[]
    for word, count in word2count.items():
        if count < min_word_freq:
            idxs=word2lineidx[word]
            idxs_to_exclude += idxs 
        print(f'Words Processed: {i}', end='\r')
        i += 1
    idxs_to_exclude = set(idxs_to_exclude)

    lineidxs = lineidxs - idxs_to_exclude 
    infile.seek(0)
    for i, line in enumerate(infile):
        if i not in lineidxs:
            continue

        words = line.strip().split()
        if len(words) > min_len and len(words) < max_len:
            print(f'Adding to file: {i}', end='\r')
            outfile.write(line)
