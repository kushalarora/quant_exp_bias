import tarfile
import os
import glob
import tarfile

def shard_file(filename: str):
    head, tail = os.path.split(filename)
    tail_prefix, tail_suffix = tail.split('.')
    shard_folder = os.path.join(head, tail_prefix)
    os.makedirs(shard_folder, exist_ok=True)
    shards = []
    for i in range(1,5):
        shard_filename =os.path.join(shard_folder, f'{tail_prefix}_{i}.{tail_suffix}')
        shards.append(open(shard_filename, 'w'))

    with open(filename) as f:
        for i,line in enumerate(f):
            shard_idx = 3
            if  (i + 3) % 4 == 0:
                shard_idx = 0
            elif (i + 2) % 4 == 0:
                shard_idx = 1
            elif (i + 1) % 4 == 0:
                shard_idx = 2
            shards[shard_idx].write(line)

    for shard_file in shards:
        shard_file.close()
    
    with tarfile.open(shard_folder + ".tar.gz", "w:gz") as tar:
      tar.add(shard_folder, arcname='.')
      # for i in range(1,5):
      #     shard_filename =os.path.join(shard_folder, f'{tail_prefix}_{i}.{tail_suffix}')
      #     print(shard_filename)
      #     with open(shard_filename) as shard_file:
      #         tar.addfile(tarfile.TarInfo(f'{tail_prefix}_{i+1}.{tail_suffix}'), shard_file)
                
    return os.path.join(shard_folder, f'{tail_prefix}_*.{tail_suffix}')


MIN_LENGTH=10
MAX_LENGTH=50
PREFIX='data/domain-adaptation-data/'
for domain in ('acquis', 'emea', 'it', 'koran', 'subtitles'):
  filename = os.path.join(PREFIX, f'{domain}-filtered.txt')
  with open(filename, 'w') as output_file:
    for split in ('test',):
      with tarfile.open(os.path.join(PREFIX, f'{domain}-{split}.en.tgz'), "r:gz") as tar:

        for member in tar.getmembers():
          input_file = tar.extractfile(member)
          assert input_file is not None
          for line in input_file.readlines():
            line = line.decode('utf-8', errors='ignore').strip()
            words = line.split() 

            if len(words) < MIN_LENGTH or len(words) > MAX_LENGTH:
              continue

            output_file.write(line + "\n")
  shard_file(filename)