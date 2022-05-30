import pandas as pd
import json

tags = json.load(open('python/tags.json'))
print('len start:', len(tags))

to_remove = set()
for tag in tags:
    if tag == 'total_paths':
        continue
    if tags[tag][1] < 100:
        to_remove.add(tag)
    elif 'name' in tag:
        to_remove.add(tag)
    elif 'source' in tag:
        to_remove.add(tag)
    elif 'fixme' in tag.lower():
        to_remove.add(tag)
    elif 'cycleway' in tag.lower():
        to_remove.add(tag)
    elif 'note' in tag.lower():
        to_remove.add(tag)
    elif 'wiki' in tag.lower():
        to_remove.add(tag)
    elif 'proposed' in tag.lower():
        to_remove.add(tag)
    elif 'prorail' in tag.lower():
        to_remove.add(tag)
    
for tag in to_remove:
    del tags[tag]

print('len end:', len(tags))

with open('python/tags.json', 'w') as f:
    json.dump(tags, f, indent=4)
# print(json.dumps(tags, indent=4))