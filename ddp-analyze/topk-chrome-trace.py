import json
from collections import defaultdict
import sys

path = sys.argv[1] # path to the chrome trace json file

with open(path, "r") as f:
    j = json.load(f)

total_us_by_name = defaultdict(float)
count_by_name = defaultdict(int)

total_us_by_cat = defaultdict(float)
count_by_cat = defaultdict(int)

# Chrome trace: events in j["traceEvents"]
for e in j.get("traceEvents", []):
    if e.get("ph") != "X":
        continue
    dur = e.get("dur")
    if dur is None:
        continue
    name = e.get("name", "<noname>")
    cat = e.get("cat", "<nocat>")

    total_us_by_name[name] += dur
    count_by_name[name] += 1

    total_us_by_cat[cat] += dur
    count_by_cat[cat] += 1

def topk(d, k=30):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

print("\nTop ops by TOTAL time (ms):")
for name, us in topk(total_us_by_name, 30):
    print(f"{us/1000:10.3f} ms  |  {count_by_name[name]:8d} calls  |  {name}")

print("\nTop categories by TOTAL time (ms):")
for cat, us in topk(total_us_by_cat, 30):
    print(f"{us/1000:10.3f} ms  |  {count_by_cat[cat]:8d} calls  |  {cat}")