import json,sys
# data captured from the engine (46 undecided + 44 proven-legal k=4 controls)
raw = sys.stdin.read()
json.dump(json.loads(raw), open('data.json','w'))
print("wrote data.json")
