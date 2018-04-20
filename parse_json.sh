#!bin/sh

# Note: index starts from 0, so if you wanna visualize all videos,
# say there are 10 videos, NUM should be 9
NUM=39;

for i in $(seq 0 $NUM);
do
    echo "do: parse_json.py -p path_to/rewards.json -i $i"
    python parse_json.py -p log/rewards.json -i $i
done