import json

temp = {
    "dockerhub-id": "jgeng99/test_env:latest",
    "build-script": "./run.py -te -e 10"
}

with open("submission.json", "w") as outfile:
    json.dump(temp, outfile)
