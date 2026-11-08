import json
from models import lwf
from trainer import train

lwf.init_milestones=[60,120,170]
lwf.epochs=80
lwf.milestones=[40,70]

with open("exps/lwf_50_50.json") as file:
    config=json.load(file)

if __name__=="__main__":
    train(config)
