import json
from models import ewc
from trainer import train

ewc.init_milestones=[60,120,170]
ewc.epochs=80
ewc.milestones=[40,70]

with open("exps/ewc_50_50.json") as file:
    config=json.load(file)

if __name__=="__main__":
    train(config)
