import os

imgDir = './dataset/data/COCO/coco/images/'
targetDir = './dataset/data/VQA/done/'

# create map from image id to file path.
def createMap(split):
    imgNames = os.listdir(imgDir + split)
    id2Path = {int(n.rsplit('.', 1)[0].rsplit('_', 1)[-1]): split + '/' + n for n in imgNames}
    return id2Path
id2Path = {}
id2Path.update(createMap('train2014'))
id2Path.update(createMap('val2014'))

with open(targetDir + 'img_ids.txt') as f:
    imgIds = [int(l) for l in f.readlines()]
imgPaths = [id2Path[imgId] for imgId in imgIds]
with open(targetDir + 'img_paths.txt', 'w') as f:
    f.write('\n'.join(imgPaths)+'\n')
