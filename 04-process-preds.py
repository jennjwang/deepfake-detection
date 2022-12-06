import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
import json

# https://spark.apache.org/docs/latest/sql-data-sources-json.html
spark = SparkSession.builder.appName("athing").getOrCreate()
# aDF = spark.read.json("./cnn_preds.json")
# # aDF.show()

def mapper0(record):
    isReal = 0
    prob = record[1]
    pathParts = record[0].split('/')
    if pathParts[0] == "real":
        isReal = 1
    imgParts = pathParts[1].split('-')
    video = imgParts[0]
    return ((isReal, video), [(prob, 1)])

def reducer0(a, b):
    return a + b

def mapperSeparatePred(record):
    isCorrect = 0
    isReal, video = record[0]
    listTups = record[1]
    rearranged = zip(*listTups)
    sumTup = [sum(zipped) for zipped in rearranged]
    actualPred = sumTup[0]/sumTup[1]
    # if actualPred <= 0.5 and isReal == 0:
    #     isCorrect = 1
    # if actualPred > 0.5 and isReal == 1:
    #     isCorrect = 1
    # return (record[0], actualPred)
    return (isReal, [(actualPred, 1)])
    # the latter part of the tuple will be summed up to track how many total videos there are here

def mapperSeparate(record):
    isCorrect = 0
    isReal, video = record[0]
    listTups = record[1]
    rearranged = zip(*listTups)
    sumTup = [sum(zipped) for zipped in rearranged]
    actualPred = sumTup[0]/sumTup[1]
    if actualPred <= 0.5 and isReal == 0:
        isCorrect = 1
    if actualPred > 0.5 and isReal == 1:
        isCorrect = 1
    # return (record[0], actualPred)
    return (isReal, [(isCorrect, 1)])
    # the latter part of the tuple will be summed up to track how many total videos there are here

def mapperTgth(record):
    isCorrect = 0
    isReal, video = record[0]
    listTups = record[1]
    rearranged = zip(*listTups)
    sumTup = [sum(zipped) for zipped in rearranged]
    actualPred = sumTup[0]/sumTup[1]
    if actualPred <= 0.5 and isReal == 0:
        isCorrect = 1
    if actualPred > 0.5 and isReal == 1:
        isCorrect = 1
    # return (record[0], actualPred)
    return (42, [(isCorrect, 1)])
    # the latter part of the tuple will be summed up to track how many total videos there are here

def reducer1(a, b):
    return a + b

def mapper2(record):
    isRealOr42 = record[0]
    listTups = record[1]
    rearranged = zip(*listTups)
    sumTup = [sum(zipped) for zipped in rearranged]
    totalCorrect = sumTup[0]
    total = sumTup[1]
    accuracy = totalCorrect/total
    return ((isRealOr42, totalCorrect), (total, accuracy))

def mapper2Pred(record):
    isReal = record[0]
    listTups = record[1]
    rearranged = zip(*listTups)
    sumTup = [sum(zipped) for zipped in rearranged]
    sumPred = sumTup[0]
    total = sumTup[1]
    overallPred = sumPred/total
    return (isReal, (total, overallPred))

res = []
with open("./cnn_preds_values.json", "r") as read_content:
    res = json.load(read_content)
read_content.close()

f = open("./mapreduce_res.txt", "w")
f.write("1. first mapper: extracted real/fake, video name.  Stored as tuple in key.\n")
f.write("the prediction and 1 are put in a tuple, then in a list as the value\n")
f.write("first reducer: combined all the frames from the same video, and append tuple lists tgth\n")
res_part1 = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0)
for ele in res_part1.collect():
    f.write(str(ele) + "\n")

f.write("\n")
f.write("2a. first variation: overall accuracy over both fake and real preds\n")
f.write("second mapper: sums corresponding eles in the tuples to form a summed pred and total photos for this vid\n")
f.write("divides the two to form actualPred, then compares with isReal to from isCorrect 0 or 1\n")
f.write("second mapper: return (42, [(isCorrect, 1)])\n")
f.write("second reducer: combines the lists in the values again\n")
res_part2a = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperTgth).reduceByKey(reducer1)
for ele in res_part2a.collect():
    f.write(str(ele) + "\n")

f.write("\n")
f.write("2b. second variation: accuracy for fake and real preds separately\n")
res_part2b = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperSeparate).reduceByKey(reducer1)
for ele in res_part2b.collect():
    f.write(str(ele) + "\n")

f.write("\n")
f.write("2c. third variation: returning averaged prediction for real/fake vids\n")
f.write("second mapper: sums corresponding eles in the tuples to form a summed pred and total photos for this vid\n")
f.write("divides the two to get actualPred for this video.  returns return (isReal, [(actualPred, 1)])\n")
f.write("same reducer\n")
res_part2c = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperSeparatePred).reduceByKey(reducer1)
for ele in res_part2c.collect():
    f.write(str(ele) + "\n")
f.write("\n")

resSeparate = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperSeparate).reduceByKey(reducer1).map(mapper2)
resTgth = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperTgth).reduceByKey(reducer1).map(mapper2)
resPred = spark.sparkContext.parallelize(res, 64).map(mapper0).reduceByKey(reducer0).map(mapperSeparatePred).reduceByKey(reducer1).map(mapper2Pred)


f.write("3a. Accuracy, overall: (42, totalCorrect), (total, accuracy)\n")
for ele in resTgth.collect():
    f.write(str(ele) + "\n")
f.write("\n")

f.write("3b. Accuracy, separated into real and fake: (isReal, totalCorrect), (total, accuracy)\n")
for ele in resSeparate.collect():
    f.write(str(ele) + "\n")
f.write("\n")

f.write("3c. OverallPred, separated into real and fake: (isReal, (total of real or fake vids, overallPred))\n")
for ele in resPred.collect():
    f.write(str(ele) + "\n")

f.close()




# with open('./reduce_result.json', 'w') as outfile:
#     json.dump(mapreduce, outfile, indent=4)
# print(mapreduce)