import tensorflow as tf

for summary in tf.train.summary_iterator("./logs/model-1580994881.941672/events.out.tfevents.1580994895.DESKTOP-9241L35"):
    print(summary)