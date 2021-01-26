import tensorflow as tf

class DigitRecognizer(object):

    def predict(self):
        pass

    def train(self):
        pass


def get_train_data(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value = '0',
        num_epochs = 1,
        )
    return dataset

csv_path = 'data/train.csv'
dataset = tf.data.TextLineDataset(csv_path).skip(1)
dataset = dataset.map(lambda _: tf.string_split([_], ','))
print (dataset)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    vals = sess.run(el)
    print (vals)



# with tf.Session() as sess:
#     raw_train_data = get_train_data('data/train.csv')
#     res = tf.data.experimental.get_single_element(raw_train_data.take(1))
#     print (res)
#     #data, label = next(iter(raw_train_data))

