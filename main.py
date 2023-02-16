from dataset import maketfdataset
from model import create_model
import tensorflow as tf
# printing one sample of tf dataset
# sample = list(train_dataset.batch(10).take(1).as_numpy_iterator())[0][0]
# print(np.shape(sample))


train_dataset, test_dataset = maketfdataset()


model = create_model()
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(train_dataset, epochs=50, verbose=1, shuffle=1)
y_pred = model.predict(test_dataset)
print("accuracy-linear:", tf.metrics.accuracy_score(y_true=test_dataset.labels, y_pred=y_pred), "\n")

