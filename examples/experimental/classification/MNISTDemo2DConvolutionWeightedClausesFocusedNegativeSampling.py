import numpy as np
from time import time

from keras.datasets import mnist

from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train >= 75, 1, 0)
X_test = np.where(X_test >= 75, 1, 0)

tm = TMCoalescedClassifier(20000, int(50*100), 5.0, platform='CUDA', patch_dim=(10, 10), focused_negative_sampling=True, weighted_clauses=True)

print("\nAccuracy over 250 epochs:\n")
for i in range(250):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Literals: %d Training: %.2fs Testing: %.2fs" % (i+1, result, tm.literal_clause_frequency().sum(), stop_training-start_training, stop_testing-start_testing))
