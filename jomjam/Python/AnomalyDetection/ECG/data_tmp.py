import numpy as np

tot_set = np.load('/works/Data/wellysis/preprocessed/5s_72_wellysis_stdscale.npy')
print(tot_set.shape)
train_set = tot_set[:2000000]
train_set = train_set[np.random.permutation(len(train_set))[:100000]]
np.save('/works/Data/wellysis/preprocessed/5s_72_wellysis_stdscale_train', train_set)
print(train_set.shape)
test_set = tot_set[-1000000:]
#test_set = test_set[np.random.permutation(len(test_set))[:30000]]
np.save('/works/Data/wellysis/preprocessed/5s_72_wellysis_stdscale_test', test_set)
print(test_set.shape)