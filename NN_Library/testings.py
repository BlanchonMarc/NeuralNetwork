from DatabaseLoader.DatabaseLoader import DatabaseLoader,DatabaseTorch

root_dataset = '../Datasets/CamVid/'
inputs = ['train/', 'val/', 'test/']
checkings = ['trainannot/', 'valannot/', 'testannot/']
Db = DatabaseTorch(root=root_dataset, train_folders=inputs, test_folders=checkings)

data_dict = Db(batch_size = 5, shuffle = True, num_workers = 4)

print(data_dict['train'])
