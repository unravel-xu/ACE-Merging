import json

ALL_DATASETS = ['MNIST', 'Cars', 'DTD', 'EuroSAT', 'GTSRB', 'RESISC45', 'SUN397', 'SVHN', 'PCAM', 'CIFAR100', 'STL10', 'OxfordIIITPet', 'Flowers102', 'FER2013', 'CIFAR10', 'Food101', 'RenderedSST2', 'EMNIST', 'FashionMNIST', 'KMNIST']

DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def compute_average(dataset_list, filepath):
    # Read the data from the JSON file
    with open(filepath, "r") as file:
        data = json.load(file)

    # Filter the values corresponding to the datasets in the list
    selected_test_score = [data[dataset] for dataset in dataset_list if dataset in data]
    selected_val_score = [data[f'{dataset}Val'] for dataset in dataset_list if dataset in data]
    # Compute the average if there are values
    
    avg_test_score = sum(selected_test_score) / len(selected_test_score)
    avg_val_score = sum(selected_val_score) / len(selected_val_score)
    
    print(f'test: {avg_test_score * 100:.2f}')
    print(f'val: {avg_val_score * 100:.2f}')

filepath = '../vit/results/ViT-L-14/ace_merging_14_tasks_results.json'

compute_average(DATASETS_20, filepath)