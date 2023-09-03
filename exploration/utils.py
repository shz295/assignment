import os
import pandas as pd

def import_train_data():
    numbers_path = 'data/숫자인식/'
    alphabets_path = 'data/알파벳인식/'
    free_patterns_path = 'data/자유패턴/'

    number_classes = [str(i) for i in range(10)]
    alphabet_classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    number_folders = {}
    for class_name in number_classes:
        number_folders[class_name] = '0' + class_name

    data = []

    for number, folder_name in number_folders.items():
        n_folder_path = os.path.join(numbers_path, folder_name, 'n'+folder_name)
        r_folder_path = os.path.join(numbers_path, folder_name, 'r_n'+folder_name)
        for file in os.listdir(n_folder_path):
            data.append({
                'image_path': os.path.join(n_folder_path, file),
                'category': 'number',
                'target': number,
                'r': False
            })
        for file in os.listdir(r_folder_path):
            data.append({
                'image_path': os.path.join(r_folder_path, file),
                'category': 'number',
                'target': number,
                'r': True
            })

    for letter in alphabet_classes:
        n_folder_path = os.path.join(alphabets_path, letter, letter)
        r_folder_path = os.path.join(alphabets_path, letter, 'r_'+letter)
        for file in os.listdir(n_folder_path):
            data.append({
                'image_path': os.path.join(n_folder_path, file),
                'category': 'alphabet',
                'target': letter,
                'r': False
            })
        for file in os.listdir(r_folder_path):
            data.append({
                'image_path': os.path.join(r_folder_path, file),
                'category': 'alphabet',
                'target': letter,
                'r': True
            })

    for root, dirs, files in os.walk(free_patterns_path):
        for file in files:
            data.append({
                'image_path': os.path.join(root, file),
                'category': 'free_pattern',
                'target': 'free',
                'r': None
            })

    df = pd.DataFrame(data)

    test_data = []
    number_folders = {str(i): 'test_'+str(i) for i in range(10)}

    for number, folder_name in number_folders.items():
        for file in os.listdir(os.path.join(numbers_path, '숫자 추가', folder_name)):
            test_data.append({
                'image_path': os.path.join(numbers_path, '숫자 추가', folder_name, file),
                'category': 'number',
                'target': number,
                'r': True if '_r_' in file else False
            })

    alphabet_folders = {chr(i): 'test_'+chr(i) for i in range(ord('A'), ord('Z')+1)}

    for letter, folder_name in alphabet_folders.items():
        for file in os.listdir(os.path.join(alphabets_path, '알파벳 추가', folder_name)):
            test_data.append({
                'image_path': os.path.join(alphabets_path, '알파벳 추가', folder_name, file),
                'category': 'alphabet',
                'target': letter,
                'r': True if '_r_' in file else False
            })

    free_test_folders = ['a_g', '10', '40']
    for folder in free_test_folders:
        for root, dirs, files in os.walk(os.path.join(free_patterns_path, folder)):
            for file in files:
                test_data.append({
                    'image_path': os.path.join(root, file),
                    'category': 'free_pattern',
                    'target': 'free',
                    'r': None
                })
        df = df.drop(df[df['image_path'].str.contains('/'+folder+'/')].index)

    test_df = pd.DataFrame(test_data)

    return df, test_df

