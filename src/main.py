import argparse

from Practica_2_3 import Practica_2_3

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    Practica_2_3('Mejorado', "C:/Users/Aussar/PycharmProjects/VA_Practica_2.3/data/train_recortadas/", "C:/Users/Aussar/PycharmProjects/VA_Practica_2.3/data/test/")

    # Practica_2_3(args.detector, args.train_path, args.test_path)