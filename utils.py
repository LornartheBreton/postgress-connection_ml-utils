import numpy as np
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import psycopg2
import requests
import json


def read_and_prep_images(img_paths, img_height=224, img_width=224):
    """
    Read and preprocess the images to load into the model.

    :param img_paths: List - list of strings with paths to the images
    :param img_height: Int - the desired height of the image
    :param img_width: Int - the desired width of the image
    :return: The encoded and preprocessed images.
    """
    images = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in images])
    output = preprocess_input(img_array)

    return output


def decode_predictions(predictions, class_list_path, top=3):
    """
    Decode the predictions made by the model according to a json class list.

    :param predictions: Numpy array - array containing the predictions
    :param class_list_path: Path to the json file containing the class list.
    :param top: Int - the top N predictions to return
    :return: A list with the predictions.
    """
    with open(class_list_path) as json_file:
        return_list = []
        data = json.load(json_file)
        indices = np.argpartition(predictions, -top, axis=1)[:, -top:]

        for ids in indices:
            values_to_annex = []
            for id in ids:
                values_to_annex.append(data[str(id)][1])
            return_list.append(values_to_annex)

        return return_list


class PostgressImageDownloader:

    def __init__(self,connection_params,table_name):
        '''
        Constructs PostgressImageDownloader class.

        :param connection_params: Dictionary - a dict containing the connection parameters
        :param table_name: String - name of the table housing the image urls
        '''
        self.connection_params = connection_params
        self.table_name = table_name

        self.conn = None

        self.cursor = None

        self.img_paths = []

    def connect_to_database(self):
        '''
        Establish a connection to the database.

        :return: True if the connection was established succesfully. False if it wasn't.
        '''
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor()

            return True
        except Exception as e:
            print("Failed to initialize database connection.")
            print("Error message:")
            print(e)

            return False

    def download_images(self, save_dir):
        """
        Download the images to the specified directory, and append their paths to self.img_paths.

        :param save_dir: Path to the directory to save the images into.
        :return: True if the images were downloaded succesfully, False otherwise.
        """
        get_img = f"""
        SELECT
            *
        FROM {self.table_name}
        """

        try:
            self.cursor.execute(get_img)
            rows = self.cursor.fetchall()

            for row in rows:
                response = requests.get(row[1])
                file_name = row[1].replace('http://pbs.twimg.com/media/','')
                file_name = file_name.replace(':','')
                file_name = file_name.replace('/','-')
                full_path = save_dir + file_name
                self.img_paths.append(full_path)
                print(f'Downloading file at {row[1]}\n'
                      f'    Saving at:  {full_path}')
                open(full_path,'wb').write(response.content)

            return True

        except Exception as e:
            print(f"Failed to download images at {save_dir}")
            print("Error message:")
            print(e)

            return False

    def close_connection(self):
        """
        Close the connection to the database.

        :return: True if it was successfully closed, False otherwise.
        """
        try:
            self.conn.close()
            return True
        except Exception as e:
            print("Failed to close the connection.")
            print("Error message:")
            print(e)

            return False
