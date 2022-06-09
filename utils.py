import numpy as np
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import psycopg2
import requests
import json
from colorama import Fore


def read_and_prep_images(img_dict, img_height=224, img_width=224):
    """
    Read and preprocess the images to load into the model.

    :param img_dict: Dictionary - list of strings with paths to the images
    :param img_height: Int - the desired height of the image
    :param img_width: Int - the desired width of the image
    :return: The encoded and preprocessed images.
    """
    img_paths = [i[0] for i in img_dict.values()]
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

    def __init__(self, connection_params, table_name):
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

        self.upload_dict = {}

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
            print(Fore.RED + "Failed to initialize database connection.")
            print(Fore.RED + "Error message:")
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
                file_name = row[1].replace('http://pbs.twimg.com/media/', '')
                file_name = file_name.replace(':', '')
                file_name = file_name.replace('/', '-')
                full_path = save_dir + file_name
                self.upload_dict[row[2]] = [full_path]
                print(f'Downloading file at {row[1]}\n'
                      f'    Saving at:  {full_path}')
                open(full_path, 'wb').write(response.content)

            return True

        except Exception as e:
            print(Fore.RED + f"Failed to download images at {save_dir}")
            print(Fore.RED + "Error message:")
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
            print(Fore.RED + "Failed to close the connection.")
            print(Fore.RED + "Error message:")
            print(e)

            return False

    def generate_upload_dict(self):
        """
        Firs prep of the upload_dict.
        
        :return: self.upload_dict
        """
        self.cursor.execute("""
                            SELECT
                                *
                            FROM sabueso_clean
                            """)
        sabueso_clean = self.cursor.fetchall()
        self.cursor.execute("""
                            SELECT
                                *
                            FROM sabueso_img
                            """)
        sabueso_img = self.cursor.fetchall()
        self.cursor.execute("""
                            SELECT
                                *
                            FROM sabueso_tweet
                            """)
        sabueso_tweet = self.cursor.fetchall()

        for row in sabueso_tweet:
            self.upload_dict[row[3]].append(row[0])

        return self.upload_dict

    def update_database(self, table_name, column_name, commit_transaction=False):
        """
        Update the database with the information in self.upload_dict.
        
        :rtype: object
        :param table_name: String - the name of the table to update.
        :param column_name: String - the name of the column to update.
        :param commit_transaction: Bool - Whether to commit the transaction.
        :return: True if the database was successfully updated, False otherwise.
        """

        try:
            for value in self.upload_dict.values():
                string_to_upload = ""

                for label in value[2]:
                    string_to_upload += label + ','

                command = f"""
                    UPDATE {table_name}
                        SET {column_name} = '{string_to_upload}'
                    WHERE parent_tweet_id = {value[1]}
                """

                if commit_transaction:
                    command += "COMMIT;"

                self.cursor.execute(command)

            print("Update Successful!")

            return True

        except Exception as e:
            print(Fore.RED + "Failed to update database.")
            print(Fore.RED + "Error message:")
            print(e)

            return False
