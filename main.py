from utils import *
from keras.applications.resnet import ResNet50


def main():
    with open('credentials.json') as credentials_file:
        CONN_PARAMS = json.load(credentials_file)

        TABLE_NAME = 'sabueso_img'
        IMAGE_DOWNLOAD_PATH = 'input/download/'
        CLASS_PATH = 'input/restnet50/imagenet_class_index.json'
        WEIGHTS_PATH = 'input/restnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

        model = ResNet50()
        model.load_weights(WEIGHTS_PATH, by_name=True)

        downloader = PostgressImageDownloader(CONN_PARAMS, TABLE_NAME)

        if downloader.connect_to_database():
            downloaded = downloader.download_images(IMAGE_DOWNLOAD_PATH)

            if downloaded:
                upload_dict = downloader.generate_upload_dict()

                data = read_and_prep_images(upload_dict)
                predictions = model.predict(data)

                labels = decode_predictions(predictions, class_list_path=CLASS_PATH)

                for index, value in enumerate(upload_dict.keys()):

                    upload_dict[value].append(labels[index])

                downloader.upload_dict = upload_dict

                downloader.update_database(table_name='sabueso_clean',
                                           column_name='dog_breed',
                                           commit_transaction=False)

                downloader.close_connection()
                del downloader


if __name__ == "__main__":
    main()
