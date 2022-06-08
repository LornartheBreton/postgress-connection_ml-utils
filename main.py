from utils import *
from keras.applications.resnet import ResNet50


def main():
    with open('credentials.json') as credentials_file:
        CONNX_PARAMS = json.load(credentials_file)

        TABLE_NAME = 'sabueso_img'
        IMAGE_DOWNLOAD_PATH = 'input/download/'
        CLASS_PATH = 'input/restnet50/imagenet_class_index.json'
        WEIGHTS_PATH = 'input/restnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

        model = ResNet50()
        model.load_weights(WEIGHTS_PATH, by_name=True)

        downloader = PostgressImageDownloader(CONNX_PARAMS, TABLE_NAME)

        if downloader.connect_to_database():
            downloaded = downloader.download_images(IMAGE_DOWNLOAD_PATH)

            if downloaded:
                img_paths = downloader.img_paths

                downloader.close_connection()
                del downloader

                data = read_and_prep_images(img_paths)
                predictions = model.predict(data)

                labels = decode_predictions(predictions, class_list_path=CLASS_PATH)

                print(labels)


if __name__ == "__main__":
    main()
