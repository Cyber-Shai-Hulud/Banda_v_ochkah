import cv2
import os
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, render_template, request
from flask import request

app = Flask(__name__)

a = "привет"
n = 12
@app.route('/', methods=['GET', ])
def hello():
    return render_template("index.html")


@app.route('/main/', methods=['POST', ])
def predict():
    imagefile = request.files['imagefile']
    imagepath = os.path.join(
        "images", imagefile.filename)
    imagefile.save(imagepath)

    model = tf.keras.models.load_model('VGG19')
    lb = pickle.loads(open(
        'lb_7_layer_CNN.pickle', "rb").read())

    image = tf.keras.preprocessing.image.load_img(
        imagepath, target_size=(256, 256))

    image = tf.keras.preprocessing.image.img_to_array(image)

    h, w, _ = image.shape

    image /= 255.0

    image_exp = np.expand_dims(image, axis=0)

    (box_preds, label_preds) = model.predict(image_exp)
    (start_x, start_y, end_x, end_y) = box_preds[0]

    # To show only the highest probability class
    i = np.argmax(label_preds, axis=1)
    label = lb.classes_[i][0]

    # To show two classes with probabilities
    labelsprob_list = label_preds[0].tolist()
    label_1_prob = sorted(labelsprob_list)[-1]
    label_2_prob = sorted(labelsprob_list)[-2]

    label_1 = lb.classes_[labelsprob_list.index(label_1_prob)]
    label_2 = lb.classes_[labelsprob_list.index(label_2_prob)]

    label_1_prob *= 100
    label_2_prob *= 100

    start_x = int(start_x * w)
    start_y = int(start_y * h)
    end_x = int(end_x * w)
    end_y = int(end_y * h)

    width = end_x - start_x
    height = end_y - start_y
    image_scaled = image * 255

    # draw a bounding box rectangle and label on the image
    cv2.rectangle(image_scaled, (start_x, start_y),
                  (end_x, end_y), (0, 255, 0), 2)
    cv2.putText(image_scaled, label, (start_x, start_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    plot_path = os.path.join(
        "static", "x_ray.jpg")
    cv2.imwrite(plot_path, image_scaled)

    return render_template('index.html', prediction_1=label_1, pred_1_prob=f'{label_1_prob:.3}', prediction_2=label_2,
                           pred_2_prob=f'{label_2_prob:.3}', filename=imagefile.filename)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
