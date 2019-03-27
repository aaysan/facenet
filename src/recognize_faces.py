#use FLASK_APP=recognize_faces.py flask run

from flask import request
import os
import time
import tensorflow as tf
import src.facenet as facenet
import src.classify_faces as classify_faces
import cv2 as cv


from flask import Flask

def create_app():
    app = Flask(__name__)
    return app

app = create_app()
sess = None
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
print("TESTING")

graph = None
images_placeholder = None
embeddings = None
phase_train_placeholder = None
embedding_size = None

@app.before_first_request
def setup_model():
    global graph,images_placeholder,embeddings,phase_train_placeholder,embedding_size,sess
    currentdir = os.getcwd()
    pythonpath = currentdir  ##+ "/facenet/src"
    os.environ["PYTHONPATH"] = pythonpath
    model = "../pretrained_model/"

    graph = tf.Graph()
    print("Hello")

    # with graph.as_default():
    sess = tf.Session();
    facenet.load_model(model, sess)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    print("loaded")

        # app.run(debug=False)



@app.route("/")
def hello():
    return "Hello World!"


@app.route('/get_name', methods=['POST'])
def get_name():

    tstart = time.time()
    os.system("rm -f aligned/somename/*")
    file = request.files['file']

    print(file.filename)
    file.save("aligned/somename/" + file.filename)

    img = cv.imread("aligned/somename/" + file.filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    roi_color = img[y:y + h, x:x + w]

    res = cv.resize(roi_color, dsize=(160, 160), interpolation=cv.INTER_CUBIC)

    cv.imwrite("aligned/somename/tmp.png", res)

    t0 = time.time()
    a, b = classify_faces.classify_face(sess,graph,images_placeholder,embeddings,phase_train_placeholder,embedding_size)

    t1 = time.time()

    print("Total time: ", str(t1-t0))

    res = "Name: " + str(a) + "\n" + \
        "align time %0.3f seconds \n" % (t0-tstart) + \
        "probability of it being correct: " + str(b[0]) + "\n" + \
        "time for classification: %0.3f seconds\n" % (t1-t0)

    return res




if __name__ == '__main__':
    pass
    # currentdir = os.getcwd()
    # pythonpath = currentdir  ##+ "/facenet/src"
    # os.environ["PYTHONPATH"] = pythonpath
    # model = "../../pretrained_model/"
    #
    # graph = tf.Graph()
    # print("Hello")
    #
    # # with graph.as_default():
    # with tf.Session() as sess:
    #     facenet.load_model(model)
    #     images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #     embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #     phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #     embedding_size = embeddings.get_shape()[1]
    #     print("loaded")
    #
    #     app.run(debug=False)