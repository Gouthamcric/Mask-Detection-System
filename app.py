from flask import Flask,request,render_template,url_for,session,redirect,Response
import bcrypt
from pymongo import MongoClient
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
from datetime import date
import cv2
import random


app = Flask(__name__)
cluster = MongoClient("mongodb+srv://goucric:gnq316@cluster0.ymixq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = cluster["mask_detection_system"]
collection = db["user"]
collection_data = db["data"]

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)


def generate_frames():
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = load_model("mask_detector.model")
    vs = VideoStream(src=0).start()
    i=0
    l=''
    l2=''
    while True:
        frame = vs.read() #reading frame
        frame = imutils.resize(frame, width=400) 

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):   #tuple unpacking
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            l2 = label
            l='with_mask' if mask>withoutMask else 'without_mask'
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret,buffer=cv2.imencode('.jpg',frame)
        cv2.imwrite('dataset/'+l+'/img'+str(i)+'.jpg',frame)
        cv2.imwrite('static/img'+str(i)+'.jpg',frame)
        file_loc = 'static/img'+str(i)+'.jpg'
        img_no = 'img'+str(i)+'.jpg'
        
        with open(file_loc, 'rb') as f:
            contents = f.read()
       
        today = date.today()
        d1 = today.strftime("%d/%m/%Y")
        t = time.localtime()
        t1 = time.strftime("%H:%M:%S", t)
        collection_data.insert_one({'date':d1,'time':t1,'img':contents,'img_no':img_no,'status':l2})
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        i=random.randint(0,1000000)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    

@app.route('/',methods=['POST','GET'])
def index():

      if 'username' in session:
            return render_template('predict.html',task="predict")
            
      return render_template('login.html',flag=0,task="database")

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/database',methods=['POST','GET'])
def database():
    return render_template('show_data.html',results = collection_data.find(),i=1,task="database")

@app.route('/login',methods=['POST'])
def login():
      user = collection.find_one({'username':request.form['username']})

      if user:
            if bcrypt.hashpw(request.form['password'].encode('utf-8'),user['password']) == user['password']:
                  session['username'] = request.form['username']
                  return redirect(url_for('index'))
      return render_template('login.html',flag=1)

@app.route('/register',methods=['POST','GET'])
def register():
      if request.method == 'POST':
            existing_user = collection.find({"username":request.form['username']})
            
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'),bcrypt.gensalt())
            collection.insert_one({'username':request.form['username'],'password':hashpass,'first_name':request.form['first_name'],'last_name':request.form['last_name']})
            if(request.form['phone'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"phone":request.form['phone']}})
            if(request.form['city'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"city":request.form['city']}})
            if(request.form['state'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"state":request.form['state']}})
            if(request.form['address'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"address":request.form['address']}})
            
            session['username'] = request.form['username']
            return redirect(url_for('index'))
      #return 'Username already exists!'
      return render_template('register.html')

@app.route('/settings',methods=['POST','GET'])
def settings():
      if request.method == 'POST':
            collection.update_one({"username":session["username"]},{"$set":{'username':request.form['username'],'first_name':request.form['first_name'],'last_name':request.form['last_name']}})
            session['username'] = request.form['username']
            if(request.form['phone'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"phone":request.form['phone']}})
            if(request.form['city'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"city":request.form['city']}})
            if(request.form['state'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"state":request.form['state']}})
            if(request.form['address'] != ""):
                  collection.update_one({"username":request.form['username']},{"$set":{"address":request.form['address']}})
            res = collection.find({"username":session['username']})
            return render_template("settings.html",res=res,res2=res,task="settings")
      res = collection.find({"username":session['username']})
      return render_template("settings.html",res=res,res2=res,task="settings")

@app.route('/delete')
def delete():
      collection.remove({"username":session['username']})
      return render_template("login.html")

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run()