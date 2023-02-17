from flask import Flask, render_template , Response, session 
from Functions import *


#Create the app object
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    return render_template('index2.html',_external=True, Class_Value=str(Class_Value(BMI_Val())) , BMI_Value = BMI_Val())
    

@app.route('/video/<bmi>',methods=['POST','GET'])
def video(bmi):
    b = bmi
    print('Application py file',b)
    return Response(generate_frames(b),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)