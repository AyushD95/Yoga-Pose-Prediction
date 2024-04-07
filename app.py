from flask import Flask, render_template, request
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
import io

# Define your classes_dict and YogaClassifier class as in your prediction code
classes_dict = {0: 'Adho Mukha Svanasana', 1: 'Adho Mukha Vrksasana', 2: 'Alanasana', 3: 'Anjaneyasana', 4: 'Ardha Chandrasana', 5: 'Ardha Matsyendrasana', 6: 'Ardha Navasana', 7: 'Ardha Pincha Mayurasana', 8: 'Ashta Chandrasana', 9: 'Baddha Konasana', 10: 'Bakasana', 11: 'Balasana', 12: 'Bitilasana', 13: 'Camatkarasana', 14: 'Dhanurasana', 15: 'Eka Pada Rajakapotasana', 16: 'Garudasana', 17: 'Halasana', 18: 'Hanumanasana', 19: 'Malasana', 20: 'Marjaryasana', 21: 'Navasana', 22: 'Padmasana', 23: 'Parsva Virabhadrasana', 24: 'Parsvottanasana', 25: 'Paschimottanasana', 26: 'Phalakasana', 27: 'Pincha Mayurasana', 28: 'Salamba Bhujangasana', 29: 'Salamba Sarvangasana', 30: 'Setu Bandha Sarvangasana', 31: 'Sivasana', 32: 'Supta Kapotasana', 33: 'Trikonasana', 34: 'Upavistha Konasana', 35: 'Urdhva Dhanurasana', 36: 'Urdhva Mukha Svsnssana', 37: 'Ustrasana', 38: 'Utkatasana', 39: 'Uttanasana', 40: 'Utthita Hasta Padangusthasana', 41: 'Utthita Parsvakonasana', 42: 'Vasisthasana', 43: 'Virabhadrasana One', 44: 'Virabhadrasana Three', 45: 'Virabhadrasana Two', 46: 'Vrksasana'}


class YogaClassifier(torch.nn.Module):
    # Your implementation here
    def __init__(self, num_classes, input_length):
        super(YogaClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_length, 64)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.layer2 = torch.nn.Linear(64, 64)
        self.outlayer = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x
    pass


def load_model():
    # Your implementation here
    model_pose = YogaClassifier(num_classes=len(classes_dict), input_length=32)
    model_pose.load_state_dict(torch.load("best.pth"))
    model_pose.eval()
    return model_pose
    pass


def make_prediction(model, image_path):
    model_yolo = YOLO("yolov8x-pose-p6.pt")

    results = model_yolo.predict(image_path, verbose=False)
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        keypoints = r.keypoints.xyn.cpu().numpy()[0]
        keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()

        # Preprocess keypoints data
        keypoints_tensor = torch.tensor(keypoints[2:], dtype=torch.float32).unsqueeze(0)

        # Prediction
        model.cpu()
        model.eval()
        with torch.no_grad():
            logit = model(keypoints_tensor)
            pred = torch.softmax(logit, dim=1).argmax(dim=1).item()
            prediction = classes_dict[pred]

        # Convert the plot image to base64 string
        image = io.BytesIO()
        plt.imshow(im_array[..., ::-1])
        plt.title(f"Prediction: {prediction}", color="green")
        plt.savefig(image, format='png')
        plt.close()
        image.seek(0)
        plot_base64 = base64.b64encode(image.read()).decode('utf-8')
        return plot_base64, prediction


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['file']
    # Save the image file to a temporary location
    image_path = 'temp.png'
    image_file.save(image_path)

    # Load the model
    model = load_model()

    # Make prediction
    plot_base64, prediction = make_prediction(model, image_path)

    # Remove the temporary image file
    os.remove(image_path)

    # Return the Prediction result and plot to the prediction.html template
    return render_template('prediction.html', prediction=prediction, plot_base64=plot_base64)


if __name__ == '__main__':
    app.run(debug=True)


