import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from skimage import io
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_similarity(img1_path, img2_path):
    # Load images using skimage.io.imread
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    
    # Convert images to grayscale if they are not already
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Ensure both images have the same dimensions by resizing
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Calculate the Structural Similarity Index (SSI)
    similarity_score = ssim(img1, img2)
    
    return similarity_score

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the POST request has the file part
        if "file1" not in request.files or "file2" not in request.files:
            return render_template("upload.html", error="Please select two images to compare.")
        
        file1 = request.files["file1"]
        file2 = request.files["file2"]
        
        # Check if the file extensions are allowed
        if not allowed_file(file1.filename) or not allowed_file(file2.filename):
            return render_template("upload.html", error="Invalid file format. Allowed formats: jpg, jpeg, png.")
        
        # Save the uploaded files
        img1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
        img2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
        file1.save(img1_path)
        file2.save(img2_path)
        
        # Calculate image similarity
        similarity_score = image_similarity(img1_path, img2_path)
        
        return render_template("upload.html", similarity=similarity_score)
    
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
