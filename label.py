from flask import Flask, render_template, request
import os
import csv
from natsort import natsorted

app = Flask(__name__)

# Load existing labels
def load_labels():
    labels = {}
    try:
        with open('plates.csv', 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                labels[row['file_name']] = row['text']
    except FileNotFoundError:
        pass
    return labels

# Route for the main page
@app.route('/')
def index():
    image_files = os.listdir('static/')

    # natsort the files using natsorted
    image_files = natsorted(image_files)

    labels = load_labels()
    print(labels)
    return render_template('index.html', images=image_files, labels=labels)

# Route to handle label updates
@app.route('/update_label', methods=['POST'])
def update_label():
    if request.method == 'POST':
        filename = request.form['filename']
        label = request.form['label']
        file_exists = os.path.isfile('plates.csv')
        
        # Read existing data
        data = []
        if file_exists:
            with open('plates.csv', 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader if row['file_name'] != f'plates/{filename}']
        
        # Append the new/updated label
        data.append({'file_name': f'plates/{filename}', 'text': label})
        
        # Write updated data back to the CSV, preserving the header
        with open('plates.csv', 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
