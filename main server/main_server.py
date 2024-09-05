from flask import Flask, jsonify
import requests

app = Flask(__name__)

@app.route('/earthquakes', methods=['GET'])
def get_cities():
    
        # Make a GET request to the Flask server running on localhost:5000
        response = requests.get('https://localhost:5000/predict')
        
       
        cities_data = response.json()
        return jsonify(cities_data), 200
      
if __name__ == '__main__':
    app.run(host='localhost', port=3000)
