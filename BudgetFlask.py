from flask import Flask, request, jsonify
import joblib

# Load the saved model and vectorizer
loaded_model = joblib.load('ridge_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask app
app = Flask(__name__)

# Define the prediction function
def predict_budget_with_average(objective_text, budget_limit):
    objective_tfidf = loaded_vectorizer.transform([objective_text])
    predicted_budget = loaded_model.predict(objective_tfidf)[0]
    averaged_budget = (predicted_budget + budget_limit) / 2
    return averaged_budget

# Define the API route for prediction
@app.route('/predict_budget', methods=['POST'])
def predict_budget():
    data = request.get_json()
    objective_text = data.get('objective')
    budget_limit = data.get('budget_limit')

    if not objective_text or budget_limit is None:
        return jsonify({'error': 'Please provide both "objective" and "budget_limit".'}), 400

    averaged_budget = predict_budget_with_average(objective_text, budget_limit)
    return jsonify({'averaged_budget': averaged_budget})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
