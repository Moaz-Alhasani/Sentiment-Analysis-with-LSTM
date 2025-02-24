# Sentiment-Analysis-with-LSTM

# Overview

This project implements a sentiment analysis model using a Bidirectional LSTM neural network. The dataset consists of labeled sentences from Yelp, Amazon, and IMDb reviews, which are used to train the model for binary classification (positive or negative sentiment).

# Dataset

The dataset includes three sources:

yelp_labelled.txt

amazon_cells_labelled.txt

imdb_labelled.txt

Each file contains sentences labeled with 0 (negative) or 1 (positive).

# Dependencies

Ensure you have the following dependencies installed:

pip install pandas scikit-learn tensorflow keras

# Model Architecture

Embedding Layer: Converts words into dense vectors.

Bidirectional LSTM (64 units, 32 units): Captures dependencies from both past and future words.

Dense Layers: Final classification layers with ReLU and sigmoid activations.

Loss Function: Binary cross-entropy.

Optimizer: Adam.

# Training the Model

The dataset is split into training and testing sets. The text is tokenized, converted into sequences, and padded before being fed into the model.

To train the model, run:

model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_data=(X_test_pad, y_test))

# Model Evaluation

After training, evaluate the model on the test set:

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Predicting Sentiment

Use the predict_sentiment function to analyze new text:

def predict_sentiment(text, tokenizer, model, max_length):
    seq = tokenizer.texts_to_sequences([text])
    seq_pad = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(seq_pad)[0][0]
    sentiment = "pos 😀" if prediction >= 0.5 else "neg 😞"
    return sentiment, prediction

text_input = "This product is amazing!"
sentiment, score = predict_sentiment(text_input, tokenizer, model, max_length)
print(f"Sentiment: {sentiment} (Score: {score:.4f})")

# Example Output

Sentiment: pos 😀 (Score: 0.8723)

# Notes

The model can be fine-tuned with more data and hyperparameter tuning.

Extend this project by integrating a web API for real-time sentiment analysis.

License

This project is licensed under the MIT License.


