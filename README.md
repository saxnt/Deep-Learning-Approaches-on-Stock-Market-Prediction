# Deep-Learning-Approaches-on-Stock-Market-Prediction

A comprehensive study comparing deep learning and traditional machine learning models for NIFTY 50 stock market prediction.

📊 Project Overview
This project explores various machine learning and deep learning approaches to predict stock market trends using the NIFTY 50 dataset. We compare the performance of traditional models (Random Forest, SVM) with advanced deep learning architectures (LSTM, Bi-LSTM, Bi-RNN) for financial time series forecasting.

👥 Team Members
Garima Roy (UG/02/BTCSEDA/2023/002)

Saikat Das (UG/02/BTCSEAIML/2023/074)

Md. Zayed Ali (UG/02/BTCSEDA/2023/001)

Sk. Nahid Faiyaz (UG/02/BTCSEDA/2023/005)

Digant Mishra (UG/02/BTCSECSF/2023/016)

Project Supervisor: Ms. Debasree Mitra
Institution: School of Engineering & Technology, ADAMAS University, Kolkata

🎯 Objectives
Evaluate deep learning models (LSTM, Bi-LSTM, Bi-RNN) for stock price prediction

Compare performance with traditional ML models (Random Forest, SVM)

Analyze hybrid approaches combining deep learning with classical methods

Provide insights into the practical value of AI for financial forecasting

🗂️ Dataset
NIFTY 50 Historical Stock Data

Source: India's National Stock Exchange (NSE)

Features: Open, High, Low, Close, Volume

Coverage: Top 50 companies across multiple sectors

Time Period: 2019–2024

🔧 Methodology
Data Preprocessing
Missing Value Handling: Forward-fill techniques

Normalization: Min-Max scaling to [0, 1]

Feature Engineering: Moving averages, RSI, MACD

Sequence Formatting: 60-day rolling windows for RNN models

Models Implemented
Deep Learning Models
LSTM: Captures long-term dependencies; handles vanishing gradient

Bi-LSTM: Bidirectional processing for better temporal recognition

Bi-RNN: Forward and backward sequence learning

CNN-LSTM Hybrid: CNN for local patterns + LSTM for sequential modeling

Traditional ML Models
Random Forest: Ensemble learning, reduces overfitting

SVM: Tuned with RandomizedSearchCV

Hybrid Approaches
Bi-RNN + Random Forest

Bi-LSTM + Random Forest

LSTM + SVM

RandomizedSearchCV + SVM

📈 Results
Performance Comparison
Model	MAE	MSE	RMSE
Bi-RNN + SVM	0.1229	0.1229	0.1229
Bi-RNN + Random Forest	0.0327	0.0327	0.0327
Bi-LSTM + Random Forest	443.64	1552.93	1552.93
Randomized SearchCV (SVM)	5.8764	0.06	6.4785
LSTM + SVM	328.01	7414.90	7414.90
CNN-LSTM	0.0165	0.0005	—

Key Findings
✅ Deep Learning Advantages:

Captures non-linear patterns better

Models long-term dependencies

Reduces manual feature engineering

✅ Best Performing Models:

CNN-LSTM Hybrid: Lowest overall error

Bi-RNN + Random Forest: Balance of accuracy & stability

✅ Traditional ML Insights:

Good performance with solid preprocessing

Faster training

Easy to interpret

🔮 Future Scope
Enhanced Data Integration
News sentiment analysis

Macroeconomic indicators (Interest rates, GDP, etc.)

Global market correlations (e.g., S&P 500)

Advanced Architectures
Attention mechanisms

Transformers (BERT, GPT)

Ensemble methods

Real-World Applications
Real-time predictions

Multi-asset portfolio modeling

Risk management systems

🛠️ Technologies Used
Programming: Python

Deep Learning: TensorFlow/Keras, PyTorch

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Optimization: GridSearchCV, RandomizedSearchCV

📊 Evaluation Metrics
MAE: Mean Absolute Error

MSE: Mean Squared Error

RMSE: Root Mean Squared Error

R² Score: Explained variance

🚀 Getting Started
Prerequisites
bash
Copy
Edit
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
Basic Usage
python
Copy
Edit
# Load and preprocess data
from data_preprocessing import load_nifty_data, preprocess_data

data = load_nifty_data('nifty50_data.csv')
processed_data = preprocess_data(data)

# Train Bi-LSTM model
from models import BiLSTMModel
model = BiLSTMModel()
model.fit(processed_data)

# Make predictions
predictions = model.predict(test_data)
📚 Key References
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.

Fischer, T., & Krauss, C. (2018). Deep learning with LSTM networks for financial market predictions. European Journal of Operational Research.

Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and LSTM. PLoS One.

🤝 Contributing
We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

📞 Contact
For questions or collaboration opportunities, reach out to any team member or our supervisor Ms. Debasree Mitra at ADAMAS University, Kolkata.

Note: This project was completed as part of the B.Tech Computer Science & Engineering curriculum at ADAMAS University, Kolkata (Jan 2025 – May 2025).

