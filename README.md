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
Time Period: Multi-year historical data (2019-2024)

🔧 Methodology
Data Preprocessing

Missing Value Handling: Forward-fill techniques
Normalization: Min-Max scaling to [0,1] range
Feature Engineering: Moving averages, RSI, MACD
Sequence Formatting: 60-day rolling windows for RNN models

Models Implemented
Deep Learning Models

LSTM (Long Short-Term Memory)

Captures long-term dependencies in time series
Handles vanishing gradient problem


Bi-LSTM (Bidirectional LSTM)

Processes sequences in both directions
Enhanced temporal pattern recognition


Bi-RNN (Bidirectional RNN)

Forward and backward sequence processing
Improved context understanding


CNN-LSTM Hybrid

Combines CNN local pattern recognition with LSTM sequence modeling



Traditional ML Models

Random Forest

Ensemble learning with multiple decision trees
Reduces overfitting through averaging


Support Vector Machine (SVM)

Optimized with RandomizedSearchCV
Hyperparameter tuning for optimal performance



Hybrid Approaches

Bi-RNN + Random Forest
Bi-LSTM + Random Forest
LSTM + SVM
RandomizedSearchCV + SVM

📈 Results
Performance Comparison
ModelMAEMSERMSEBi-RNN with SVM0.12290.12290.1229Bi-RNN with Random Forest0.03270.03270.0327Bi-LSTM with Random Forest443.6411552.9398552.9398Randomized Search CV (SVM)5.87640.06.4785LSTM with SVM328.0172414.9087414.9087LSTM with CNN0.01650.0005—
Key Findings
✅ Deep Learning Advantages:

Superior at capturing non-linear temporal patterns
Better long-term dependency modeling
Reduced need for manual feature engineering

✅ Best Performing Models:

CNN-LSTM Hybrid: Lowest overall error rates
Bi-RNN with Random Forest: Excellent balance of accuracy and stability

✅ Traditional ML Insights:

Competitive performance with proper preprocessing
Faster training times
Good interpretability

🔮 Future Scope
Enhanced Data Integration

News Sentiment Analysis: Market-related news impact
Macroeconomic Indicators: Interest rates, inflation, GDP
Global Market Correlations: S&P 500, Dow Jones integration

Advanced Architectures

Attention Mechanisms: Focus on relevant sequence parts
Transformer Models: BERT/GPT-based approaches
Ensemble Methods: Combined model predictions

Real-World Applications

Real-Time Predictions: Live market condition analysis
Multi-Asset Portfolios: Cross-asset correlation modeling
Risk Management: Dynamic portfolio optimization

🛠️ Technologies Used

Programming: Python
Deep Learning: TensorFlow/Keras, PyTorch
Machine Learning: Scikit-learn
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Optimization: GridSearchCV, RandomizedSearchCV

📊 Evaluation Metrics

MAE (Mean Absolute Error): Average absolute prediction errors
MSE (Mean Squared Error): Average squared differences
RMSE (Root Mean Squared Error): Standard deviation of residuals
R² Score: Proportion of variance explained

🚀 Getting Started
Prerequisites
bashpip install tensorflow pandas numpy scikit-learn matplotlib seaborn
Basic Usage
python# Load and preprocess data
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
Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and LSTM. PloS One.

🤝 Contributing
We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact
For questions or collaboration opportunities, please reach out to any of the team members or our supervisor Ms. Debasree Mitra at ADAMAS University.

Note: This project was completed as part of the B.Tech Computer Science & Engineering curriculum at ADAMAS University, Kolkata (Jan 2025 - May 2025).
