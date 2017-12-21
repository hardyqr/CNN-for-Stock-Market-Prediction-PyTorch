# Can Machine Reads Like Analysts Do?

- Train a CNN to read candlestick graphs, predicting future trend.

- Training & testing Dataset from [Huge Stock Market Dataset-Full Historical Daily Price + Volume Data For All U.S. Stocks & ETFs](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs).

- Pytorch implementation

## Two Approaches
### Approach 1
- `cnn4matrix.py`

- Apply convolution on data matrices directly.

Input: (5\*n) matrices -> (Open,High,Low,Close,Volume)\*(d1,d2,...,dn)

Output: classification result

### Approach 2
- `cnn.py`

- Generate candlestick graphs first.
![sample](https://github.com/hardyqr/CNN-for-Stock-Market-Prediction/blob/master/screen_shots_logs/sample.png)

Input: candlestick graphs

Output: classification result


## Current results

11 layers + residual block
![prediction](https://github.com/hardyqr/Deep-Learning-for-Stock-Market-Prediction/blob/master/screen_shots_logs/sota/acc+loss.png)
