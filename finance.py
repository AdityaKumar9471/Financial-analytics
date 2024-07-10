import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def SMA(df, n):
    return df["Close"].rolling(window=n).mean()


def EMA(df, n):
    return df["Close"].ewm(span=n, adjust=False).mean()


def RSI(df, n=14):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def MACD(df, short_window=12, long_window=26, signal_window=9):
    short_ema = EMA(df, short_window)
    long_ema = EMA(df, long_window)
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df["MACD"].ewm(span=signal_window).mean()
    return df


def generate_signals(df):

    trading_days = len(df)

    if trading_days <= 50:
        short_sma = 5
        long_sma = 10

    elif trading_days <= 200:
        short_sma = 10
        long_sma = 50
    else:
        short_sma = 20
        long_sma = 80
    if trading_days <= 50:
        df['SMA_Short'] = EMA(df, short_sma)
        df['SMA_Long'] = EMA(df, long_sma)
    else:
       df['SMA_Short'] = SMA(df, short_sma)
       df['SMA_Long'] = SMA(df, long_sma)


    df['RSI'] = RSI(df)
    df = MACD(df)

    df['Signal'] = np.where(df["SMA_Short"] > df["SMA_Long"], 1, 0)
    df['Position'] = df['Signal'].diff()
    return df

def signal_gen(df):
    df["Signal_1"]=np.where(df["MACD"]>df["Signal_Line"],1,0)
    df["Position_1"]=df["Signal_1"].diff()

    return df

def buy_sell(df):
    df["buy_1"]=np.where(df["Position_1"]==1,df["MACD"],np.nan)
    df["sell_1"]=np.where(df["Position_1"]==-1,df["MACD"],np.nan)
    return df


def buy_and_sell(df):
    df['Buy'] = np.where(df['Position'] == 1, df['SMA_Short'], np.nan)
    df['Sell'] = np.where(df['Position'] == -1, df['SMA_Short'], np.nan)
    return df


def portfolio_performance(df, initial_investment=100000):
    df['Portfolio_Value'] = initial_investment
    shares = 0
    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:  # Buy signal
            shares = df['Portfolio_Value'].iloc[i - 1] / df['Close'].iloc[i]
            df['Portfolio_Value'].iloc[i] = shares * df['Close'].iloc[i]
        elif df['Position'].iloc[i] == -1:  # Sell signal
            df['Portfolio_Value'].iloc[i] = shares * df['Close'].iloc[i]
            shares = 0
        else:  # Hold
            if shares > 0:
                df['Portfolio_Value'].iloc[i] = shares * df['Close'].iloc[i]
            else:
                df['Portfolio_Value'].iloc[i] = df['Portfolio_Value'].iloc[i - 1]

    overall_return = df['Portfolio_Value'].iloc[-1] - initial_investment
    return df, overall_return

def portfolio_performance_1(df, initial_investment=100000):
    df['Portfolio_Value_1'] = initial_investment
    shares_1 = 0
    for i in range(1, len(df)):
        if df['Position_1'].iloc[i] == 1:  # Buy signal
            shares_1 = df['Portfolio_Value_1'].iloc[i - 1] / df['Close'].iloc[i]
            df['Portfolio_Value_1'].iloc[i] = shares_1 * df['Close'].iloc[i]
        elif df['Position_1'].iloc[i] == -1:  # Sell signal
            df['Portfolio_Value_1'].iloc[i] = shares_1 * df['Close'].iloc[i]
            shares_1 = 0
        else:  # Hold
            if shares_1 > 0:
                df['Portfolio_Value_1'].iloc[i] = shares_1 * df['Close'].iloc[i]
            else:
                df['Portfolio_Value_1'].iloc[i] = df['Portfolio_Value_1'].iloc[i - 1]

    overall_return_1 = df['Portfolio_Value_1'].iloc[-1] - initial_investment
    return df, overall_return_1



# Streamlit App
st.title("Stock Analysis and Trading Strategy")

# User Input
ticker = st.text_input("Enter Stock Ticker")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
Investment=st.number_input("Enter initial investment")

if ticker:
    # Fetch Data
    df = yf.download(ticker, start=start_date, end=end_date)
    df = pd.DataFrame(df)
    df = generate_signals(df)
    df = buy_and_sell(df)
    df=signal_gen(df)
    df=buy_sell(df)
    df, overall_return = portfolio_performance(df,Investment)
    df,overall_return_1= portfolio_performance_1(df,Investment)

    # Display Data
    st.subheader(f"Stock Data for {ticker}")
    st.subheader("Closing Prices with MAs")
    st.line_chart(df[['Close', 'SMA_Short', 'SMA_Long']])

    st.subheader("RSI")
    st.line_chart(df[['RSI']])

    st.subheader("MACD and Signal Line")
    st.line_chart(df[['MACD', 'Signal_Line']])

    st.subheader("Portfolio Value Over Time")
    st.line_chart(df['Portfolio_Value'])

    # Buy/Sell Signals
    st.subheader("Buy and Sell Signals")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(df['Close'], label='Closing Prices')
    ax.plot(df['SMA_Short'], label=f'SMA Short ({df["SMA_Short"].name})')
    ax.plot(df['SMA_Long'], label=f'SMA Long ({df["SMA_Long"].name})', color='black')
    ax.scatter(df.index, df['Buy'], marker='^', color='green', label='Buy')
    ax.scatter(df.index, df['Sell'], marker='v', color='red', label='Sell')
    ax.legend()
    st.pyplot(fig)

    # Portfolio Performance
    st.subheader("Portfolio Performance")
    cumulative_return = (df['Portfolio_Value'].iloc[-1] / Investment - 1) * 100
    st.write(f"Overall return: {overall_return}" )
    st.write(f"Cumulative Return: {cumulative_return:.2f}%")

    cumulative_return_1= (df['Portfolio_Value_1'].iloc[-1] / Investment - 1) * 100
    st.write(f"Overall return based on MACD: {overall_return_1}")
    st.write(f" Cumulative Return based on MACD: {cumulative_return_1: .2f}%")

else:
    st.write("Please enter a stock ticker.")
