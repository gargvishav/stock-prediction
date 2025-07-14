# daily_finetune.py

from your_pipeline import fetch_new_bar, add_technical_indicators, fine_tune_on_new_data

def main():
    # 1) Pull today’s data (or last available bar)
    new_bar = fetch_new_bar('AAPL')              # return 1-row DataFrame
    new_bar_ind = add_technical_indicators(new_bar)
    # 2) Fine-tune your model
    fine_tune_on_new_data(new_bar_ind)

if __name__ == "__main__":
    main()
