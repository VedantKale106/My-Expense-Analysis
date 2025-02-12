# app.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from flask import Flask, render_template, send_file

app = Flask(__name__)

# Load the data
df = pd.read_csv("Expenses.csv")

# Preprocessing
df = df[df["TYPE"] != "(+) Income"]
df["TIME"] = pd.to_datetime(df["TIME"])

def generate_plot(plot_function, high_res=False):
    """Helper function to generate base64 encoded plot"""
    figsize = (15, 10) if high_res else (10, 6)
    dpi = 300 if high_res else 100
    plt.figure(figsize=figsize, facecolor='#121212')
    plt.style.use('dark_background')
    plot_function()
    plt.tight_layout()
    
    # Save plot to a temporary buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#121212', dpi=dpi)
    plt.close()
    
    # Encode the buffer to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def plot_expense_distribution():
    """Expense Distribution by Category"""
    df.groupby("CATEGORY")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="coolwarm", startangle=90)
    plt.title("Expense Distribution by Category", color='white')
    plt.ylabel("")

def plot_category_breakdown():
    """Expense Breakdown by Category"""
    sns.barplot(x=df.groupby("CATEGORY")["AMOUNT"].sum().index, 
                y=df.groupby("CATEGORY")["AMOUNT"].sum().values, 
                palette="viridis")
    plt.xlabel("Category", color='white')
    plt.ylabel("Total Amount Spent", color='white')
    plt.title("Expense Breakdown by Category", color='white')
    plt.xticks(rotation=45, color='white')

def plot_amount_distribution():
    """Distribution of Expense Amounts"""
    sns.histplot(df["AMOUNT"], bins=5, kde=True, color="teal")
    plt.xlabel("Expense Amount", color='white')
    plt.ylabel("Frequency", color='white')
    plt.title("Distribution of Expense Amounts", color='white')

def plot_transactions_per_category():
    """Number of Transactions Per Category"""
    sns.countplot(x=df["CATEGORY"], palette="coolwarm")
    plt.xlabel("Category", color='white')
    plt.ylabel("Number of Transactions", color='white')
    plt.title("Number of Transactions Per Category", color='white')
    plt.xticks(rotation=45, color='white')

def plot_account_split():
    """Expense Split by Account Type"""
    df.groupby("ACCOUNT")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="Set3", startangle=90)
    plt.title("Expense Split by Account Type", color='white')
    plt.ylabel("")

def plot_cumulative_expenses():
    """Running Total of Expenses"""
    df["CUM_SUM"] = df["AMOUNT"].cumsum()
    sns.lineplot(x=df["TIME"], y=df["CUM_SUM"], marker="o", color="purple")
    plt.xlabel("Time", color='white')
    plt.ylabel("Cumulative Expense", color='white')
    plt.title("Running Total of Expenses", color='white')
    plt.xticks(rotation=45, color='white')

def plot_spending_by_day():
    """Spending Pattern by Day of the Week"""
    df["DAY_OF_WEEK"] = df["TIME"].dt.day_name()
    sns.barplot(x=df["DAY_OF_WEEK"], y=df["AMOUNT"], 
                estimator=sum, 
                palette="pastel", 
                order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.xlabel("Day of the Week", color='white')
    plt.ylabel("Total Amount Spent", color='white')
    plt.title("Spending Pattern by Day of the Week", color='white')
    plt.xticks(rotation=45, color='white')

def plot_spending_by_hour():
    """Spending Trend Throughout the Day"""
    df["HOUR"] = df["TIME"].dt.hour
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df["HOUR"], y=df["AMOUNT"], estimator=sum, marker="o", color="red")
    plt.xlabel("Hour of the Day", color='white')
    plt.ylabel("Total Amount Spent", color='white')
    plt.title("Spending Trend Throughout the Day", color='white')
    plt.xticks(range(0, 24), color='white')

def plot_category_spending_over_time():
    """Spending Amount by Category Over Time"""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df["TIME"], y=df["CATEGORY"], size=df["AMOUNT"], sizes=(10, 500), alpha=0.6, palette="muted")
    plt.xlabel("Time", color='white')
    plt.ylabel("Category", color='white')
    plt.title("Spending Amount by Category Over Time", color='white')
    plt.xticks(rotation=45, color='white')

def plot_essential_vs_nonessential():
    """Essential vs. Non-Essential Spending"""
    essential_categories = ["Food", "Rent", "Bills", "Transport"]
    df["NEED_WANT"] = df["CATEGORY"].apply(lambda x: "Need" if x in essential_categories else "Want")
    df.groupby("NEED_WANT")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="Pastel1", startangle=90)
    plt.title("Essential vs. Non-Essential Spending", color='white')
    plt.ylabel("")

def plot_monthly_spending():
    """Monthly Spending Trend"""
    df["MONTH"] = df["TIME"].dt.strftime("%b-%Y")
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df["MONTH"], y=df["AMOUNT"], estimator=sum, palette="coolwarm")
    plt.xlabel("Month", color='white')
    plt.ylabel("Total Amount Spent", color='white')
    plt.title("Monthly Spending Trend", color='white')
    plt.xticks(rotation=45, color='white')

def plot_spending_variability():
    """Spending Variability by Category"""
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=df["CATEGORY"], y=df["AMOUNT"], palette="coolwarm")
    plt.xlabel("Category", color='white')
    plt.ylabel("Amount Spent", color='white')
    plt.title("Spending Variability by Category", color='white')
    plt.xticks(rotation=45, color='white')
# Dictionary to store plot generation functions
PLOT_FUNCTIONS = {
    'expense-distribution': plot_expense_distribution,
    'category-breakdown': plot_category_breakdown,
    'amount-distribution': plot_amount_distribution,
    'transactions-per-category': plot_transactions_per_category,
    'account-split': plot_account_split,
    'cumulative-expenses': plot_cumulative_expenses,
    'spending-by-day': plot_spending_by_day,
    'spending-by-hour': plot_spending_by_hour,
    'category-spending-over-time': plot_category_spending_over_time,
    'essential-vs-nonessential': plot_essential_vs_nonessential,
    'monthly-spending': plot_monthly_spending,
    'spending-variability': plot_spending_variability
}

@app.route('/')
def dashboard():
    # Generate all plots
    plots = {
        'Expense Distribution': ('expense-distribution', generate_plot(plot_expense_distribution)),
        'Category Breakdown': ('category-breakdown', generate_plot(plot_category_breakdown)),
        'Amount Distribution': ('amount-distribution', generate_plot(plot_amount_distribution)),
        'Transactions per Category': ('transactions-per-category', generate_plot(plot_transactions_per_category)),
        'Account Split': ('account-split', generate_plot(plot_account_split)),
        'Cumulative Expenses': ('cumulative-expenses', generate_plot(plot_cumulative_expenses)),
        'Spending by Day': ('spending-by-day', generate_plot(plot_spending_by_day)),
        'Spending by Hour': ('spending-by-hour', generate_plot(plot_spending_by_hour)),
        'Category Spending Over Time': ('category-spending-over-time', generate_plot(plot_category_spending_over_time)),
        'Essential vs Non-Essential': ('essential-vs-nonessential', generate_plot(plot_essential_vs_nonessential)),
        'Monthly Spending': ('monthly-spending', generate_plot(plot_monthly_spending)),
        'Spending Variability': ('spending-variability', generate_plot(plot_spending_variability))
    }
    
    return render_template('dashboard.html', plots=plots)

@app.route('/graph/<plot_id>')
def graph_detail(plot_id):
    if plot_id not in PLOT_FUNCTIONS:
        return "Graph not found", 404
    
    # Generate high-resolution plot
    plot_function = PLOT_FUNCTIONS[plot_id]
    plot_url = generate_plot(plot_function, high_res=True)
    
    # Get plot title and description
    plot_details = {
        'expense-distribution': {
            'title': 'Expense Distribution by Category',
            'description': 'A comprehensive breakdown of expenses across different categories, showing the percentage of total spending for each category.'
        },
        'category-breakdown': {
            'title': 'Category Expense Breakdown',
            'description': 'Detailed bar chart showing the total amount spent in each expense category.'
        },
        'amount-distribution': {
            'title': 'Expense Amount Distribution',
            'description': 'Histogram displaying the frequency of different expense amounts.'
        },
        'transactions-per-category': {
            'title': 'Transactions per Category',
            'description': 'Number of transactions made in each expense category.'
        },
        'account-split': {
            'title': 'Expense Split by Account',
            'description': 'Pie chart showing the distribution of expenses across different accounts.'
        },
        'cumulative-expenses': {
            'title': 'Cumulative Expenses Over Time',
            'description': 'Line graph tracking the total expenses accumulated over time.'
        },
        'spending-by-day': {
            'title': 'Spending Pattern by Day of Week',
            'description': 'Bar chart showing total spending for each day of the week.'
        },
        'spending-by-hour': {
            'title': 'Spending Trend Throughout the Day',
            'description': 'Line graph illustrating spending patterns across different hours of the day.'
        },
        'category-spending-over-time': {
            'title': 'Category Spending Over Time',
            'description': 'Scatter plot showing spending amounts for different categories across time.'
        },
        'essential-vs-nonessential': {
            'title': 'Essential vs Non-Essential Spending',
            'description': 'Pie chart comparing spending on essential and non-essential categories.'
        },
        'monthly-spending': {
            'title': 'Monthly Spending Trend',
            'description': 'Bar chart displaying total spending for each month.'
        },
        'spending-variability': {
            'title': 'Spending Variability by Category',
            'description': 'Violin plot showing the distribution and variability of spending across categories.'
        }
    }
    
    details = plot_details.get(plot_id, {'title': 'Graph Details', 'description': 'Detailed view of the expense graph.'})
    
    return render_template('graph_detail.html', plot_url=plot_url, **details)

if __name__ == '__main__':
    app.run(debug=True)