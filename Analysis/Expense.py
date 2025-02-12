#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Expenses.csv")


# In[3]:


df.head(20)


# In[4]:


# Remove rows where TYPE is "(+) Income"
df = df[df["TYPE"] != "(+) Income"]


# In[5]:


df["TIME"] = pd.to_datetime(df["TIME"])
df["DATE"] = df["TIME"].dt.date


# In[6]:


# Expense Distribution by Category
plt.figure(figsize=(6, 6))
df.groupby("CATEGORY")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="coolwarm", startangle=90)
plt.title("Expense Distribution by Category")
plt.ylabel("")  # Hide y-label
plt.show()


# In[7]:


# Expense Breakdown by Category (Bar Plot)
plt.figure(figsize=(7, 4))
sns.barplot(x=df.groupby("CATEGORY")["AMOUNT"].sum().index, y=df.groupby("CATEGORY")["AMOUNT"].sum().values, palette="viridis")
plt.xlabel("Category")
plt.ylabel("Total Amount Spent")
plt.title("Expense Breakdown by Category")
plt.show()


# In[8]:


plt.figure(figsize=(7, 4))
sns.histplot(df["AMOUNT"], bins=5, kde=True, color="teal")
plt.xlabel("Expense Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Expense Amounts")
plt.show()


# In[9]:


plt.figure(figsize=(7, 4))
sns.countplot(x=df["CATEGORY"], palette="coolwarm")
plt.xlabel("Category")
plt.ylabel("Number of Transactions")
plt.title("Number of Transactions Per Category")
plt.show()


# In[10]:


plt.figure(figsize=(6, 6))
df.groupby("ACCOUNT")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="Set3", startangle=90)
plt.title("Expense Split by Account Type")
plt.ylabel("")  # Hide y-label
plt.show()


# In[11]:


df["CUM_SUM"] = df["AMOUNT"].cumsum()

plt.figure(figsize=(8, 4))
sns.lineplot(x=df["TIME"], y=df["CUM_SUM"], marker="o", color="purple")
plt.xlabel("Time")
plt.ylabel("Cumulative Expense")
plt.title("Running Total of Expenses")
plt.xticks(rotation=45)
plt.show()


# In[12]:


df["DAY_OF_WEEK"] = df["TIME"].dt.day_name()

plt.figure(figsize=(8, 4))
sns.barplot(x=df["DAY_OF_WEEK"], y=df["AMOUNT"], estimator=sum, palette="pastel", order=[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])
plt.xlabel("Day of the Week")
plt.ylabel("Total Amount Spent")
plt.title("Spending Pattern by Day of the Week")
plt.xticks(rotation=45)
plt.show()


# In[13]:


df["HOUR"] = df["TIME"].dt.hour

plt.figure(figsize=(8, 4))
sns.lineplot(x=df["HOUR"], y=df["AMOUNT"], estimator=sum, marker="o", color="red")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Amount Spent")
plt.title("Spending Trend Throughout the Day")
plt.xticks(range(0, 24))
plt.show()


# In[14]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["TIME"], y=df["CATEGORY"], size=df["AMOUNT"], sizes=(10, 500), alpha=0.6, palette="muted")
plt.xlabel("Time")
plt.ylabel("Category")
plt.title("Spending Amount by Category Over Time")
plt.xticks(rotation=45)
plt.show()


# In[15]:


import squarify

category_expense = df.groupby("CATEGORY")["AMOUNT"].sum()

plt.figure(figsize=(6, 4))
squarify.plot(sizes=category_expense, label=category_expense.index, alpha=0.8, color=sns.color_palette("pastel"))
plt.title("Category-wise Spending Treemap")
plt.axis("off")  # Remove axes
plt.show()


# In[16]:


df["MONTH"] = df["TIME"].dt.strftime("%b-%Y")

plt.figure(figsize=(8, 4))
sns.barplot(x=df["MONTH"], y=df["AMOUNT"], estimator=sum, palette="coolwarm")
plt.xlabel("Month")
plt.ylabel("Total Amount Spent")
plt.title("Monthly Spending Trend")
plt.xticks(rotation=45)
plt.show()


# In[17]:


plt.figure(figsize=(8, 5))
sns.violinplot(x=df["CATEGORY"], y=df["AMOUNT"], palette="coolwarm")
plt.xlabel("Category")
plt.ylabel("Amount Spent")
plt.title("Spending Variability by Category")
plt.show()


# In[18]:


top_categories = df.groupby("CATEGORY")["AMOUNT"].sum().nlargest(3).index
df_top = df[df["CATEGORY"].isin(top_categories)]

plt.figure(figsize=(8, 5))
sns.lineplot(x=df_top["TIME"], y=df_top["AMOUNT"], hue=df_top["CATEGORY"], marker="o")
plt.fill_between(df_top["TIME"], df_top["AMOUNT"], alpha=0.3)
plt.xlabel("Time")
plt.ylabel("Amount Spent")
plt.title("Top 3 Expense Categories Over Time")
plt.xticks(rotation=45)
plt.show()


# In[19]:


import networkx as nx

# Create graph
G = nx.Graph()

# Add nodes
categories = df["CATEGORY"].unique()
accounts = df["ACCOUNT"].unique()

G.add_nodes_from(categories, color="blue")
G.add_nodes_from(accounts, color="green")

# Add edges (links between categories and payment methods)
for _, row in df.iterrows():
    G.add_edge(row["CATEGORY"], row["ACCOUNT"], weight=row["AMOUNT"])

# Draw the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Layout for better visualization
colors = ["blue" if node in categories else "green" for node in G.nodes]
nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", node_size=1000, font_size=10)
plt.title("Expense Category & Payment Method Relationship")
plt.show()


# In[20]:


df["HOUR"] = df["TIME"].dt.hour
heatmap_data = df.pivot_table(index="HOUR", values="AMOUNT", aggfunc="sum")

plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Hourly Spending Heatmap")
plt.xlabel("Hour")
plt.ylabel("Total Expense")
plt.show()


# In[21]:


# Define essential and non-essential categories
essential_categories = ["Food", "Rent", "Bills", "Transport"]
df["NEED_WANT"] = df["CATEGORY"].apply(lambda x: "Need" if x in essential_categories else "Want")

plt.figure(figsize=(6, 6))
df.groupby("NEED_WANT")["AMOUNT"].sum().plot(kind="pie", autopct="%1.1f%%", cmap="Pastel1", startangle=90)
plt.title("Essential vs. Non-Essential Spending")
plt.ylabel("")  # Hide y-label
plt.show()


# In[22]:


import numpy as np

df["CASH_FLOW"] = np.where(df["TYPE"] == "(-) Expense", -df["AMOUNT"], df["AMOUNT"])

plt.figure(figsize=(8, 5))
sns.lineplot(x=df["TIME"], y=df["CASH_FLOW"].cumsum(), marker="o", color="purple")
plt.xlabel("Time")
plt.ylabel("Cumulative Cash Flow")
plt.title("Cash Flow Over Time (Income vs Expenses)")
plt.xticks(rotation=45)
plt.axhline(y=0, color="gray", linestyle="--", label="Break-even")
plt.legend()
plt.show()


# In[23]:


# Identify top unnecessary spending categories
non_essentials = df[~df["CATEGORY"].isin(["Food", "Rent", "Bills", "Transport"])]
top_unnecessary = non_essentials.groupby("CATEGORY")["AMOUNT"].sum().nlargest(3)

plt.figure(figsize=(7, 4))
sns.barplot(x=top_unnecessary.index, y=top_unnecessary.values, palette="coolwarm")
plt.xlabel("Expense Category")
plt.ylabel("Total Amount Spent")
plt.title("Top 3 Unnecessary Expenses")
plt.show()


# In[24]:


df["MONTH"] = df["TIME"].dt.strftime("%Y-%m")  # Extract Year-Month

plt.figure(figsize=(8, 5))
sns.barplot(x=df["MONTH"], y=df["AMOUNT"], estimator=sum, palette="Blues_r")
plt.xlabel("Month")
plt.ylabel("Total Expense")
plt.title("Monthly Spending Overview")
plt.xticks(rotation=45)
plt.show()


# In[25]:


df["DAY"] = df["TIME"].dt.day
df["MONTH"] = df["TIME"].dt.strftime("%b %Y")

heatmap_data = df.pivot_table(index="DAY", columns="MONTH", values="AMOUNT", aggfunc="sum").fillna(0)

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap="Reds", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Daily Spending Heatmap")
plt.xlabel("Month")
plt.ylabel("Day of Month")
plt.show()


# In[26]:


import plotly.express as px

fig = px.sunburst(df, path=["CATEGORY", "ACCOUNT"], values="AMOUNT", title="Expense Distribution")
fig.show()


# In[27]:


df["MONTH"] = df["TIME"].dt.strftime("%Y-%m")

monthly_summary = df.groupby(["MONTH", "TYPE"])["AMOUNT"].sum().reset_index()

fig = px.bar(monthly_summary, x="MONTH", y="AMOUNT", color="TYPE", barmode="group",
             title="Monthly Income vs Expenses", color_discrete_map={"(+) Income": "green", "(-) Expense": "red"})
fig.show()


# In[28]:


df["CASH_FLOW"] = df["AMOUNT"] * np.where(df["TYPE"] == "(-) Expense", -1, 1)
df["CUMULATIVE_SAVINGS"] = df["CASH_FLOW"].cumsum()

fig = px.line(df, x="TIME", y="CUMULATIVE_SAVINGS", title="Cumulative Savings Over Time", markers=True)
fig.show()


# In[29]:


df["HOUR"] = df["TIME"].dt.hour

fig = px.histogram(df[df["TYPE"] == "(-) Expense"], x="HOUR", nbins=24, title="Spending by Hour of Day",
                   labels={"HOUR": "Hour of Day"}, color_discrete_sequence=["red"])
fig.show()


# In[30]:


fig = px.treemap(df[df["TYPE"] == "(-) Expense"], path=["CATEGORY"], values="AMOUNT",
                 title="Top Spending Categories", color="AMOUNT", color_continuous_scale="reds")
fig.show()


# In[31]:


df["EXPENSES_ONLY"] = np.where(df["TYPE"] == "(-) Expense", df["AMOUNT"], 0)
df["ROLLING_AVG"] = df["EXPENSES_ONLY"].rolling(window=7, min_periods=1).mean()

fig = px.line(df, x="TIME", y="ROLLING_AVG", title="7-Day Moving Average of Expenses", markers=True)
fig.show()


# In[32]:


df["WEEKDAY"] = df["TIME"].dt.day_name()
weekly_expense = df[df["TYPE"] == "(-) Expense"].groupby("WEEKDAY")["AMOUNT"].sum().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

fig = px.line_polar(r=weekly_expense.values, theta=weekly_expense.index, line_close=True, title="Weekly Spending Pattern")
fig.show()


# In[2]:


get_ipython().system('jupyter nbconvert --to script Expense Analysis 5th Sem.ipynb')


# In[ ]:




