# Import das libs
import json
import os 
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# CRIANDO YAHOO FINANCE TOOL
def fetch_stock_prince(ticket):
  stock = yf.download(ticket, start="2023-01-01", end="2024-08-08")
  return stock

yahoo_finance_tool = Tool(
  name = "Yahoo Finance Tool",
  description = "Fetches stocks prices for {ticket} from the last year about a specific stock from Yahoo Finance API",
  func = lambda ticket: fetch_stock_prince(ticket)
)

# IMPORTANDO OPENAI LLM - GPT

llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent (
  role = "Senior stock price Analyst",
  goal = "Finde the {ticket} stock price and analyses trends",
  backstory = "You are a hight in analyzing the price of an specific stock and make predictions about its future price.",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True,
  tools = [yahoo_finance_tool]
)


# In[6]:


getStockPrice = Task (
  description = "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways.",
  expected_output = "Specify the current trend stock price - upda, down or sideways. eg. stock = 'AAPL', price UP.",
  agent = stockPriceAnalyst
)


# In[7]:


# Importando a TOOL de SEARCH 
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[8]:


newsAnalyst = Agent (
  role = "Stock News Analyst ",
  goal = "Create a short summary of the market news related to the stock {ticket} company. Specify the current tren - up, down or sideways with the news context. For eacth request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
  backstory = """You are highly experience in analyzing the market trends and news and have tracket asses for more than 30 years.
  You're also master level analyst in the tradicional markets and have deep understanding of human psychology.
  You understand news, theirs tittles and information, but you look at those with a helth dose of skepticism.
  You consider also the source of the news article.
  """,
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  allow_delegation = True,
  tools = [search_tool]
)


# In[9]:


get_news = Task( 
  description = "Take the stock and always include BTC to it (if not request). Use the search tool to search each one individually. Compose the results into a helpful report.",
  expected_output = """A summary of the overall market and one setence summary for each request asset. Include a fear/greed socre for each asset based on the news. 
  Use the format: 
  <STOCK ASSET>
  <SUMMARY BASE ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent = newsAnalyst
)


# In[10]:


stockAnalystWriter = Agent(
  role = "Senior Stock Analyst Writer",
  goal = "Analyze the trends and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.",
  backstory = "You are widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences. You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True
)


# In[11]:


writeAnalyses = Task(
  description = "Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company",
  expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readble manner. It should contain: 
  
  - 3 bullets executive summary 
  - Introduction - set the overall pictures and spike up the interest
  - main part provides the meat of the analysis including the news summary and fead/greed scores\n",
  - summary - key facts and concrete future trend prediction - up, down or sideways

  """,
  agent = stockAnalystWriter,
  context = [getStockPrice, get_news]
)


# In[12]:


crew = Crew(
  agent = [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
  tasks = [getStockPrice, get_news, writeAnalyses],
  verbose = True,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter=15
)

# results = crew.kickoff(inputs={'ticket': 'AAPL'})

with st.sidebar:
  st.header('Enter the Stock to Research: ')

  with st.form(key='research_form'):
    topic = st.text_input("Select the ticket")
    submit_buttom = st.form_submit_button(label="Run Research")

if submit_buttom:
  if not topic:
    st.error("Please fill the ticket field") 
  else: 
    results = crew.kickoff(inputs={'ticket': topic})

    st.subheader("Results of your research: ")
    st.write(results["final_output"])