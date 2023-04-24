# -*- coding: utf-8 -*-


import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time
import urllib.request 
from selenium.webdriver import Chrome, ChromeOptions
import json
import re     
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt


def return_balancetable(corp_code, year_quarter):
    # Option =====
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    
    # Input url =====
    url = 'https://finance.naver.com/item/coinfo.nhn?code='+corp_code
    driver = Chrome('C:\\Users\\sdrft\\source\\repos\\My_stock_project\\chromedriver', chrome_options=options)
    driver.implicitly_wait(3)
    driver.get(url)
    
    
    # Switch to frame =====
    driver.switch_to_frame(driver.find_element_by_id('coinfo_cp'))
    
    # Make Year or Quarter index ======
    if year_quarter=="year":
        year_quarter_idx = '3'
    elif year_quarter=="quarter":
        year_quarter_idx = '4'
        
    # Enter Year or Quarter window
    driver.find_elements_by_xpath('//*[@class="schtab"][1]/tbody/tr/td['+year_quarter_idx+']')[0].click()
    
    # Get Soup from html
    html0 = driver.page_source
    html1 = BeautifulSoup(html0,'html.parser')
    
    # Get Corp.title
    title0 = html1.find('head').find('title').text
    title0.split('-')[-1]

    # Get unit
    unit_str = html1.find('table',{'class':'schbox', 'id':'finSummary'}).find('span',{'id':'unit_text'}).text
    unit_str_s = unit_str.split(" ")
    if unit_str_s[3] == "억원,":
        amount_unit = 100000000
    elif unit_str_s[3] == "백만,":
        amount_unit = 1000000
    else:
        amount_unit = 1

    # Get date data
    html22 = html1.find('table',{'class':'gHead01 all-width','summary':'주요재무정보를 제공합니다.'})
    thead0 = html22.find('thead')
    tr0 = thead0.find_all('tr')[1]
    th0 = tr0.find_all('th')
    # Extract Date
    date = []
    for i in range(len(th0)):
        date.append(''.join(re.findall('[0-9/]',th0[i].text)))
        
    # Get table data
    tbody0 = html22.find('tbody')
    tr0 = tbody0.find_all('tr')

    # collect column name data
    col = []
    for i in range(len(tr0)):
        if '\xa0' in tr0[i].find('th').text:
            tx = re.sub('\xa0','',tr0[i].find('th').text)
        else:
            tx = tr0[i].find('th').text
        col.append(tx)
    
    # Get Table data
    td = []
    for i in range(len(tr0)):
        td0 = tr0[i].find_all('td')
        td1 = []
        for j in range(len(td0)):
            if td0[j].text == '':
                td1.append('0')
            else:
                td1.append(td0[j].text)
        td.append(td1)
    td2 = list(map(list,zip(*td)))
    
    result_df = pd.DataFrame(td2, columns = col, index = date)

    # Equity replace
    val_list=[]
    for val in result_df['자본총계(지배)'].values:
        val_list.append(float(val.replace(",",""))*amount_unit)
    result_df['자본총계(지배)'] = val_list
    
    driver.quit()
    return result_df



def s_rim_value(stock_code, chart_type, last_quarter, market_growth = 0.0792):

    df = return_balancetable(corp_code=stock_code, year_quarter=chart_type)
    df_cut = df.iloc[df.index<=last_quarter]
    try:
        stock_equity = df_cut['자본총계(지배)'].values[-1]
        roes = df_cut['ROE(%)'].values[-3:]
        new_roes = []
        for r in roes:
            new_roes.append(float(r.replace(",","")))
        num_of_stock = float(df_cut['발행주식수(보통주)'].values[-1].replace(",",""))
        roe_means = (new_roes[0]*1 + new_roes[1]*2 + new_roes[2]*3) / (6*100)

        excess_return = stock_equity * (roe_means - market_growth)

        srim_val_100 = ( stock_equity + (excess_return/market_growth) ) / num_of_stock
        srim_val_90 = ( stock_equity + (excess_return*(0.9/(1+market_growth-0.9))) ) / num_of_stock
        srim_val_80 = ( stock_equity + (excess_return*(0.8/(1+market_growth-0.8))) ) / num_of_stock
        srim_val_70 = ( stock_equity + (excess_return*(0.7/(1+market_growth-0.7))) ) / num_of_stock

        if (roe_means-market_growth) < 0:
            result_str = "100% : {} / Low ROE".format(srim_val_100)
        else:
            result_str = "70% : {} / 80% : {} / 90% : {} / 100% : {}".format(srim_val_70, srim_val_80, srim_val_90, srim_val_100)
        return result_str
    except:
        return -1
