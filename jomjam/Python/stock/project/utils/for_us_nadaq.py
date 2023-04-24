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

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from IPython.display import display, HTML



def return_nasdaq_list(page_num=5):
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    component_url = 'https://kr.investing.com/indices/nasdaq-composite-components/'

    corp_url_list=[]
    for ii in range(1,page_num+1):
        url = component_url + str(ii)
        driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
        driver.implicitly_wait(3)
        driver.get(url)

        html0 = driver.page_source
        html1 = BeautifulSoup(html0,'html.parser')

        body = html1.find('body', {'class':'takeover dfpTakeovers '})
        wrapper = body.find('div',{'class':'wrapper'})

        left_section = wrapper.find('section',{'id':'leftColumn'})
        inner_content = left_section.find('div',{'id':'marketInnerContent'})

        table = inner_content.find('table',{'id':'cr1'})
        tbody = table.find('tbody')

        trs = tbody.find_all('tr')
        for tr in trs:
            bold_name = tr.find('td',{'class':'bold left noWrap elp plusIconTd'})
            tmp_url = bold_name.find('a').get('href')
            corp_url_list.append(tmp_url)
        driver.quit()

    print("Getting URL is finished")

    code_list = []
    tot_code_num=len(corp_url_list)
    print("Total code numer is {}".format(tot_code_num))
    tmp_idx=1
    for url_ in corp_url_list:
        url = 'https://kr.investing.com' + url_
        driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
        driver.implicitly_wait(3)
        driver.get(url)

        html0 = driver.page_source
        html1 = BeautifulSoup(html0,'html.parser')

        body = html1.find('body', {'class':'takeover dfpTakeovers '})
        wrapper = body.find('div',{'class':'wrapper'})

        left_section = wrapper.find('section',{'id':'leftColumn'})
        instrumentHead = left_section.find('div',{'class':'instrumentHead'})

        name = instrumentHead.find('h1',{'class':'float_lang_base_1 relativeAttr'}).text
        code = name.split('(')[-1].split(')')[0]
        code_list.append(code)
        print("Code is {} & complete rate is {}".format(code, float(tmp_idx/tot_code_num)))
        tmp_idx+=1
        driver.quit()
    
    print("Getting Code Complete!")

    return code_list



def return_fin_info_table(code, table_type, year=True):
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    if table_type == 'ic':
        url = 'https://finance.yahoo.com/quote/'+code+'/financials?p='+code
    elif table_type == 'bs':
        url = 'https://finance.yahoo.com/quote/'+code+'/balance-sheet?p='+code
    elif table_type == 'cf':
        url = 'https://finance.yahoo.com/quote/'+code+'/cash-flow?p='+code
    else:
        print('wrong option')
        return 1

    print(url)
    driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
    driver.implicitly_wait(3)
    driver.get(url)

    if year:
        print("Finding year")
    else:
        button = driver.find_elements_by_xpath('//div[@class="Fl(end) smartphone_Fl(n) IbBox smartphone_My(10px) smartphone_D(b)"]/button[@class="P(0px) M(0px) C($linkColor) Bd(0px) O(n)"]')[0]
        button.click()
        try:
            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//div[@class="Fz(s) Fw(500) D(ib) H(18px) C($primaryColor):h C($primaryColor)"]/span[contains(text(),"Quarterly")]'))
            )
        finally:
            print("Finding quarter")

    html0 = driver.page_source
    html1 = BeautifulSoup(html0,'html.parser')

    body = html1.find('body')
    app_ = body.find('div',{'id':'Main', 'role':'content'})
    fin_proxy = app_.find('div',{'id':'Col1-1-Financials-Proxy'})
    data_sec = fin_proxy.find('section',{'data-test':'qsp-financial'})
    data_sec2 = data_sec.find('div',{'class':'Pos(r)'})
    data_sec3 = data_sec2.find('div',{'class':'W(100%) Whs(nw) Ovx(a) BdT Bdtc($seperatorColor)'})
    data_sec4 = data_sec3.find('div',{'class':'M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)'})
    #Column name section
    col_list=[]
    col_parts = data_sec4.find('div',{'class':'D(tbhg)'}).find_all('span')
    for col_part in col_parts:
        col_list.append(col_part.text)
    col_list=col_list[1:]
    #Row section
    row_parts = data_sec4.find('div',{'class':'D(tbrg)'}).find_all('div',{'data-test':'fin-row'})
    index_list = []
    content_data_list = []
    for row_part in row_parts:
        index_list.append(row_part.find('span',{'class','Va(m)'}).text)
        values_ = row_part.find_all('div',{'data-test':'fin-col'})
        tmp_val_list=[]
        for value_ in values_:
            ttt = value_.text
            ttt = ttt.replace(',','')
            if (len(ttt)==1)&(ttt=='-'):
                ttt=0
            tmp_val_list.append(ttt)
        content_data_list.append(tmp_val_list)
    content_data_arr = np.array(content_data_list, dtype=np.float32)

    df = pd.DataFrame(content_data_arr, columns=col_list, index=index_list)
    driver.quit()
    return df



def print_all_table_with_plot(code_input, year_type=True):
    #IC
    ic_df = return_fin_info_table(code=code_input, table_type='ic', year=year_type)
    col_list = ic_df.columns.values[::-1]
    tot_revenue = ic_df.loc['Total Revenue'].values.reshape(-1)[::-1]
    gross_profit = ic_df.loc['Gross Profit'].values.reshape(-1)[::-1]
    operating_income = ic_df.loc['Operating Income'].values.reshape(-1)[::-1]

    ebit = ic_df.loc['EBIT'].values.reshape(-1)[::-1]
    ebitda = ic_df.loc['EBITDA'].values.reshape(-1)[::-1]

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,6), sharex=True)

    ax1.plot(col_list, tot_revenue, color='r', label='Total Revenue')
    ax1.plot(col_list, gross_profit, color='b', label='Gross Profit')
    ax1.plot(col_list, operating_income, color='g', label='Operating Income')
    for ii in range(len(col_list)):
        ax1.annotate(str(tot_revenue[ii]), xy=(col_list[ii], tot_revenue[ii]))
    for ii in range(len(col_list)):
        ax1.annotate(str(gross_profit[ii]), xy=(col_list[ii], gross_profit[ii]))
    for ii in range(len(col_list)):
        ax1.annotate(str(operating_income[ii]), xy=(col_list[ii], operating_income[ii]))
    ax1.legend()

    ax2.plot(col_list, ebit, color='r', label='EBIT')
    ax2.plot(col_list, ebitda, color='b', label='EBITDA')
    for ii in range(len(col_list)):
        ax2.annotate(str(ebit[ii]), xy=(col_list[ii], ebit[ii]))
    for ii in range(len(col_list)):
        ax2.annotate(str(ebitda[ii]), xy=(col_list[ii], ebitda[ii]))
    ax2.legend()

    #BS
    bs_df = return_fin_info_table(code=code_input, table_type='bs', year=year_type)
    col_list2 = bs_df.columns.values[::-1]
    asset = bs_df.loc['Total Assets'].values.reshape(-1)[::-1]
    debt = bs_df.loc['Total Liabilities Net Minority Interest'].values.reshape(-1)[::-1]
    equity = bs_df.loc['Total Equity Gross Minority Interest'].values.reshape(-1)[::-1]
    fig2, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(10,3))
    ax3.plot(col_list2, asset, color='r', label='asset')
    ax3.plot(col_list2, debt, color='b', label='debt')
    ax3.plot(col_list2, equity, color='g', label='equity')
    for ii in range(len(col_list2)):
        ax3.annotate(str(asset[ii]), xy=(col_list2[ii], asset[ii]))
    for ii in range(len(col_list2)):
        ax3.annotate(str(debt[ii]), xy=(col_list2[ii], debt[ii]))
    for ii in range(len(col_list2)):
        ax3.annotate(str(equity[ii]), xy=(col_list2[ii], equity[ii]))
    ax3.legend()

    #CF
    cf_df = return_fin_info_table(code=code_input, table_type='cf', year=year_type)
    col_list3 = cf_df.columns.values[::-1]
    operating_cash_flow = cf_df.loc['Operating Cash Flow'].values.reshape(-1)[::-1]
    investing_cash_flow = cf_df.loc['Investing Cash Flow'].values.reshape(-1)[::-1]
    end_cash_position = cf_df.loc['End Cash Position'].values.reshape(-1)[::-1]
    fig3, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(10,3))
    ax4.plot(col_list3, operating_cash_flow, color='r', label='Operating Cash Flow')
    ax4.plot(col_list3, investing_cash_flow, color='b', label='Investing Cash Flow')
    ax4.plot(col_list3, end_cash_position, color='g', label='End Cash Position')
    for ii in range(len(col_list3)):
        ax4.annotate(str(operating_cash_flow[ii]), xy=(col_list3[ii], operating_cash_flow[ii]))
    for ii in range(len(col_list3)):
        ax4.annotate(str(investing_cash_flow[ii]), xy=(col_list3[ii], investing_cash_flow[ii]))
    for ii in range(len(col_list3)):
        ax4.annotate(str(end_cash_position[ii]), xy=(col_list3[ii], end_cash_position[ii]))
    ax4.legend()

    plt.show()


def print_all_table_with_plot2(code_input, year_type=True):
    ic_df = return_fin_info_table(code=code_input, table_type='ic', year=year_type)
    bs_df = return_fin_info_table(code=code_input, table_type='bs', year=year_type)
    cf_df = return_fin_info_table(code=code_input, table_type='cf', year=year_type)

    ic_df_length = ic_df.shape[0]
    ic_df_tot_idx = ic_df_length//5
    for i1 in range(ic_df_tot_idx):
        tmp_df = ic_df.iloc[i1*5:(i1+1)*5]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Income Statement")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()
    if (ic_df_length%5) != 0:
        tmp_df = ic_df.iloc[ic_df_tot_idx*5:]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Income Statement")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()

    bs_df_length = bs_df.shape[0]
    bs_df_tot_idx = bs_df_length//5
    for i1 in range(bs_df_tot_idx):
        tmp_df = bs_df.iloc[i1*5:(i1+1)*5]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Balance Sheet")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()
    if (bs_df_tot_idx%5) != 0:
        tmp_df = bs_df.iloc[bs_df_tot_idx*5:]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Balance Sheet")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()

    cf_df_length = cf_df.shape[0]
    cf_df_tot_idx = cf_df_length//5
    for i1 in range(cf_df_tot_idx):
        tmp_df = cf_df.iloc[i1*5:(i1+1)*5]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Cash Flow")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()
    if (cf_df_tot_idx%5) != 0:
        tmp_df = cf_df.iloc[cf_df_tot_idx*5:]
        tmp_ax = tmp_df.T.plot(figsize=(20,5))
        tmp_ax.set_title("Cash Flow")
        tmp_ax.set_xlabel("Time")
        tmp_ax.set_ylabel("Data")
        plt.show()


    
def return_summary(code_input):
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    url = "https://finance.yahoo.com/quote/"+code_input+"?p="+code_input
    driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
    driver.implicitly_wait(3)
    driver.get(url)

    html0 = driver.page_source
    html1 = BeautifulSoup(html0,'html.parser')

    content_body = html1.find('div',{'id':'quote-summary', 'data-test':'quote-summary-stats'})
    now_volume = content_body.find('td', {'data-test':'TD_VOLUME-value'}).find('span').text.replace(',','').replace('N/A','-99')
    avg_volume = content_body.find('td', {'data-test':'AVERAGE_VOLUME_3MONTH-value'}).find('span').text.replace(',','').replace('N/A','-99')
    
    eps_ttm = content_body.find('td', {'data-test':'EPS_RATIO-value'}).find('span').text.replace(',','').replace('N/A','-99')
    per = content_body.find('td', {'data-test':'PE_RATIO-value'}).find('span').text.replace(',','').replace('N/A','-99')
    beta = content_body.find('td', {'data-test':'BETA_5Y-value'}).find('span').text.replace(',','').replace('N/A','-99')
    
    val_list = [eps_ttm, per, beta, now_volume, avg_volume]
    val_arr = np.array(val_list, dtype=np.float32)
    return val_arr

def print_plot_one_by_one(page_num=5, year_idx=True):
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    component_url = 'https://kr.investing.com/indices/nasdaq-composite-components/'

    corp_url_list=[]
    for ii in range(1,page_num+1):
        url = component_url + str(ii)
        driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
        driver.implicitly_wait(3)
        driver.get(url)

        html0 = driver.page_source
        html1 = BeautifulSoup(html0,'html.parser')

        body = html1.find('body', {'class':'takeover dfpTakeovers '})
        wrapper = body.find('div',{'class':'wrapper'})

        left_section = wrapper.find('section',{'id':'leftColumn'})
        inner_content = left_section.find('div',{'id':'marketInnerContent'})

        table = inner_content.find('table',{'id':'cr1'})
        tbody = table.find('tbody')

        trs = tbody.find_all('tr')
        for tr in trs:
            bold_name = tr.find('td',{'class':'bold left noWrap elp plusIconTd'})
            tmp_url = bold_name.find('a').get('href')
            corp_url_list.append(tmp_url)
        driver.quit()

    print("Getting URL is finished")

    code_list = []
    tot_code_num=len(corp_url_list)
    print("Total code numer is {}".format(tot_code_num))
    tmp_idx=1
    for url_ in corp_url_list:
        url = 'https://kr.investing.com' + url_
        driver = Chrome('/Users/jaemin/Desktop/crawler/python/chromedriver', chrome_options=options)
        driver.implicitly_wait(3)
        driver.get(url)

        html0 = driver.page_source
        html1 = BeautifulSoup(html0,'html.parser')

        body = html1.find('body', {'class':'takeover dfpTakeovers '})
        wrapper = body.find('div',{'class':'wrapper'})

        left_section = wrapper.find('section',{'id':'leftColumn'})
        instrumentHead = left_section.find('div',{'class':'instrumentHead'})

        name = instrumentHead.find('h1',{'class':'float_lang_base_1 relativeAttr'}).text
        code = name.split('(')[-1].split(')')[0]
        code_list.append(code)
        print("Code is {} & complete rate is {}".format(code, float(tmp_idx/tot_code_num)))
        tmp_idx+=1
        driver.quit()
        try:
            summary_arr = return_summary(code_input=code)
            print("EPS = {} , PER = {} , BETA (5 years_value) = {} , Now Volume = {} , Avg. Volume (3 month) = {}".format(summary_arr[0], summary_arr[1], summary_arr[2], summary_arr[3], summary_arr[4]))
        except:
            print('Printing summary fail')
        try:
            print_all_table_with_plot2(code_input=code, year_type=year_idx)
        except:
            print("Printing plot fail")
    
    print("Getting Code Complete!")

    return code_list
