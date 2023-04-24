import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time
import urllib.request #
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
    driver = Chrome('/home/jaemincho/works/stock/chromedriver', chrome_options=options)
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
    
    print("Loading Compelte!!!!")
    driver.quit()
    return result_df


def cal_price(df, rolling_val=5):
    arr = df.values
    arr2 = []
    for raw in arr:
        tmp_arr = []
        for col in raw:
            tmp_val = col.replace(",","")
            tmp_val = tmp_val.replace("N/A","0")
            tmp_arr.append(tmp_val)
        arr2.append(tmp_arr)

    arr2 = np.array(arr2, dtype=np.float32)
    df = pd.DataFrame(arr2, columns = df.columns.values, index = df.index.values)
    # Cal with bps roe
    roe_series = df['ROE(%)']
    bps_series = df['BPS(원)']
    price_bps_roe = bps_series * (roe_series/10)

    # Cal with eps per
    eps_series = df['EPS(원)']
    per_series = df['PER(배)'].rolling(rolling_val).mean()
    price_eps_per = eps_series*per_series

    # Cal with pbr bps
    pbr_series = df['PBR(배)'].rolling(rolling_val).mean()
    bps_series = df['BPS(원)']
    price_pbr_bps = pbr_series*bps_series

    price_bps_roe = price_bps_roe.values.reshape(-1,1)
    price_eps_per = price_eps_per.values.reshape(-1,1)
    price_pbr_bps = price_pbr_bps.values.reshape(-1,1)

    tot_arr = np.concatenate((price_bps_roe, price_eps_per, price_pbr_bps), axis=1)
    result = pd.DataFrame(tot_arr, index=df.index.values, columns = ['BPS*(ROE/10)', 'EPS*PER(mean)', 'BPS*PBR(mean)'])
    return  result


def return_theme_list():
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    # Input url =====
    first_page_url = 'https://finance.naver.com/sise/theme.nhn?&page=1'
    driver = Chrome('/home/jaemincho/works/stock/chromedriver', chrome_options=options)
    driver.implicitly_wait(3)
    driver.get(first_page_url)

    for_last_page = driver.page_source
    for_last_page1 = BeautifulSoup(for_last_page,'html.parser')
    pgrr = for_last_page1.find('td', {'class':'pgRR'})
    last_page_num = pgrr.find('a').get('href').split("page=")[-1]
    driver.quit()

    title_list={}
    for i in range(1, int(last_page_num)+1):
        url = 'https://finance.naver.com/sise/theme.nhn?&page='+str(i)
        driver = Chrome('/home/jaemincho/works/stock/chromedriver', chrome_options=options)
        driver.implicitly_wait(3)
        driver.get(url)

        html0 = driver.page_source
        html1 = BeautifulSoup(html0,'html.parser')

        div_wrap = html1.find('div', {'id':'wrap'})
        content_area = div_wrap.find('div', {'id':'contentarea'})
        content_left = content_area.find('div', {'id':'contentarea_left'})
        type_1_theme = content_left.find('table', {'class':'type_1 theme'})
        tbody_s = type_1_theme.find('tbody')

        titles = tbody_s.findAll('td', {'class':'col_type1'})
        
        for title in titles:
            title_list[title.text] = title.find('a').get('href')

        driver.quit()

    return title_list


def return_table_using_theme(theme_url):
    options = ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")

    new_url = 'https://finance.naver.com' + theme_url
    driver = Chrome('/home/jaemincho/works/stock/chromedriver', chrome_options=options)
    driver.implicitly_wait(3)
    print(new_url)
    driver.get(new_url)
    html0 = driver.page_source
    html1 = BeautifulSoup(html0,'html.parser')

    div_wrap = html1.find('div', {'id':'wrap'})
    newarea = div_wrap.find('div', {'id':'newarea'})
    contentarea = newarea.find('div', {'id':'contentarea'})

    type_5 = contentarea.find('table', {'summary':'업종별 시세 리스트'})
    tbody = type_5.find('tbody')
    
    names = tbody.findAll('td', {'class':'name'})
    fin_result={}
    for name in names:
        code_ = name.find('div',{'class':'name_area'}).find('a').get('href').split('code=')[-1]
        code_name = name.find('div',{'class':'name_area'}).find('a').text
        table_df_year = return_balancetable(corp_code=code_, year_quarter='year')
        table_df_quarter = return_balancetable(corp_code=code_, year_quarter='quarter')
        price_table = cal_price(df=table_df_year, rolling_val=5)
        
        fin_result[code_] = [code_name, table_df_year, table_df_quarter, price_table]

    return fin_result

def vis_table(df):
    x_data = df.index.values

    arr = df.values
    arr2 = []
    for raw in arr:
        tmp_arr = []
        for col in raw:
            tmp_val = col.replace(",","")
            tmp_val = tmp_val.replace("N/A","0")
            tmp_arr.append(tmp_val)
        arr2.append(tmp_arr)

    arr2 = np.array(arr2, dtype=np.float32)
    trans_df = pd.DataFrame(arr2, columns = df.columns.values, index = df.index.values)

    sale = trans_df['매출액'].values
    profit = trans_df['영업이익'].values
    net_profit = trans_df['당기순이익'].values

    asset = trans_df['자산총계'].values
    debit = trans_df['부채총계'].values
    equity = trans_df['자본총계'].values

    profit_rate = trans_df['영업이익률'].values
    net_profit_rate = trans_df['순이익률'].values

    roe = trans_df['ROE(%)'].values
    roa = trans_df['ROA(%)'].values

    eps = trans_df['EPS(원)'].values
    per = trans_df['PER(배)'].values
    bps = trans_df['BPS(원)'].values
    pbr = trans_df['PBR(배)'].values

    fig, (ax, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, figsize=(10,15), sharex=True)

    ax.plot(x_data, sale, color='r', label='sale')
    ax.plot(x_data, profit, color='b', label='profit')
    ax.plot(x_data, net_profit, color='g', label='net profit')
    for ii in range(len(x_data)):
        ax.annotate(str(sale[ii]), xy=(x_data[ii], sale[ii]))
    for ii in range(len(x_data)):
        ax.annotate(str(profit[ii]), xy=(x_data[ii], profit[ii]))
    for ii in range(len(x_data)):
        ax.annotate(str(net_profit[ii]), xy=(x_data[ii], net_profit[ii]))
    ax.legend()

    ax2.plot(x_data, asset, color='r', label='asset')
    ax2.plot(x_data, debit, color='b', label='debit')
    ax2.plot(x_data, equity, color='g', label='equity')
    for ii in range(len(x_data)):
        ax2.annotate(str(asset[ii]), xy=(x_data[ii], asset[ii]))
    for ii in range(len(x_data)):
        ax2.annotate(str(debit[ii]), xy=(x_data[ii], debit[ii]))
    for ii in range(len(x_data)):
        ax2.annotate(str(equity[ii]), xy=(x_data[ii], equity[ii]))
    ax2.legend()

    ax3.plot(x_data, profit_rate, color='r', label='profit rate')
    ax3.plot(x_data, net_profit_rate, color='b', label='net profit rate')
    ax3.plot(x_data, roe, color='g', label='ROE(%)')
    ax3.plot(x_data, roa, color='black', label='ROA(%)')
    for ii in range(len(x_data)):
        ax3.annotate(str(profit_rate[ii]), xy=(x_data[ii], profit_rate[ii]))
    for ii in range(len(x_data)):
        ax3.annotate(str(net_profit_rate[ii]), xy=(x_data[ii], net_profit_rate[ii]))
    for ii in range(len(x_data)):
        ax3.annotate(str(roe[ii]), xy=(x_data[ii], roe[ii]))
    for ii in range(len(x_data)):
        ax3.annotate(str(roa[ii]), xy=(x_data[ii], roa[ii]))
    ax3.legend()

    ax4.plot(x_data, eps, color='r', label='EPS(won)')
    ax4.plot(x_data, bps, color='b', label='BPS(won)')
    for ii in range(len(x_data)):
        ax4.annotate(str(eps[ii]), xy=(x_data[ii], eps[ii]))
    for ii in range(len(x_data)):
        ax4.annotate(str(bps[ii]), xy=(x_data[ii], bps[ii]))
    ax4.legend()

    ax5.plot(x_data, per, color='r', label='PER(x)')
    ax5.plot(x_data, pbr, color='b', label='PBR(x)')
    for ii in range(len(x_data)):
        ax5.annotate(str(per[ii]), xy=(x_data[ii], per[ii]))
    for ii in range(len(x_data)):
        ax5.annotate(str(pbr[ii]), xy=(x_data[ii], pbr[ii]))
    ax5.legend()

    plt.show()
