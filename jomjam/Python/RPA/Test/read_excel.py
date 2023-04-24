import win32com.client
import pandas as pd

excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = True

file_path = "C:\\Users\\7039966\\Desktop\\excel_test.xlsx"
wb = excel.Workbooks.Open(file_path, ReadOnly=False)

print(wb.Sheets.Count)

for sheet in wb.Sheets:
    print(sheet.Cells(2,9).Value)
    print(sheet.Columns.Count)
    print(sheet.name)
    list_for_df = []
    #for ir in range(sheet.Rows.Count):
    for ir in range(20):
        column_dump = []
        #for ic in range(sheet.Columns.Count):
        for ic in range(20):
            column_dump.append(sheet.Cells(ir+1, ic+1).Value)
        list_for_df.append(column_dump)

    print(list_for_df)

