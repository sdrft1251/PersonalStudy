import win32com.client
import win32com


def down(msges, keyword_list, save_dir):
    save_num=0
    for ms_ in msges:
        if ms_.Attachments:
            for att in ms_.Attachments:
                for keyword in keyword_list:
                    if keyword in att.FileName:
                        if (att.FileName.split(".")[-1]=="pptx") | (att.FileName.split(".")[-1]=="ppt"):
                            att.SaveAsFile(r""+save_dir+"/" + att.FileName)
                            save_num+=1
    return save_num

def massage_process(parent, start_date, end_date, keyword_list, save_dir):
    return_save_end=0
    if hasattr(parent, 'Items'):
        messages = parent.Items
        sFilter = "[ReceivedTime]>='{0}".format((start_date).strftime('%m/%d/%Y %H:%M %p')) + "'"
        sFilter2 = "[ReceivedTime]<='{0}".format((end_date).strftime('%m/%d/%Y %H:%M %p')) + "'"
        messages = messages.Restrict(sFilter)
        messages = messages.Restrict(sFilter2)
        return_save_end += down(msges=messages, keyword_list=keyword_list, save_dir=save_dir)
    if hasattr(parent, 'Folders'):
        for folder in parent.Folders:
            return_save_end += massage_process(parent=folder, start_date=start_date, end_date=end_date, keyword_list=keyword_list, save_dir=save_dir)
    return return_save_end

def start_func(keyword_list, start_date, end_date, save_dir):

    outlook = win32com.client.Dispatch('Outlook.Application')
    inbox = outlook.GetNamespace('MAPI').GetDefaultFolder(6)

    return massage_process(parent=inbox, start_date=start_date, end_date=end_date, keyword_list=keyword_list, save_dir=save_dir)





