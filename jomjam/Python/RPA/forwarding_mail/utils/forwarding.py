import win32com.client
import win32com
from utils import mail_contents


class Forwarding:

    def __init__(self, last_end_datetime, now_datetime, mail_from, send_to, done_list):
        self.last_end_datetime = last_end_datetime
        self.now_datetime = now_datetime
        self.mail_from = mail_from
        self.send_to = send_to
        self.done_list = done_list
        self.success_num = 0
        self.success_list = []
        self.error_list = []
        self.error_with_subject = {}

    def forward_mail(self, msges):
        for ms_ in msges:
            if ("운송의뢰서" in str(ms_.Subject)) and (self.mail_from in str(ms_.Sender)):
                unique_index = ms_.Subject.split("]")[0].split("[")[-1].strip()
                if unique_index not in self.success_list and unique_index not in self.done_list:
                    for_forward_msg = ms_.Forward()
                    error_idx_list = []
                    try:
                        new_msg, new_sub, error_idx_list = mail_contents.return_new_msg(old_msg=ms_.HTMLBody)
                        for_forward_msg.To = self.send_to
                        for_forward_msg.Subject = new_sub
                        for_forward_msg.HTMLBody = new_msg + for_forward_msg.HTMLBody
                        for_forward_msg.Send()
                    except Exception:
                        print("Forwarding fail...")

                    if len(error_idx_list) >= 1:
                        self.error_with_subject[unique_index] = error_idx_list

                    self.success_num += 1
                    self.success_list.append(unique_index)

                    if ms_.Unread:
                        ms_.Unread = False

    def massage_process(self, parent):

        if hasattr(parent, 'Items'):
            messages = parent.Items
            s_filter = "[ReceivedTime]>='{0}".format(self.last_end_datetime.strftime('%m/%d/%Y %H:%M %p')) + "'"
            s_filter2 = "[ReceivedTime]<'{0}".format(self.now_datetime.strftime('%m/%d/%Y %H:%M %p')) + "'"
            messages = messages.Restrict(s_filter)
            messages = messages.Restrict(s_filter2)
            self.forward_mail(msges=messages)

        if hasattr(parent, 'Folders'):
            for folder in parent.Folders:
                self.massage_process(parent=folder)

    def func_start(self):

        outlook = win32com.client.Dispatch('Outlook.Application')
        inbox = outlook.GetNamespace('MAPI').GetDefaultFolder(6)
        self.massage_process(parent=inbox)

    def return_result(self):
        return self.success_num, self.success_list, self.error_list, self.error_with_subject
