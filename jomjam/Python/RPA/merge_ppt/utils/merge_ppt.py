import os
import win32com.client

def merge_data(file_path, save_path):
    Application = win32com.client.Dispatch("PowerPoint.Application")
    Application.Visible = True


    file_list = os.listdir(file_path)
    ppt_list = [file for file in file_list if file.endswith(".ppt") or file.endswith(".pptx")]
    file_path = file_path.replace("/","\\") + "\\"
    merge_num = 0
    for i, ppt_name in enumerate(ppt_list):
        try:
            if i == 0:
                mainPresentation = Application.Presentations.Open(file_path + ppt_name, ReadOnly= False)
                merge_num+=1
            else:
                if not mainPresentation:
                    break
                newPresentation = Application.Presentations.Open(file_path + ppt_name, ReadOnly= False, WithWindow=False)

                for Slide in newPresentation.Slides:
                    Slide.Select()
                    Slide.Copy()
                    mainPresentation.Slides.Paste(-1)
                merge_num+=1
                newPresentation.Close()
        except Exception:
            return -1

    present_file_list = os.listdir(save_path)
    present_file_name = "Result_file.pptx"
    file_idx = 2
    while(1):
        if present_file_name in present_file_list:
            present_file_name = "Result_file_" + str(file_idx) + ".pptx"
            file_idx += 1
        else:
            break
    if save_path[-1] == "\\":
        mainPresentation.SaveAs(save_path + present_file_name)
    else:
        mainPresentation.SaveAs(save_path + "\\" + present_file_name)
    mainPresentation.Close()
    Application.Quit()

    return merge_num
