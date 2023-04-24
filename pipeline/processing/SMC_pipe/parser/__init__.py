

# For default Values
default_none = "XXXX"
# 날짜 관련
birth_year_default = "1900"
birth_month_default = "01"
birth_day_default = "01"
startdatetime_year_default = "1900"
startdatetime_month_default = "01"
startdatetime_day_default = "01"
startdatetime_time_default = "00:00:00"

def get_path_xml_data(root_path):
    file_dirs = []
    for root, dirs, files in os.walk(root_path):
        if len(files)>0:
            for file_name in files:
                if file_name.endswith('.xml'):
                    file_dirs.append(os.path.join(root, file_name))
    return file_dirs


def get_text(root, target):
    return root.find(target).text.strip() if root.find(target).text else default_none

def make_textlist_under(root):
    result_list = []
    for child in root:
        if child.text:
            result_list.append(child.text)
        under_list = make_textlist_under(child)
        if len(under_list) != 0:
            result_list += under_list
    return result_list

class philips_parser:
    def __init__(self, root):
        self.root = root
        self.patient_info = None
        self.study_info = None
        self.examination = None
        self.wave_info = None
        
        self.get_patient_info()
        self.get_study_info()
        self.get_examination()
        self.get_wave_info()


    def get_patient_info(self):
        self.patient_info = self.root.find("PatientInfo")

    def get_study_info(self):
        self.study_info = self.root.find("StudyInfo")

    def get_examination(self):
        self.examination = self.root.find("Examination")
    
    def get_wave_info(self):
        self.wave_info = self.root.find("WaveInfo")


    def get_id(self):
        try:
            id_ = get_text(self.patient_info, "ID")
        except:
            id_ = default_none
        return id_

    def get_birthdate(self):   # YYYYMMDD
        try:
            birthdate = get_text(self.patient_info, "BirthDate")
            if birthdate == default_none:
                raise Exception("Wrong Format")
        except:
            birthdate = birth_year_default+birth_month_default+birth_day_default
        return birthdate
    
    def get_gender(self):   # Femal & Male
        try:
            gender = get_text(self.patient_info, "Gender")
        except:
            gender = default_none
        return gender

    def get_startdatetime(self):   #2016-01-04_00:47:54
        try:
            date_ = get_text(self.study_info, "Date")
            if date_ == default_none:
                raise Exception("Wrong Format")
        except:
            date_ = startdatetime_year_default+"-"+startdatetime_month_default+"-"+startdatetime_day_default
        try:
            time_ = get_text(self.study_info, "Time")
            if time_ == default_none:
                raise Exception("Wrong Format")
        except:
            time_ = startdatetime_time_default
        return date_+"_"+time_

    def get_diagnosis(self):
        sever = get_text(self.examination, "Severity")
        diagnosis_list = [sever]
        for diag in self.examination.findall("Diagnosis"):
            diagnosis_list += make_textlist_under(diag)
        return diagnosis_list

    def get_sampling_rate(self):
        try:
            sampling_rate = get_text(self.wave_info, "SamplingRate")
        except:
            sampling_rate = default_none
        return sampling_rate

    def get_amplitude(self):
        try:
            amplitude = get_text(self.wave_info, "Amplitude")
        except:
            amplitude = default_none
        return amplitude

    def get_wavename_list(self):
        name_list = []
        try:
            waves = self.wave_info.findall("WaveData")
            for wa in waves:
                name_list.append(get_text(wa, "Name"))
        except:
            name_list = []
        while(len(name_list)<12):
            name_list.append(default_none)
        return name_list

    def get_wavedata(self, lead="II"):
        waves = self.wave_info.findall("WaveData")
        for wa in waves:
            name = get_text(wa, "Name")
            if name == lead:
                sampling_rate = get_text(wa, "SamplingRate")
                data = get_text(wa, "Data")
                
                return sampling_rate, data

        return default_none, "0"


class past_parser:
    def __init__(self, root):
        self.root = root
        self.patient_info = None
        self.study_info = None

        self.get_patient_info()
        self.get_study_info()

    def get_patient_info(self):
        self.patient_info = self.root.find("PatientInfo")

    def get_study_info(self):
        self.study_info = self.root.find("StudyInfo")

    def get_id(self):
        try:
            id_ = get_text(self.patient_info, "PatientID")
        except:
            id_ = default_none
        return id_

    def get_birthdate(self, xml_path=""):
        try:
            birthdate = get_text(self.patient_info, "PatientDOB")
            if birthdate == default_none:
                years = get_text(self.patient_info, "PatientAge").replace("yrs","")
                date_ = self.get_startdatetime(xml_path=xml_path)
                birthdate_year = int(date_[:4]) - int(years) + 1
                birthdate=str(birthdate_year)+birth_month_default+birth_day_default
        except:
            birthdate = birth_year_default+birth_month_default+birth_day_default
        return birthdate
    
    def get_gender(self):
        try:
            gender = get_text(self.patient_info, "PatientSEX")
        except:
            gender = default_none
        return gender

    def get_startdatetime(self, xml_path=""):
        try:
            date_ = get_text(self.study_info, "StudyDate")
            if date_ == default_none: #경로로라도...
                path_list = xml_path.split("/")
                date_ = path_list[3] + "-" + path_list[4][:2] + "-" + path_list[4][2:]
        except:
            date_ = startdatetime_year_default+"-"+startdatetime_month_default+"-"+startdatetime_day_default
        try:
            time_ = get_text(self.study_info, "StudyTime")
            if time_ == default_none:
                raise Exception("Wrong Format")
        except:
            time_ = startdatetime_time_default
        return date_+"_"+time_


    def get_diagnosis(self):
        diagnosis_list = []
        for diag in self.study_info.findall("Interpretation"):
            diagnosis_list += make_textlist_under(diag)
        return diagnosis_list

    def get_sampling_rate(self):
        try:
            sampling_rate = get_text(self.study_info, "SampleRate")
        except:
            sampling_rate = default_none
        return sampling_rate

    def get_amplitude(self): #amplitudegain
        try:
            amplitude = get_text(self.study_info, "amplitudegain")
        except:
            amplitude = default_none
        return amplitude

    def get_wavename_list(self):
        name_list = []
        try:
            waves = self.study_info.findall("RecordData")
            for wa in waves:
                name_list.append(get_text(wa, "Channel"))
        except:
            name_list = []
        while(len(name_list)<12):
            name_list.append(default_none)
        return name_list

    def get_wavedata(self, lead="II"):
        waves = self.study_info.findall("RecordData")
        for wa in waves:
            name = get_text(wa, "Channel")
            if name == lead:
                sampling_rate = get_text(self.study_info, "SampleRate")
                data = get_text(wa.find('Waveform'), 'Data')
                
                return sampling_rate, data

        return default_none, "0"


