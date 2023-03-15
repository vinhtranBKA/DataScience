# %% [code]
import numpy as np
import pandas as pd
import math
import os

def pre_process(data):
    feature = ['EVENT_ID','L_CAUSE_PROT_TYPE','CAUSE_CODE','EVENT_TIME']
    pre_data = data[feature]
    pre_data.fillna('0', inplace=True)
    pre_data = pre_data.replace(['unspecified', 'unknown-E-RAB-ID'], -1)
    pre_data['CAUSE_CODE'] = pre_data['CAUSE_CODE'].astype('int')
    return pre_data

class slicing_window_test(object):
    def __init__(self, data, feature_time):
        self.data = data
        self.feature_time = np.array(data[feature_time]) 
        self.temp_time = math.ceil(self.feature_time[0]/60000) * 60000
    
    def get_time(self):
        return self.temp_time
    
    def get_item(self):
        return self.data

def KPI_eventID_ccode(data, eventID, cause_code):
    count = 0
    list_eventID = list(data['EVENT_ID'])
    list_cause_code = list(data['CAUSE_CODE'])
    for i in range(data.shape[0]):
        if ((list_eventID[i] == eventID) and (list_cause_code[i] == cause_code)):
            count = count + 1
    if data.shape[0] == 0:
        return 0
    else:
        return count/(data.shape[0])

def KPI_EVENT_ID(unique_id, count_id,unique_combine, count_combine):
    kpi = np.zeros(len(unique_combine))
    for index_combine in range(len(unique_combine)):
        for index_ID in range(len(unique_id)):
            if unique_combine[index_combine][0] == unique_id[index_ID]:
                kpi[index_combine] = count_combine[index_combine]/float(count_id[index_ID])
                break
    return kpi    


def make_new_data_test(data):
    loader_time = slicing_window_test(data,'EVENT_TIME')
    new_data =  pd.DataFrame()
    data.sort_values(by=['EVENT_TIME'])
    temp_time = loader_time.get_time()
    data_window = loader_time.get_item()
    unique_id, count_id = np.unique(data_window['EVENT_ID'].to_numpy().astype("<U22"), axis=0, return_counts=True)
    unique_combine, count_combine = np.unique(data_window.drop(['EVENT_TIME'], axis = 1).to_numpy().astype("<U22"), axis=0, return_counts=True)
    name_feature = []
    kpi_combine = KPI_EVENT_ID(unique_id, count_id,unique_combine, count_combine)
    kpi_combine = np.insert(kpi_combine, len(kpi_combine), temp_time, axis=0)
    for pre_name in unique_combine:
        name = pre_name[0] + "_" + pre_name[1] + "_"  + pre_name[2]
        name_feature.append(name)
    name_feature.append('EVENT_TIME')
    new_data = pd.DataFrame(np.array([kpi_combine]), columns = name_feature)
    return new_data

class slicing_window_train(object):
    def __init__(self, data, feature_time, window_size = 1000, repeat_size=0.1):
        self.data = data
        self.window_size = window_size
        self.feature_time = np.array(data[feature_time]) 
        self.temp_time = math.ceil(self.feature_time[0]/self.window_size) * self.window_size
        self.flag = True
        self.repeat_size = repeat_size
    
    def get_time(self):
        return self.temp_time
    
    def get_item(self):
        temp_time = self.temp_time
        if (self.temp_time > (self.feature_time[-1] - (1 + self.repeat_size) * self.window_size)):
            self.flag = False
        self.temp_time = int(self.temp_time + self.window_size * self.repeat_size)
        return (self.data).loc[((self.data)['EVENT_TIME'] >= temp_time) & ((self.data)['EVENT_TIME'] < (temp_time + self.window_size))]

def make_new_data_train(data, window_size, repeat_size):
    loader_time = slicing_window_train(data,'EVENT_TIME',window_size * 1000, repeat_size)
    new_data =  pd.DataFrame()
    data.sort_values(by=['EVENT_TIME'])
    while(loader_time.flag == True):
        temp_time = loader_time.get_time()
        data_window = loader_time.get_item()
        unique_id, count_id = np.unique(data_window['EVENT_ID'].to_numpy().astype("<U22"), axis=0, return_counts=True)
        unique_combine, count_combine = np.unique(data_window.drop(['EVENT_TIME'], axis = 1).to_numpy().astype("<U22"), axis=0, return_counts=True)
        name_feature = []
        kpi_combine = KPI_EVENT_ID(unique_id, count_id,unique_combine, count_combine)#######################################
        kpi_combine = np.insert(kpi_combine, len(kpi_combine), temp_time, axis=0)
        for pre_name in unique_combine:
            name = pre_name[0] + "_" + pre_name[1] + "_"  + pre_name[2]
            name_feature.append(name)
        name_feature.append('EVENT_TIME')
        new_data_unit = pd.DataFrame(np.array([kpi_combine]), columns = name_feature)
        new_data = pd.concat([new_data, new_data_unit])
    new_data.fillna(0, inplace=True)
    return new_data

def main(path):
    data = pd.read_csv(path, sep=";", header=None)
    data.columns = ['EVENT_ID', 'EVENT_RESULT', 'DURATION', 'REQUEST_RETRIES', 
                    'SUB_TYPE', 'MSISDN', 'IMSI','MTMSI','IMEISV','MMEGI','MMEC',
                    'TAC','ECI','SGW','SGSN','L_CAUSE_PROT_TYPE','CAUSE_CODE',
                    'SUB_CAUSE_CODE','APN','PDN_DEFAULT_BEARER_ID','PDN_PAA',
                    'PDN_PGW','ORIGINATING_CAUSE_PROT_TYPE','ORIGINATING_CAUSE_CODE',
                    'CSG_ID','OLD_MTMSI','OLD_TAC','OLD_MMEGI','OLD_MMEC','OLD_ECI',
                    'OLD_SGW','OLD_SGSN','MSC','TARGET_LAC','LAC','RAC','CI',
                    'HANDOVER_NODE_ROLE','HANDOVER_RAT_CHANGE_TYPE','HANDOVER_SGW_CHANGE_TYPE',
                    'TARGET_RNC_ID','TARGET_MACRO_ENODEB_ID','SRVCC_TYPE',
                    'CS_FALLBACK_SERVICE_TYPE','CSFB_TRIGGERED','L_SERVICE_REQ_TRIGGER','COMBINED_TAU_TYPE',
                    'DETACH_TRIGGER','EVENT_TIME','PAGING_ATTEMPTS','UE_REQUESTED_APN','DATE_HOUR']
    pre_data = pre_process(data)
    new_data = make_new_data_test(pre_data)
    return new_data


if __name__ == "__main__":
    new_data = pd.DataFrame()
    path = input("Mời bạn nhập đường dẫn dữ liệu: ")
    os.walk(path)
    list_folder = [x[0] for x in os.walk(path)]
    for sub_folder in list_folder[1:-2]:
        dir_list = os.listdir(sub_folder)
        for file in dir_list:
            new_data = pd.concat([new_data,main(sub_folder + '\\' + file)])
            new_data.fillna(0, inplace=True)
    new_data.to_csv("data_abnormal/Data.csv")

