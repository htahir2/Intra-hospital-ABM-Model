"""
    
    @author: hannantahir
    
"""

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import networkx as nx

''' set paths to directories '''
#indir = '/SPECIFY_PATH_TO_DIRECTORY/'
datadir = indir+'Data/'
''' Uncomment one line from the below three lines. Only one line to be uncommented at a time '''
#resultdir = indir+'Results/' # for complete hospital analysis
#resultdir = indir+'Results_lowrisk/' # for only low risk patient analysis
#resultdir = indir+'Results_highrisk/' # for only high risk patient analysis

#load all files with .xlsx extension from the datadir directory 
filenames = glob.glob(datadir + "/*.xlsx")
df1 = pd.DataFrame()
for f in filenames:
    data = pd.read_excel(f, 'Admisiones')
    df1 = df1.append(data, ignore_index=True)

df1['Date of admission'] = pd.to_datetime(df1['Date of admission'], format='%d/%m/%Y %H:%M:%S')
df1['Date of discharge'] = pd.to_datetime(df1['Date of discharge'], format='%d/%m/%Y %H:%M:%S')
df1['Start transfer'] = pd.to_datetime(df1['Start transfer'], format='%d/%m/%Y %H:%M:%S')
df1['Finish transfer'] = pd.to_datetime(df1['Finish transfer'], format='%d/%m/%Y %H:%M:%S')
### Rename enteries in the data to appropriate names i.e M is replaced with Male, F is replaced with Female ###
df1['Gender'].replace('F','Female',inplace=True)
df1['Gender'].replace('M','Male',inplace=True)
df1['Ward'].replace('psychiatry - Hosp Virgen Macarena','Psychiatry',inplace=True)
df1['Ward'].replace('obstetrics','Obstetrics',inplace=True)
df1['Ward'].replace('otorhinolaryngology ','Otorhinolaryngology',inplace=True)
df1['Ward'].replace('Otorhinolaryngology ','Otorhinolaryngology',inplace=True)
df1['Ward'].replace('Psychiatry - Hosp Virgen Macarena','Psychiatry',inplace=True)
df1['Ward'].replace('Pain unit (anestesiology)','Pain Unit (anestesiology)',inplace=True)
df1['Ward'].replace('NeUrology','Neurology',inplace=True)
df1['Ward'].replace('Oftalmología General','Oftalmology',inplace=True)
df1['Ward'].replace('Pain Unit (anestesiology)','Anesthesiology',inplace=True)
df1['Ward'].replace('Anestesiology','Anesthesiology',inplace=True)
df1['Ward'].replace('Postsurgery Unit','Anesthesiology',inplace=True)
df1['Ward'].replace('Alergología General','Anesthesiology',inplace=True)
df1['Ward'].replace('Anestesia Extraquirúrgica','Anesthesiology',inplace=True)
df1['Ward'].replace('Anestesia para Cirugía Cardiovascular','Anesthesiology',inplace=True)
df1['Ward'].replace('General pediatrics','Pediatrics',inplace=True)
df1['Ward'].replace('Cuidados Críticos Pediátricos','Pediatric ICU',inplace=True)
df1['Ward'].replace('Cuidados Críticos Neonatológicos','Neonatology ICU',inplace=True)
df1['Ward'].replace('Paliative care unit','Paliative care',inplace=True)
df1['Ward'].replace('Nefrology','Nephrology',inplace=True)
df1['Ward'].replace('Reumatology','Rheumatology',inplace=True)
df1['Ward'].replace('Oftalmology','Ophthalmology',inplace=True)
df1['Ward'].replace('Medicina Nuclear General','General Nuclear Medicine',inplace=True)
df1['Ward'].replace('Cirugía Oral y Maxilofacial General','Maxillofacial Surgery',inplace=True)

### Delete admissions where patient was admitted to one of the below wards. even if patient moved just once in one single admission to these ward, complete admission entery is removed. 
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Alergología general', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Aparato Gastroenterology Pediátrico', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Chronic Critical Ischemia Unit', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Cirugía Mayor Ambulatoria (CMA)', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Endovascular General', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Endovascular surgery', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Hematología y Hemoterapia General (Laboratorio)', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Oncología Radioterápica General', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Unidad de Neumología Pediátrica', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Pediatric Pneumology Unit', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Psychiatry', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Obstetrics', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Breast Unit', 'Episode ID'].unique()), 'flag'] = '1'
df1.loc[df1['Episode ID'].isin(df1.loc[df1['Ward'] == 'Paliative care', 'Episode ID'].unique()), 'flag'] = '1'
df1.drop(df1[df1['flag'] == '1'].index, inplace=True)

# make a copy of the df1 and remove un-necessary data columns and duplicates
df11 = df1.copy()
df11.drop(df11.columns[[1,2,3,4,5,6,7,8,15,16,17,18,19]], axis=1, inplace=True)
df11.drop_duplicates(subset=['Episode ID', 'Main diagnosis', 'Diagnosis 1','Diagnosis 2','Diagnosis 3','Diagnosis 4','Diagnosis 5' ], inplace = True)
df12 = df11.replace('',np.nan).melt('Episode ID').dropna()
df12.drop(df12.columns[[1]], axis=1, inplace=True)

## separate letters and numbers from the ICD 10 codesso that stratification can be applied in the nex step
df12[['Let', 'Num']] = df12['value'].str.extract(r'([A-Za-z]+)([\d\.]+)', expand=True)
df12['Num'] = df12['Num'].astype(float)

### add a new column and assign "Y" if patient meets high risk group conditions. Thise conditions are applied to every single row of an admission. 
df12['patient_with_high_riskICD'] = np.where( \
               (((df12['Let']=='C') & ((df12['Num'] >= 00.00) & (df12['Num'] <= 96.00))) | \
                ((df12['Let']=='E') & ((df12['Num'] >= 10.00) & (df12['Num'] <= 14.00))) | \
                ((df12['Let']=='I') & ((df12['Num'] >= 50.00) & (df12['Num'] < 51.00))) | \
                ((df12['Let']=='N') & ((df12['Num'] >= 18.30) & (df12['Num'] <= 18.60))) | \
                ((df12['Let']=='D') & ((df12['Num'] >= 80.00) & (df12['Num'] <= 89.00))) | \
                ((df12['Let']=='M') & ((df12['Num'] >= 34.00) & (df12['Num'] <= 35.00))) | \
                ((df12['Let']=='L') & ((df12['Num'] >= 40.00) & (df12['Num'] < 41.00))) | \
                ((df12['Let']=='R') & ((df12['Num'] >= 76.00) & (df12['Num'] < 77.00))) \
                ),'Y', 'N')

## Look if a patient has been marked as high risk in any row during a single admission. If yes, then add "Y" in the patient_with_high_riskICD column for all subsequent rows. 
df12.loc[df12['Episode ID'].isin(df12.loc[df12['patient_with_high_riskICD'] == 'Y', 'Episode ID'].unique()), 'patient_with_high_riskICD'] = 'Y'
## drop duplicates and few other columns. Resulting dataframe only include Episode ID and highrisk patient "Y" or "N" informations. This dataframe will be later merged to another dataframe.
df12.drop_duplicates(subset=['Episode ID'], inplace = True)
df12.drop(df12.columns[[1,2,3]], axis=1, inplace=True)

## check if data contains enteries with missing gender information
missing_gender_count = df1['Gender'].isnull().sum()
print('Total number of episode/admissions with missing gender information is',missing_gender_count)    

df2=pd.DataFrame()
#df3_Males=pd.DataFrame()
#df4_Females=pd.DataFrame()

# copy selected data columns from df1 to df2
df2['Episode ID']=df1['Episode ID']
df2['NUHSA']=df1['NUHSA']
df2['Centre code']=df1['Centre code']
df2['Gender']=df1['Gender']
df2['Birth year']=df1['Birth year']
df2['Date of admission']=df1['Date of admission']
df2['Date of discharge']=df1['Date of discharge']
df2['Ward']=df1['Ward']
df2['Start transfer']=df1['Start transfer']
df2['Finish transfer']=df1['Finish transfer']
df2['Age']=pd.DatetimeIndex(df2['Date of discharge']).year - df2['Birth year'] # calculate age in years
df2['LOS'] = df2['Date of discharge'] - df2['Date of admission']
df2['LOSdays'] = df2['LOS'].dt.days
df2.reset_index(drop=True,inplace = True)
df2.index = df2.index + 1

## df12 computes Y or N for a patient to be high risk based on all diagnosis in one admission. Now merge high risk patient (Y or N tag) with df2. 
df3 = pd.merge(df2, df12, on='Episode ID', how='outer')

df3.sort_values(['NUHSA','Episode ID','Date of admission'], ascending= [True,True,True], inplace = True)
df3['risk'] = df3['patient_with_high_riskICD'].ne('Y').groupby(df3['NUHSA']).cumprod().map({True:'L',False:'H'})

##### Filter data with LOS > 1 day and for hospital with a specific center code 
df_without_zero_LOS = df3[df3['LOSdays']>=1]
df4 = df_without_zero_LOS[(df_without_zero_LOS['Centre code'] == 10005)]

## keep only rows where date of admissions is between a given period
df5 = df4.loc[(df4['Date of admission'] >= '2016-01-01') & (df4['Date of admission'] <= '2017-01-31')]

#raise Exception('exit')
#df5.drop_duplicates(subset=['Episode ID'], inplace = True)

'''
Make a single choice of either analyzing full hospital without risk stratification or lowrisk data or highrisk data. 
Only one at a time.
'''
df6=df5.copy() ## uncomment to analyze full hospital without risk stratification
#df6=df5[df5['risk'].apply(lambda x: x == 'L')] ## uncomment for lowrisk patients
#df6=df5[df5['risk'].apply(lambda x: x == 'H')] ## uncomment for highrisk patients

'''
Now analyze either complete data or a specific risk group
'''
df_mov = df6.copy()
df6.drop_duplicates(subset=['Episode ID'], inplace = True)
# *************************************************
df5_gender = df6.drop_duplicates(subset=['NUHSA']) # to get males and females patients count, drop duplicates based on patient ids. 
patient_gender_count = df5_gender.groupby('Gender').size()

patient_count=df6.groupby('NUHSA').size()
Admissions_gender_count = df6.groupby('Gender').size()
admissions_count=df6.groupby('Episode ID').size()

## Open a text file where descriptive statistics results will be written
f = open(resultdir+'Descriptive_results.txt', 'w')
f.write('In the full dataset, number of episode/admissions with missing gender information is '+ str(df6['Gender'].isnull().sum())+ '\n')
f.write('The number of patients = ' + str(len(patient_count)) + '\n')
f.write('Total Number of Males patients =' + str(patient_gender_count['Male']) + ' ( ' + str(100*patient_gender_count['Male']/len(patient_count)) + ' % ) \n')
f.write('Total Number of Females patients =' +str(patient_gender_count['Female']) + ' ( ' + str(100*patient_gender_count['Female']/len(patient_count)) + ' % ) \n')
f.write('\n \n')
f.write('Total number of admissions =' + str(len(admissions_count)) + '\n')
f.write('Total Number of Male admissions =' + str(Admissions_gender_count['Male']) + ' ( ' + str(100*Admissions_gender_count['Male']/len(admissions_count)) + ' % ) \n')
f.write('Total Number of Female admissions =' + str(Admissions_gender_count['Female']) + ' ( ' + str(100*Admissions_gender_count['Female']/len(admissions_count)) + ' % ) \n')


# ***** To get patient admission distribution on hourly basis
df6['AdmissionTime'] = pd.DatetimeIndex(df6['Date of admission']).hour
admission_time= df6.groupby('AdmissionTime').size().reset_index(name = "count")
admission_time['count_perc'] = admission_time['count'].apply(lambda x: 100*x/len(admissions_count))
g1=sns.barplot(x='AdmissionTime', y='count_perc', data=admission_time, linewidth=1, color = 'salmon')
plt.ylabel('Admissions [%]', size=18)
plt.xlabel('Time [hours]', size=18)
plt.xticks(rotation = 90, size=15)
plt.yticks(size=15)
plt.tight_layout()
plt.ylim(0,15)
plt.savefig(resultdir+'Patient_arrival_time_hours.png', dpi=600)
plt.close()


# ***** To get patient admission distribution on daily basis
df6['AdmissionDay'] = pd.DatetimeIndex(df6['Date of admission']).date
adm_counts_per_day = df6.groupby(['AdmissionDay'])
df_adm_counts_per_day=pd.DataFrame(adm_counts_per_day.size().reset_index(name = "count"))
df_adm_counts_per_day.set_index('AdmissionDay', inplace=True)
df_adm_counts_per_day.index = pd.DatetimeIndex(data=df_adm_counts_per_day.index)
df_adm_counts_per_day=df_adm_counts_per_day.loc[df_adm_counts_per_day.index >= pd.to_datetime('2016-01-01')]

weekly_admissions = pd.DataFrame()
weekly_admissions['adm_counts'] = df_adm_counts_per_day['count'].resample('W').sum()
weekly_admissions['count_perc'] = weekly_admissions['adm_counts'].apply(lambda x:100*x/len(df6))
weekly_admissions = weekly_admissions.truncate(before='2016-01-01', after='2017-01-31')
weekly_admissions.drop(weekly_admissions.columns[[1]], axis=1, inplace=True)
weekly_admissions.index = pd.to_datetime(weekly_admissions.index)
weekly_admissions.plot(lw=1, x_compat=True, legend=None, c='salmon')
plt.xlim('2016-01', '2017-02')
plt.ylabel('Number of Admissions', size=18)
plt.xlabel('Time', size=18)
plt.xticks(rotation=45, fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.savefig(resultdir+'Patient_weekly_arrivals_absolute.png', dpi=600)
plt.close()

weekly_admissions.to_excel(resultdir+'Patient_weekly_arrivals.xlsx')
df_adm_counts_per_day.to_excel(resultdir+'Admission_counts_per_day.xlsx')
f.write('Average Number of admissions per day =' + str(df_adm_counts_per_day.mean(0)) + '\n')
f.write('with a standard deviation of +/- '+ str(df_adm_counts_per_day.std(0)) + '\n')
f.write('total number days on which admissions took place'+ str(len(df_adm_counts_per_day)) + '\n')

plt.plot(df_adm_counts_per_day)
plt.ylabel('Number of Admissions')
plt.xlabel('Month view [Days]')
plt.title('Number of admissions per day')
plt.xticks(rotation='vertical', fontsize = 6)
plt.savefig(resultdir+'Patient_arrival_time_days.png', dpi=600)
plt.close()


# ******************* make separate copies of data for males and females 
df6_Males = df6[(df6['Gender'] == "Male")]
df6_Females = df6[(df6['Gender'] == "Female")] 

age_min = 0
binwidth = 10
break_points = np.arange(age_min, df6['Age'].max() + binwidth, binwidth)
male_out = df6_Males.groupby([pd.cut(df6_Males['Age'], bins=break_points, include_lowest=False)]).size().reset_index(name = "count")
male_out['count_perc'] = male_out['count'].apply(lambda x: 100*x/len(df6))
patient_out = df6.groupby([pd.cut(df6['Age'], bins=break_points, include_lowest=False)]).size().reset_index(name = "count")
patient_out['count_perc'] = patient_out['count'].apply(lambda x: 100*x/len(df6))

df6['Age'].to_excel(resultdir+'All patient age bins.xlsx')
df6_Males['Age'].to_excel(resultdir+'Male patient age bins.xlsx')
df6_Females['Age'].to_excel(resultdir+'Male patient age bins.xlsx')

p1 = plt.bar(patient_out.index, patient_out['count_perc'], color="b")
# superimpose male patient age on patient data, the remaining patients will be females
p2 = plt.bar(male_out.index, male_out['count_perc'], color="g")
plt.ylabel('Admissions [%]', size=18)
plt.xlabel('Patient Age', size=18)
plt.xticks([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ('0-10', '10-20', '20-30','30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-110', '110-120'))
plt.xticks(rotation=90, size=15)
plt.yticks(size=15)
plt.ylim(0,20)
plt.legend([p2, p1], ["Male", "Female"], fontsize=15)
plt.tight_layout()
plt.savefig(resultdir+'Patient_Age_vs_Number.png', dpi=600)
plt.close()


# ***** To get LOS distribution
# &&&&&& to get LOS for all patients &&&&&&&
counts_pat=df6.groupby(['LOSdays'])
df_LOS_count=pd.DataFrame(counts_pat.size().reset_index(name = "LOS_Count"))
df_LOS_count.sort_values(by=['LOSdays'])
df_LOS_count.set_index('LOSdays', inplace=True)
df_LOS_count['LOS_perc'] = df_LOS_count['LOS_Count'].apply(lambda x: 100*x/len(df6))
#print(df_LOS_count)

# &&&&&&&&&&&&&& to get LOS for Males &&&&&&&&&&&&
df6_Males['LOS'] = df6_Males['Date of discharge'] - df6_Males['Date of admission']  # to get month or day, change the year to month or day. This adds a new colum at the end of df1 dataframe.
df6_Males['LOSdays'] = (df6_Males['LOS'].dt.days)
counts_Males=df6_Males.groupby(['LOSdays'])
df6_Males_LOS_count=pd.DataFrame(counts_Males.size().reset_index(name = "LOS_Count"))
df6_Males_LOS_count.sort_values(by=['LOSdays'])
df6_Males_LOS_count.set_index('LOSdays', inplace=True)
df6_Males_LOS_count['LOS_perc'] = df6_Males_LOS_count['LOS_Count'].apply(lambda x: 100*x/len(df6))

# &&&&&&&&&&&&&& to get LOS for Females &&&&&&&&&&&&
df6_Females['LOS'] = df6_Females['Date of discharge'] - df6_Females['Date of admission']  # to get month or day, change the year to month or day. This adds a new colum at the end of df1 dataframe.
df6_Females['LOSdays'] = (df6_Females['LOS'].dt.days)
counts_Females=df6_Females.groupby(['LOSdays'])
df6_Females_LOS_count=pd.DataFrame(counts_Females.size().reset_index(name = "LOS_Count"))
df6_Females_LOS_count.sort_values(by=['LOSdays'])
df6_Females_LOS_count.set_index('LOSdays', inplace=True)
df6_Females_LOS_count['LOS_perc'] = df6_Females_LOS_count['LOS_Count'].apply(lambda x: 100*x/len(df6))

df_LOS_count.to_excel(resultdir+'LOS_data_Total.xlsx')
df6_Males_LOS_count.to_excel(resultdir+'LOS_data_Males.xlsx')
df6_Females_LOS_count.to_excel(resultdir+'LOS_data_Females.xlsx')

plt.plot(df_LOS_count.index, df_LOS_count['LOS_perc'], '.-', c='salmon', linewidth=1, markersize=10)
plt.xlim(-5,400)
plt.ylim(-2,30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('Admissions [%]', size = 18)
plt.xlabel('Length of Stay [Days]', size = 18)
plt.tight_layout()
plt.savefig(resultdir+'patients_LOS_Perc_total.png', dpi=600)
plt.close()

plt.semilogy(df_LOS_count.index,df_LOS_count['LOS_perc'], color='red', lw=1)
plt.xlim(-5,400)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel('Length of Stay [Days]', size = 18)
plt.tight_layout()
plt.savefig(resultdir+'patients_LOS_Perc_total_logY.png', dpi=600)
plt.close()

plt.scatter(df6_Males_LOS_count.index, df6_Males_LOS_count['LOS_perc'], marker='d', c='g', s=10,clip_on=False)
plt.scatter(df6_Females_LOS_count.index, df6_Females_LOS_count['LOS_perc'], marker='.', c='b', s=10,clip_on=False)
plt.xticks(size=15)
plt.yticks(size=15)
plt.ylabel('Admissions [%]', size = 18)
plt.xlabel('Length of Stay [Days]', size = 18)
plt.legend(["Male", "Female"], fontsize=15, loc='upper right')
plt.ylim(-2,30)
plt.xlim(-5,400)
plt.tight_layout()
plt.savefig(resultdir+'patients_LOS.png', dpi=600)
plt.close()


header = ["LOS", "LOSdays"]
df6.to_excel(resultdir+'LOS_data_all_for_mean_and_std.xlsx', columns = header)
df6_Males.to_excel(resultdir+'LOS_data_Males_for_mean_and_std.xlsx', columns = header)
df6_Females.to_excel(resultdir+'LOS_data_Females_for_mean_and_std.xlsx', columns = header)

f.write('Mean length of stay (including males and females) =  ' + str(df6['LOSdays'].mean()) + ' and standard deviation +/- ' + str(df6['LOSdays'].std()) + '\n')
f.write('Mean length of stay in highrisk males =  ' + str(df6_Males['LOSdays'].mean()) + ' and standard deviation +/- ' + str(df6_Males['LOSdays'].std()) +  '\n')
f.write('Mean length of stay in highrisk females =  ' + str(df6_Females['LOSdays'].mean()) + ' and standard deviation +/- ' + str(df6_Females['LOSdays'].std()) +  '\n')
f.close()




''' Analyze patient movements inside the hospital  '''

df_mov['LOSward'] = df_mov['Finish transfer'] - df_mov['Start transfer']
df_mov.sort_values(['Episode ID','Start transfer','Finish transfer'],ascending=True, inplace = True)
df_mov.drop(df_mov[df_mov['LOSward'] <= timedelta(0)].index, inplace=True)  ### to drop 0 duration LOS in wards and negative LOS
df2_mov = df_mov.assign(Key=(df_mov.Ward.shift()!=df_mov.Ward).cumsum()).groupby(['Episode ID','Ward','Key','risk','LOSdays']).agg({'Start transfer':'first','Finish transfer':'last'}).reset_index().sort_values('Key')


total_admission = df2_mov.copy()
total_admission.drop_duplicates(subset=['Episode ID'], inplace = True)

df_mov_disc=df2_mov.copy()

############ code here to calculate ward discharge probability excluding final discharge.
df_mov_disc['Start transfer date']= df_mov_disc['Start transfer'].dt.date
df_mov_disc['Finish transfer date']= df_mov_disc['Finish transfer'].dt.date
df_mov_disc['Start transfer date'] = pd.to_datetime(df_mov_disc['Start transfer date'])
df_mov_disc['Finish transfer date'] = pd.to_datetime(df_mov_disc['Finish transfer date'])
df_mov_disc["Dst"] = df_mov_disc.groupby("Episode ID").shift(-1)["Ward"]
df_mov_disc.drop(df_mov_disc.columns[[2,4,5,6]], axis=1, inplace=True)
df_mov_disc.sort_values(['Episode ID','Start transfer date'], inplace = True)

df_mov_disc['Dst'].fillna('Final_discharge', inplace=True) ## This marks the last row of every episode ID as final discharge from the hospital
df_mov_disc_excl_final_disch = df_mov_disc.loc[df_mov_disc['Dst'] != 'Final_discharge']

df_mov_disc['row'] = range(len(df_mov_disc))
starts_mov = df_mov_disc[['Start transfer date', 'Ward', 'row', 'Episode ID', 'Dst']].rename(columns={'Start transfer date': 'date'})
ends_mov = df_mov_disc[['Finish transfer date', 'Ward', 'row', 'Episode ID', 'Dst']].rename(columns={'Finish transfer date':'date'})
start_end_mov = pd.concat([starts_mov, ends_mov]).set_index('date')
start_end_mov.sort_values('row', inplace = True)
fact_table_mov = start_end_mov.groupby(["Ward", 'row']).apply(lambda x: x.resample('D').fillna(method='pad'))
del fact_table_mov["row"]
del fact_table_mov["Ward"]

### Max patient counts per day in a ward
df_mov_count_max = fact_table_mov.groupby(['Ward', 'date']).size().reset_index(name = "Max_counts").groupby('Ward').max()
df_mov_count_max.reset_index(inplace = True)
df_mov_count_max.drop(df_mov_count_max.columns[[1]], axis=1, inplace=True)

# Mean patient counts per day in a ward
df_mov_count_mean = fact_table_mov.groupby(['Ward', 'date']).size().reset_index(name = "Mean_counts").groupby('Ward').mean()
df_mov_count_mean.reset_index(inplace = True)

# Merge Max and Mean counts per date in every ward
df_mov_mean_max = pd.merge(df_mov_count_max, df_mov_count_mean, on=['Ward'], how='outer')

# get all ward names after removing final discharges from the hospital
ward_names = df_mov_disc_excl_final_disch.drop_duplicates(subset=['Ward'])
df_mov_mean_dis_per_day = pd.DataFrame(columns=ward_names['Ward'])
df_mov_mean_dis_per_day['Finish transfer date'] = pd.date_range('2016-1-1', periods=397, freq='D')  # This creates one column of dates for the whole dataset period.
df_mov_mean_dis_per_day['Finish transfer date'] = pd.to_datetime(df_mov_mean_dis_per_day['Finish transfer date'], format = '%y%m%d')
# calculate mean discharge from a ward to other wards on every date present in the data
mean_discharge_per_day_per_ward_mov = df_mov_disc_excl_final_disch.groupby(['Ward', 'Finish transfer date']).size().reset_index(name = "count_per_date")

# create a pivot table with date as index and wards as columns and append sum of discharge counts to other wards per date in appropriate cells
m=mean_discharge_per_day_per_ward_mov.pivot_table(index='Finish transfer date',columns='Ward',values='count_per_date',aggfunc='sum')
df_mov_mean_dis_per_day_final=df_mov_mean_dis_per_day.set_index('Finish transfer date').combine_first(m).fillna(0).reset_index()
mean_mov_discharge_per_day =  df_mov_mean_dis_per_day_final.loc[:, df_mov_mean_dis_per_day_final.columns != 'Finish transfer date'].mean().reset_index(name = "Mean_discharge")
df_mov_discharge_prob = pd.merge(df_mov_mean_max, mean_mov_discharge_per_day, on=['Ward'], how='outer')
df_mov_discharge_prob['discharge_prob_max_wardsize'] = df_mov_discharge_prob['Mean_discharge'] / df_mov_discharge_prob['Max_counts']
df_mov_discharge_prob['discharge_prob_mean_wardsize'] = df_mov_discharge_prob['Mean_discharge'] / df_mov_discharge_prob['Mean_counts']
df_mov_discharge_prob.plot(x = 'Ward', y =['discharge_prob_max_wardsize','discharge_prob_mean_wardsize'], kind = 'bar', figsize=(12,10))
plt.ylabel('Daily discharge probability')
plt.legend(["discharge_prob_max_wardsize","discharge_prob_mean_wardsize"], fontsize=10, loc='upper right',fancybox=True, framealpha=0.5)
plt.tight_layout()
plt.savefig(resultdir+'Ward_only_discharge_probability_all.png',dpi=600)
plt.close()

d = mean_mov_discharge_per_day[mean_mov_discharge_per_day['Ward'] != 'Final_discharge'] # to remove final discharge from dataframe
d.plot(x = 'Ward', y =['Mean_discharge'], kind = 'bar', figsize=(12,10))
plt.ylabel('Mean discharge per day')
plt.ylim(0,3.2)
plt.tight_layout()
plt.savefig(resultdir+'Mean_discharge_per_day_discharge_only.png',dpi=600)
plt.close()

df_mov_discharge_prob.to_excel(resultdir+'discharge_mov_probability_data.xlsx')



# to get LOS per ward
df2_mov['LOS'] = df2_mov['Finish transfer'] - df2_mov['Start transfer']
df2_mov['LOSperWard'] = (df2_mov['LOS'].dt.days) + ((df2_mov['LOS'].dt.seconds)/(60*60*24)) # this gives LOS per ward in numeric form. 
df3_mov = df2_mov.copy()
df3_mov.drop(df3_mov.columns[[0,2,3,4,5,6,7]], axis=1, inplace=True)
df3_mov.sort_values(by = 'Ward', inplace = True)
df_LOSperWard= df2_mov.groupby(['Ward','LOSperWard']).size().reset_index(name = "count")
del df_LOSperWard['count']
g = sns.boxplot(x='Ward', y='LOSperWard', data=df3_mov, fliersize=1, linewidth=1, meanline = True, showmeans=True, showfliers=True)
plt.xticks(rotation=90, size=6)
#plt.ylim(0,20)
plt.ylabel('LOS in days')
plt.tight_layout()
plt.savefig(resultdir+'LOS_per_Ward_total_boxplot_with_outliers.png',dpi=600)
plt.close()


## to get LOS per ward ######  plotted using bar plot
g1 = sns.barplot(x='Ward', y='LOSperWard', data=df_LOSperWard, linewidth=1, errwidth=1)
plt.xticks(rotation=90, size=6)
plt.ylabel('LOS in days')
plt.gcf().subplots_adjust(bottom=0.5)
plt.savefig(resultdir+'LOS_per_Ward_total_bargraph.png',dpi=600)
plt.close()

# write LOS per ward data to excel file
df_los_ward = df_LOSperWard.groupby('Ward', as_index=False)['LOSperWard'].mean()
df_los_ward.to_excel(resultdir+'Mean_LOS_per_ward.xlsx')

## Movements versus hospital LOS
movement_vs_LOS= df2_mov.groupby(['Episode ID', 'LOSdays']).size().reset_index(name = "movements")
movement_vs_LOS["movements"] = movement_vs_LOS["movements"].apply(lambda x: x - 1) # to calculate jumps as movements, we subtract -1 from the count data. 
df_mov_los=movement_vs_LOS.groupby(['LOSdays','movements']).size().reset_index(name = "count")
movement_vs_LOS.to_excel(resultdir+'LOS_MOV_all_datapoints.xlsx')

df_mov_los['value'] = df_mov_los['count']*df_mov_los['movements']
mean_mov_vs_los = df_mov_los.groupby(['LOSdays']).sum()
mean_mov_vs_los['Mean Movements'] = mean_mov_vs_los['value'] / mean_mov_vs_los['count']
mean_mov_vs_los.drop(mean_mov_vs_los.columns[[0]], axis=1, inplace=True)

mean_mov_vs_los.to_excel(resultdir+'LOS_vs_Mean_Movements.xlsx')
plt.plot(mean_mov_vs_los.index, mean_mov_vs_los['Mean Movements'], '.', c='salmon', linewidth=1, markersize=10)
plt.xlabel('LOS in days')
plt.ylabel('Mean Movements')
plt.xlim(0,400)
plt.ylim(0,25)
plt.tight_layout()
plt.savefig(resultdir+'LOS_vs_mean_movements.png',dpi=600)
plt.close()

## Percentage of admissions versus movements
movement_stats= df2_mov.groupby('Episode ID').size().reset_index(name = "count")
movement_stats["count"] = movement_stats["count"].apply(lambda x: x - 1) # to calculate jumps as movements, we subtract -1 from the count data. 
df_counts=movement_stats.groupby(['count'])
df_movement_counts=pd.DataFrame(df_counts.size().reset_index(name = "movements_count"))
df_movement_counts.sort_values(by=['count'])
df_movement_counts['value'] = df_movement_counts['count'] * df_movement_counts['movements_count']
df_movement_counts.loc[0:0,'value'] = df_movement_counts.loc[0:0,'movements_count'] # this will override the zero movements count set by the last step to zero back to actual value
df_movement_counts.set_index('count', inplace=True)

df_movement_counts_total_sum = df_movement_counts['movements_count'].sum()
df_movement_counts['mov_perc'] = df_movement_counts['movements_count'].apply(lambda x: 100*x/df_movement_counts_total_sum)


f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
ax.scatter(df_movement_counts.index, df_movement_counts['mov_perc'], marker='d', c='b', s=10,clip_on=False)
ax2.scatter(df_movement_counts.index, df_movement_counts['mov_perc'], marker='d', c='b', s=10,clip_on=False)

ax.set_ylim(85,100)  
ax2.set_ylim(0, 10)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()


d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

f.text(0.02, .5, 'opname (%)', ha='center', va='center', rotation='vertical')
plt.xlabel('Number of movements')
plt.tight_layout()
plt.savefig(resultdir+'patient_movements_count_perc.png', dpi=600)
plt.close()

df_movement_counts.to_excel(resultdir+'patient_movements_count_perc.xlsx')

plt.plot(df_movement_counts.index, df_movement_counts['mov_perc'], '.-', c='salmon', linewidth=1, markersize=10)
plt.ylabel('admissions (%)')
plt.xlabel('Number of Movements')
plt.ylim(0,100)
plt.tight_layout()
plt.savefig(resultdir+'patient_movements_count_perc.png',dpi=600)
plt.close()

movement_stats.to_excel(resultdir+'movement_per_episode_data.xlsx')

df2_mov['jump_count'] = df2_mov['Episode ID'].duplicated('last').astype(int)
movement_stats_per_day = df2_mov.groupby(df2_mov['Finish transfer'].dt.date)['jump_count'].sum().reset_index(name = "total_count")
df2_mov['Start transfer date'] = pd.DatetimeIndex(df2_mov['Start transfer']).date
movement_stats_per_day_per_ward= df2_mov.groupby(['Start transfer date','Ward']).size().reset_index(name = "count")
movement_stats_per_day_per_ward_count= movement_stats_per_day_per_ward.groupby(['Ward','Start transfer date']).sum().reset_index()
mean_ward_mov = movement_stats_per_day_per_ward_count.groupby('Ward', as_index=False)['count'].mean()
mean_ward_mov.set_index('Ward', inplace = True)
fig=mean_ward_mov.plot(kind='bar', title ="average patients movement per day per ward", legend=False, fontsize=10, width=0.25)
plt.ylabel('Average number of movements')
plt.xticks(fontsize=4)
plt.ylim(0,25)
plt.gcf().subplots_adjust(bottom=0.35)
plt.savefig(resultdir+'Ward_vs_mean_patient_movement.png',dpi=600)
plt.close()

## Write descriptive data to file     
f = open(resultdir+'Results_Migration.txt', 'w')
f.write('patient movement data: Number of movements per episode ID is counted for every episode and then following statistics calculated on the total data  \n')
f.write('Total patient movement count: '+ str(movement_stats['count'].sum())+ '\n')
f.write(' Average patient movement per episodeID is : '+ str(movement_stats['count'].mean())+ '\n')
f.write(' Standard deviation patient movement per episodeID is : '+ str(movement_stats['count'].std())+ '\n')
f.write(' Maximum patient movement per episodeID is : '+ str(movement_stats['count'].max())+ '\n')
f.write(' Minimum patient movement per episodeID is : '+ str(movement_stats['count'].min())+ '\n')

f.write('\n \n \n patient movement data: Number of movements per day is counted based on transfer start date and then following statistics calculated on the total data  \n')
f.write(' total patient movements : '+ str(movement_stats_per_day['total_count'].sum())+ '\n')
f.write(' Average patient movement per day is : '+ str(movement_stats_per_day['total_count'].mean())+ '\n')
f.write(' Standard deviation patient movement per day is : '+ str(movement_stats_per_day['total_count'].std())+ '\n')
f.write(' Maximum patient movement per day is : '+ str(movement_stats_per_day['total_count'].max())+ '\n')
f.write(' Minimum patient movement per day is : '+ str(movement_stats_per_day['total_count'].min())+ '\n')
f.close()

movement_stats.drop(movement_stats.columns[[0]], axis=1, inplace=True)
movement_stats.to_excel(resultdir+'mean_movements_per_episode.xlsx')

''' create network file with edges and weight '''
df_edges = df2_mov["Ward"].to_frame()
df_edges["Dst"] = df2_mov.groupby("Episode ID").shift(-1)["Ward"]
df_edges.dropna(inplace = True)
df_edges.rename(columns={"Ward":"Src"}, inplace = True)
df_network=df_edges.groupby(['Src', 'Dst']).size().reset_index(name = "weight")

### add missing wards for high risk group
#df_network = df_network.append(pd.Series(['Pediatric surgery','Pediatric surgery', 1], index=df_network.columns ), ignore_index=True)
#df_network = df_network.append(pd.Series(['General pediatric surgery','General pediatric surgery', 1], index=df_network.columns ), ignore_index=True)
#df_network = df_network.append(pd.Series(['Neonatology','Neonatology', 1], index=df_network.columns ), ignore_index=True)
#df_network = df_network.append(pd.Series(['Neonatology ICU','Neonatology ICU', 1], index=df_network.columns ), ignore_index=True)
##### add missing wards for low risk group
#df_network = df_network.append(pd.Series(['Dermatology','Dermatology', 1], index=df_network.columns ), ignore_index=True)
#df_network = df_network.append(pd.Series(['Maxillofacial Surgery','Maxillofacial Surgery', 1], index=df_network.columns ), ignore_index=True)

df_network.sort_values(['Src','Dst'], inplace = True)
### writing network file
network = nx.from_pandas_dataframe(df_network,source='Src', target='Dst', edge_attr=["weight"], create_using=nx.DiGraph())
nx.write_graphml(network,resultdir+'movement_network.graphml')

#raise Exception('exit')
