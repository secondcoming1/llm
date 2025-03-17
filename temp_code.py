import pandas as pd

# Define the data for the positive class example
data = {
    'user': [675.0, 675.0, 675.0],  # User ID remains the same
    'logon_on_own_pc_normal': [1.0, 1.0, 1.0],  # Normal logon activity on own PC
    'logon_on_other_pc_normal': [0.0, 0.0, 0.0],  # No logon activity on other PCs
    'logon_on_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour logon activity
    'logon_on_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour logon activity on other PCs
    'logon_hour': [9.0, 10.0, 11.0],  # Normal working hours
    'day_of_a_week': [1.0, 2.0, 3.0],  # Normal weekdays
    'device_connects_on_own_pc': [2.0, 2.0, 2.0],  # Limited device connections on own PC
    'device_connects_on_other_pc': [0.0, 0.0, 0.0],  # No device connections on other PCs
    'device_connects_on_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour device connections
    'device_connects_on_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour device connections on other PCs
    'documents_copy_own_pc': [5.0, 5.0, 5.0],  # Limited document copying on own PC
    'documents_copy_other_pc': [0.0, 0.0, 0.0],  # No document copying on other PCs
    'exe_files_copy_own_pc': [0.0, 0.0, 0.0],  # No executable file copying on own PC
    'exe_files_copy_other_pc': [0.0, 0.0, 0.0],  # No executable file copying on other PCs
    'documents_copy_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour document copying
    'documents_copy_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour document copying on other PCs
    'exe_files_copy_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour executable file copying
    'exe_files_copy_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour executable file copying on other PCs
    'neutral_sites': [20.0, 20.0, 20.0],  # Limited neutral site visits
    'job_search': [0.0, 0.0, 0.0],  # No job search activity
    'hacking_sites': [0.0, 0.0, 0.0],  # No hacking site visits
    'neutral_sites_off_hour': [0.0, 0.0, 0.0],  # No off-hour neutral site visits
    'job_search_off_hour': [0.0, 0.0, 0.0],  # No off-hour job search activity
    'hacking_sites_off_hour': [0.0, 0.0, 0.0],  # No off-hour hacking site visits
    'total_emails': [5.0, 5.0, 5.0],  # Limited total emails sent
    'int_to_int_mails': [4.0, 4.0, 4.0],  # Mostly internal emails
    'int_to_out_mails': [1.0, 1.0, 1.0],  # Limited internal-to-external emails
    'out_to_int_mails': [0.0, 0.0, 0.0],  # No external-to-internal emails
    'out_to_out_mails': [0.0, 0.0, 0.0],  # No external-to-external emails
    'internal_recipients': [3.0, 3.0, 3.0],  # Limited internal recipients
    'external_recipients': [1.0, 1.0, 1.0],  # Limited external recipients
    'distinct_bcc': [0.0, 0.0, 0.0],  # No BCC usage
    'mails_with_attachments': [1.0, 1.0, 1.0],  # Limited emails with attachments
    'after_hour_mails': [0.0, 0.0, 0.0],  # No after-hour emails
    'role': [1.0, 1.0, 1.0],  # Role remains the same
    'business_unit': [14.0, 14.0, 14.0],  # Business unit remains the same
    'functional_unit': [1.0, 1.0, 1.0],  # Functional unit remains the same
    'department': [2.0, 2.0, 2.0],  # Department remains the same
    'team': [3.0, 3.0, 3.0],  # Team remains the same
    'O': [39.0, 39.0, 39.0],  # Openness score remains the same
    'C': [40.0, 40.0, 40.0],  # Conscientiousness score remains the same
    'E': [47.0, 47.0, 47.0],  # Extraversion score remains the same
    'A': [50.0, 50.0, 50.0],  # Agreeableness score remains the same
    'N': [26.0, 26.0, 26.0]   # Neuroticism score remains the same
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('temp_csv.csv', index=False)