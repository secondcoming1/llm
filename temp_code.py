import pandas as pd

# Create a DataFrame with positive class data for the user in the anomaly
data = {
    'user': [675.0, 675.0, 675.0],  # User ID remains the same
    'logon_on_own_pc_normal': [1.0, 1.0, 1.0],  # Normal logon activity
    'logon_on_other_pc_normal': [0.0, 0.0, 0.0],  # No logon on other PCs
    'logon_on_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour logon
    'logon_on_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour logon on other PCs
    'logon_hour': [9.0, 10.0, 11.0],  # Normal working hours
    'day_of_a_week': [1.0, 2.0, 3.0],  # Normal weekdays
    'device_connects_on_own_pc': [2.0, 2.0, 2.0],  # Normal device connections
    'device_connects_on_other_pc': [0.0, 0.0, 0.0],  # No device connections on other PCs
    'device_connects_on_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour device connections
    'device_connects_on_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour device connections on other PCs
    'documents_copy_own_pc': [5.0, 5.0, 5.0],  # Below excessive copying threshold
    'documents_copy_other_pc': [0.0, 0.0, 0.0],  # No copying on other PCs
    'exe_files_copy_own_pc': [0.0, 0.0, 0.0],  # No executable file copying
    'exe_files_copy_other_pc': [0.0, 0.0, 0.0],  # No executable file copying on other PCs
    'documents_copy_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour document copying
    'documents_copy_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour document copying on other PCs
    'exe_files_copy_own_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour executable file copying
    'exe_files_copy_other_pc_off_hour': [0.0, 0.0, 0.0],  # No off-hour executable file copying on other PCs
    'neutral_sites': [50.0, 50.0, 50.0],  # Normal web activity
    'job_search': [0.0, 0.0, 0.0],  # No job search activity
    'hacking_sites': [0.0, 0.0, 0.0],  # No hacking-related activity
    'neutral_sites_off_hour': [0.0, 0.0, 0.0],  # No off-hour web activity
    'job_search_off_hour': [0.0, 0.0, 0.0],  # No off-hour job search activity
    'hacking_sites_off_hour': [0.0, 0.0, 0.0],  # No off-hour hacking-related activity
    'total_emails': [10.0, 10.0, 10.0],  # Normal email activity
    'int_to_int_mails': [8.0, 8.0, 8.0],  # Internal-to-internal emails
    'int_to_out_mails': [2.0, 2.0, 2.0],  # Below suspicious threshold
    'out_to_int_mails': [0.0, 0.0, 0.0],  # No external-to-internal emails
    'out_to_out_mails': [0.0, 0.0, 0.0],  # No external-to-external emails
    'internal_recipients': [5.0, 5.0, 5.0],  # Normal internal recipients
    'external_recipients': [0.0, 0.0, 0.0],  # No external recipients
    'distinct_bcc': [0.0, 0.0, 0.0],  # No BCC recipients
    'mails_with_attachments': [2.0, 2.0, 2.0],  # Below suspicious threshold
    'after_hour_mails': [0.0, 0.0, 0.0],  # No after-hour emails
    'role': [1.0, 1.0, 1.0],  # User's role remains the same
    'business_unit': [1.0, 1.0, 1.0],  # Business unit remains the same
    'functional_unit': [1.0, 1.0, 1.0],  # Functional unit remains the same
    'department': [1.0, 1.0, 1.0],  # Department remains the same
    'team': [1.0, 1.0, 1.0],  # Team remains the same
    'O': [39.0, 39.0, 39.0],  # Openness score remains the same
    'C': [40.0, 40.0, 40.0],  # Conscientiousness score remains the same
    'E': [47.0, 47.0, 47.0],  # Extraversion score remains the same
    'A': [50.0, 50.0, 50.0],  # Agreeableness score remains the same
    'N': [26.0, 26.0, 26.0]   # Neuroticism score remains the same
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a pickle file
df.to_pickle('temp_csv.pkl')