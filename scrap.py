import pandas as pd  
  
# Define the rules  
rules = {  
    "Rule 1: Frequent off-hour logins on the user’s own PC are highly indicative of anomalous behavior.": lambda row: row['logon_on_own_pc_off_hour'] < 0.5,  
    "Rule 2: A sudden increase in device connections during off-hours is strongly anomalous.": lambda row: row['device_connects_on_own_pc_off_hour'] < 0.5,  
    "Rule 3: Large-scale document copying on the user’s own PC during off-hours is a critical anomaly indicator.": lambda row: row['documents_copy_own_pc_off_hour'] < 0.5,  
    "Rule 4: Copying executable files on the user’s own PC during off-hours is anomalous behavior.": lambda row: row['exe_files_copy_own_pc_off_hour'] < 0.5,  
    "Rule 5: Elevated browsing activity on job search sites is a potential anomaly trigger.": lambda row: row['job_search'] < 2.0,  
    "Rule 6: High browsing activity on neutral sites during off-hours may contribute to anomalous behavior.": lambda row: row['neutral_sites_off_hour'] < 4.0,  
}  
  
# Anomalous outcome (most recent timestep `t`)  
anomalous_data = {  
    'logon_on_own_pc_off_hour': 1.0,  
    'device_connects_on_own_pc_off_hour': 0.0,  
    'documents_copy_own_pc_off_hour': 0.0,  
    'exe_files_copy_own_pc_off_hour': 0.0,  
    'job_search': 0.0,  
    'neutral_sites_off_hour': 6.0,  
}  
  
# Counterfactuals (most recent timestep `t`)  
counterfactuals_data = [  
    {'logon_on_own_pc_off_hour': 0.418662, 'device_connects_on_own_pc_off_hour': 0.006546, 'documents_copy_own_pc_off_hour': 0.0, 'exe_files_copy_own_pc_off_hour': 0.62919, 'job_search': 1.917232, 'neutral_sites_off_hour': 2.963571},  
    {'logon_on_own_pc_off_hour': 0.350859, 'device_connects_on_own_pc_off_hour': 0.011873, 'documents_copy_own_pc_off_hour': 0.0, 'exe_files_copy_own_pc_off_hour': 0.565188, 'job_search': 1.332852, 'neutral_sites_off_hour': 2.09122},  
    {'logon_on_own_pc_off_hour': 0.395705, 'device_connects_on_own_pc_off_hour': 0.022897, 'documents_copy_own_pc_off_hour': 0.25723, 'exe_files_copy_own_pc_off_hour': 0.542593, 'job_search': 2.037543, 'neutral_sites_off_hour': 3.592119},  
    {'logon_on_own_pc_off_hour': 0.352738, 'device_connects_on_own_pc_off_hour': 0.022659, 'documents_copy_own_pc_off_hour': 0.301467, 'exe_files_copy_own_pc_off_hour': 0.898456, 'job_search': 1.635325, 'neutral_sites_off_hour': 3.387855},  
    {'logon_on_own_pc_off_hour': 0.403395, 'device_connects_on_own_pc_off_hour': 0.010309, 'documents_copy_own_pc_off_hour': 0.478423, 'exe_files_copy_own_pc_off_hour': 0.742353, 'job_search': 1.835372, 'neutral_sites_off_hour': 2.768982},  
]  
  
# Convert counterfactuals to a DataFrame  
counterfactuals_df = pd.DataFrame(counterfactuals_data)  
  
# Count rule consistency  
results = {}  
for rule_name, rule_func in rules.items():  
    # Apply the rule to each row and count how many rows satisfy the rule  
    consistent_count = counterfactuals_df.apply(rule_func, axis=1).sum()  
    results[rule_name] = consistent_count  
  
# Sort the rules by the number of consistent counterfactuals in descending order  
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)  
  
# Print the results  
print("Rule Consistency Results:")  
for rule_name, count in sorted_results:  
    print(f"{rule_name}: {count} counterfactuals consistent")  