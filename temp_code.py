import pandas as pd

# Counterfactual data
counterfactual_data = [
    [0.918692, 0.035198, 0.418662, 0.233116, 0.0, 0.009844, 0.62919, 1.917232, 1.946101],
    [0.850021, 0.019444, 0.350859, 0.212401, 0.0, 0.052881, 0.565188, 1.332852, 1.721338],
    [0.874619, 0.012317, 0.395705, 0.231019, 0.0, 0.0, 0.542593, 2.037543, 2.159656],
    [0.852693, 0.085231, 0.352738, 0.26441, 0.0, 0.097775, 0.898456, 1.635325, 1.712093],
    [0.837595, 0.0, 0.403395, 0.221397, 0.0, 0.0, 0.742353, 1.835372, 2.028352]
]

# Anomalous data (for comparison against counterfactuals)
anomalous_data = {
    "logon_on_own_pc_normal": 0.0,
    "logon_on_own_pc_off_hour": 1.0,
    "documents_copy_own_pc_off_hour": 10.0,
    "device_connects_on_other_pc_off_hour": 1.0,
    "exe_files_copy_own_pc_off_hour": 2.0,
    "job_search": 1.0,
    "after_hour_mails": 2.0,
}

# Rules
def rule_1(logon_on_own_pc_normal):
    return logon_on_own_pc_normal > 0.5

def rule_2(logon_on_own_pc_off_hour):
    return logon_on_own_pc_off_hour < 0.5

def rule_3(documents_copy_own_pc_off_hour, exe_files_copy_own_pc_off_hour):
    return documents_copy_own_pc_off_hour < 5.0 and exe_files_copy_own_pc_off_hour < 1.0

def rule_4(job_search):
    return job_search < 1.5

def rule_5(after_hour_mails):
    return after_hour_mails < 1.5

def rule_6(device_connects_on_other_pc_off_hour):
    return device_connects_on_other_pc_off_hour < 0.5

# Create a DataFrame from counterfactual data
columns = [
    "logon_on_own_pc_normal",
    "logon_on_own_pc_off_hour",
    "documents_copy_own_pc_off_hour",
    "device_connects_on_other_pc_off_hour",
    "exe_files_copy_own_pc_off_hour",
    "job_search",
    "after_hour_mails",
    "hacking_sites",  # Not used in rules but included in data
    "total_emails"    # Not used in rules but included in data
]
df_counterfactuals = pd.DataFrame(counterfactual_data, columns=columns)

# Apply rules and count how many counterfactuals are consistent with each rule
rule_counts = {
    "Rule 1": df_counterfactuals["logon_on_own_pc_normal"].apply(rule_1).sum(),
    "Rule 2": df_counterfactuals["logon_on_own_pc_off_hour"].apply(rule_2).sum(),
    "Rule 3": df_counterfactuals.apply(lambda x: rule_3(x["documents_copy_own_pc_off_hour"], x["exe_files_copy_own_pc_off_hour"]), axis=1).sum(),
    "Rule 4": df_counterfactuals["job_search"].apply(rule_4).sum(),
    "Rule 5": df_counterfactuals["after_hour_mails"].apply(rule_5).sum(),
    "Rule 6": df_counterfactuals["device_connects_on_other_pc_off_hour"].apply(rule_6).sum(),
}

# Sort the rules by the number of consistent counterfactuals
sorted_rule_counts = dict(sorted(rule_counts.items(), key=lambda item: item[1], reverse=True))

# Print the results
print("Number of counterfactuals consistent with each rule (sorted):")
for rule, count in sorted_rule_counts.items():
    print(f"{rule}: {count}")