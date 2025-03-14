import pandas as pd

# Load the counterfactual data from the file
counterfactuals_file = 'counterfactuals.pkl'
counterfactuals_df = pd.read_pickle(counterfactuals_file)

# Extract the values for the most recent timestep (t) from the list in the cells
counterfactuals_df = counterfactuals_df.applymap(lambda cell: cell[2] if isinstance(cell, list) else cell)

# Define the rules
rules = {
    "Rule 1": lambda row: row['logon_on_own_pc_off_hour'] > 0.5 or row['logon_on_other_pc_off_hour'] > 0.5,
    "Rule 2": lambda row: row['device_connects_on_own_pc_off_hour'] > 0.5 or row['device_connects_on_other_pc_off_hour'] > 0.5,
    "Rule 3": lambda row: row['documents_copy_own_pc_off_hour'] > 0.3 or row['exe_files_copy_own_pc_off_hour'] > 0.3,
    "Rule 4": lambda row: row['hacking_sites_off_hour'] > 0,
    "Rule 5": lambda row: row['after_hour_mails'] > 0.5 or row['distinct_bcc'] > 0.5,
    "Rule 6": lambda row: (
        (row['logon_on_own_pc_off_hour'] > 0.5 or row['logon_on_other_pc_off_hour'] > 0.5) and
        (row['device_connects_on_own_pc_off_hour'] > 0.5 or row['device_connects_on_other_pc_off_hour'] > 0.5) and
        (row['documents_copy_own_pc_off_hour'] > 0.3 or row['exe_files_copy_own_pc_off_hour'] > 0.3) and
        (row['hacking_sites_off_hour'] > 0) and
        (row['after_hour_mails'] > 0.5 or row['distinct_bcc'] > 0.5)
    )
}

# Count how many counterfactuals are consistent with each rule
results = {}
for rule_name, rule_func in rules.items():
    results[rule_name] = counterfactuals_df.apply(rule_func, axis=1).sum()

# Print the results
print("Consistency of Counterfactuals with Rules:")
for rule_name, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{rule_name}: {count} counterfactuals consistent")