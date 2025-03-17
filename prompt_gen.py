def get_prompts(model_type: str, question_type: str) -> tuple:
    """
    Generate system and user prompts based on the ML model type and question type.
    
    Parameters:
    model_type (str): Type of ML model (e.g., "LSTM", "RF").
    question_type (str): Type of question asked (e.g., "generate_rule", "generate_code").
    
    Returns:
    tuple: (system_prompt, user_prompt)
    """
    
    # Define system prompts based on model type
    system_prompts = {
        "LSTM": (
            "You are a network security expert. Your task is to analyze anomalies "
            "detected by an ML system that flags insider threats on a network. "
            "Interpret the provided anomaly data and counterfactual cases and extract "
            "the most important observed rules."
        ),
        "RF": (
            "You are a network security expert specialized in decision-tree-based anomaly detection. "
            "Your task is to analyze feature-based anomalies detected by a Random Forest-based insider threat detection system. "
            "Interpret the provided anomaly data and extract meaningful decision rules."
        )
    }
    
    # Define user prompts based on model type and question type
    user_prompts = {
        ("LSTM", "generate_rule"): (
            "The Immutable features, negative outcome, counterfactual cases, and inferred rules are provided below.\n\n"
            "Immutable Features:\n{immutable_features}\n\n"
            "Negative Outcome (Anomaly) Data:\n{negative_outcome}\n\n"
            "Counterfactual Cases Data:\n{positive_outcome}\n\n"
            "Extract the most important observed rules from the provided insider anomaly data. "
            "Please explain which features are driving the anomaly and how the counterfactual cases inform the rule extraction. "
            "The Anomaly and counterfactual data is structured as sequences of 3 timesteps (lookback of 3) for each user. "
            "Each feature has values for t-2, t-1, and t. The rules should be applied to the most recent timestep (t) for each feature."
            " ----- Rules -----"
            " <List of Rules>"
        ),
        ("RF", "generate_rule"): (
            "The feature importance and decision rules are provided below for analysis.\n\n"
            "Feature Importance:\n{feature_importance}\n\n"
            "Negative Outcome (Anomaly) Data:\n{negative_outcome}\n\n"
            "Counterfactual Cases Data:\n{positive_outcome}\n\n"
            "Extract decision tree-based rules that best explain the anomalies detected. "
            "Identify the key feature splits and their thresholds that distinguish anomalies from normal behavior."
        ),
        ("LSTM", "generate_code"): (
    "The Immutable features, negative outcome, counterfactual cases, and inferred rules are provided below.\n\n"
    "Immutable Features:\n{immutable_features}\n\n"
    "Negative Outcome (Anomaly) Data:\n{negative_outcome}\n\n"
    "Counterfactual Cases Data:\n{positive_outcome}\n\n"
    "Derived Rules:\n{rules}\n\n"
     "The following info about the dataset is available:\n{dataset_info}"
       
    "you should generate python code to count how many of the counterfactuals are consistent with the rule.do not skip any of the data in the code generated. The code should create a df with the counterfactuals provided and then check for each rule how many of them follow the rules. Order the rules. Finally, you should print the results. "
    "assume that the counterfactual data is going to be read from a file named 'counterfactuals.pkl'. and use applymap(lambda cell: cell[2] if isinstance(cell, list) else cell) to extract the values from the list in the cells."
    "The counterfactual data is structured as sequences of 3 timesteps (lookback of 3) for each user. Each feature has values for t-2, t-1, and t. The rules should be applied to the most recent timestep (t) for each feature."
        """        ----- Code -----
            ```
            import pandas as pd
            #complete this code
            ```
    """
 
         ),
        ("RF", "generate_code"): (
            "Generate a Python script to analyze feature-based anomaly detection data using a Random Forest model. The script should: \n"
            "- Process and format the feature importance data.\n"
            "- Extract decision tree-based rules that highlight anomalies.\n"
            "- Provide an explanation of the most influential features.\n\n"
            "Input Data:\n"
            "Feature Importance: {feature_importance}\n"
            "Negative Outcome Data: {negative_outcome}\n"
            "Counterfactual Cases Data: {positive_outcome}\n"
            "Rules should be extracted using decision paths and returned as structured output."
        )
    }
    
    # Get the appropriate prompts
    system_prompt = system_prompts.get(model_type, "You are an AI assistant specialized in anomaly detection.")
    user_prompt = user_prompts.get((model_type, question_type), "Provide detailed insights based on the given data.")
    
    return system_prompt, user_prompt
