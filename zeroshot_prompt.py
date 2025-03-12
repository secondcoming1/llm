def ZeroShotExplain():
    return  """
        Im providing a negative outcome from a {ML-system} and your task provide obeservations over the provided Insider Anomaly flagged by a LSTM model by comparing them to Counterfactuals generated from the Anomaly. 
        ----- Anomolous outcome -----
        {negative_outcome}

        
        ----- Positive couterfactual outcome -----
        {positive_outcome}


        ----- Observations -----
        <List of Observations>
        """



################### Zero Shot Prompts #############################
def ZeroShotRules():
    return  """
        Im providing a negative outcome from a {ML-system} and your task is to extract the most important observed rules based on a set of counterfactual cases. 
        ----- Negative assessment outcome -----
        {negative_outcome}

        ----- Positive couterfactual outcome -----
        {positive_outcome}

        ----- Rules -----
        <List of Rules>
        """

def ZeroShotRulesCode():
    return  """
        Im providing a negative outcome from a {ML-system}, a set of counterfactual cases that flip the decision of the system and the main rules inferred from the counterfactuals.
        You should generate python code to count how many of the counterfactuals are consistent with the rule. The code should create a df with the counterfactuals provided and then check for each rule how many of them follow the rules. Order the rules. Finally, you should print the results.
 
        ----- Negative assessment outcome -----
        {negative_outcome}

        ----- Positive couterfactual outcome -----
        {positive_outcome}

        ----- Rules -----
        {rules}
        
        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}
        
        ----- Code -----
        ```
        import pandas as pd
        #complete this code
        ```
        """ 

def ZeroShotExplanation(user_input = False):
    if user_input:
        return  """
            A person has been classified in the negative class of {ML-system}. The data is the following:
            ----- Negative assessment outcome -----
            {negative_outcome}

            ----- Positive couterfactual outcome -----
            {positive_outcome}

            ----- Rules -----
            By generating counterfactuals, we obtained the following rules:
            {rules}


            ----- Results -----
            We have checked that the rules are followed by n counterfactuals:
            {results}

            ----- Dataset info -----
            The following info about the dataset is available:
            {dataset_info}

            ----- Explanation -----
            Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. Furthermore, an expert user has said that the most relevant rules are {user_input}
            <explanation>
            """
    else:
        return  """
            A person has been classified in the negative class of {ML-system}. The data is the following:
            ----- Negative assessment outcome -----
            {negative_outcome}

            ----- Positive couterfactual outcome -----
            {positive_outcome}

            ----- Rules -----
            By generating counterfactuals, we obtained the following rules:
            {rules}


            ----- Results -----
            We have checked that the rules are followed by n counterfactuals:
            {results}

            ----- Dataset info -----
            The following info about the dataset is available:
            {dataset_info}

            ----- Explanation -----
            Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
            <explanation>
            """

def ZeroShotExample():
    return """
        A person has been classified in the negative class of {ML-system}. The data is the following:
        ----- Negative assessment outcome -----
        {negative_outcome}

        
        ----- Explanation -----
        The following explanation was given inorder to try and change the class.
        {explanation}


        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}


        ----- Example -----
        Given this information, provide an example in the format of a pandas dataframe that would be in the positive class. Complete the code below and note that it is very important to use the name 'temp_csv.csv', since later processes rely on it.
        
        ```
        import pandas as pd
        df = pd.DataFrame(...) #complete this line
        df.to_csv('temp_csv.csv', index = False)

        ```
        """

def ZeroShotExampleCode():
    return  """
    Im providing a negative outcome from a {ML-system}. A counterfactual example in the format os a single row dataframe was created in temp_csv from the rules that are also provided. Give some code to check the number of rules followed by the example. The result must be given in the format of a dataframe and saved as a csv. The dataframe must have columns 'Rule' with the text of the rule, 'Importance' with the number of counterfactuals follow each rule, and 'In explanation' (1 or 0) depending if the final example follows the explanation or not. It is very important to save the csv as 'evaluation.csv'.
    
    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Rules -----
    {rules}

    ----- Results -----
    We have checked that the rules are followed by n counterfactuals:
    {results}
    
    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}

    ----- Code -----
    ```
    import pandas as pd
    df = pd.read_csv('temp_csv.csv')

    #COMPLETE CODE

    # Save to csv
    df_final.to_csv('evaluation.csv', index = False)
    ```
    """ 