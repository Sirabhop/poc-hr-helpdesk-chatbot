# Upload and preprocess the DataFrame
def preprocess_dataframe(df):
    """Converts a DataFrame into a mapping of questions to answers."""
    mapping = {}
    for idx, row in df.iterrows():
        mapping[row["core_question"].strip()] = row["core_answer"].strip()
    return mapping