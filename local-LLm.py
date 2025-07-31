"""
Local LLM-Powered Data Assistant for CSVs (Fee + Residential)
- Uses LangChain + Ollama + Pandas
- Supports separate agents and helpers for each dataset
- Now supports smart routing based on columns mentioned in the question

Before running:
    pip install -U pandas langchain langchain-community langchain-experimental
Start model:
    ollama run llama3
"""

import pandas as pd
from langchain_ollama import OllamaLLM as Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent

# ─────────────────────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────────────────────

fee_df = pd.read_csv("Fee.csv", dtype=str, low_memory=False)
res_df = pd.read_csv("Residential.csv", dtype=str, low_memory=False)

for df in [fee_df, res_df]:
    df.columns = [col.strip().upper() for col in df.columns]
    if 'VISITDT' in df.columns:
        df['VISITDT'] = pd.to_datetime(df['VISITDT'], errors='coerce')
    if 'BIRTHDATE' in df.columns:
        df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
    if 'STARTINGDATE' in df.columns and 'ENDINGDATE' in df.columns:
        df['STARTINGDATE'] = pd.to_datetime(df['STARTINGDATE'], errors='coerce')
        df['ENDINGDATE'] = pd.to_datetime(df['ENDINGDATE'], errors='coerce')

# ─────────────────────────────────────────────────────────
# Step 2: Load Dictionary
# ─────────────────────────────────────────────────────────

dict_df = pd.read_csv("dictionary.csv")
column_descriptions = "\n".join(
    f"{row['Abbreviation']}: {row['Description']}" for _, row in dict_df.iterrows()
)

# ─────────────────────────────────────────────────────────
# Step 3: System Prompt
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are a helpful data assistant analyzing human services data.

You can:
- Filter, summarize, and sort columns
- Extract information from Fee or Residential datasets
- Answer date range, most frequent value, NA counts, and similar questions

Column info:
{column_descriptions}
"""

# ─────────────────────────────────────────────────────────
# Step 4: Helper Functions
# ─────────────────────────────────────────────────────────

def count_unique_values(df, column):
    return f"✅ Number of unique values in '{column}': {df[column].nunique()}"

def count_rows_in_date_range(df, column, start_date, end_date):
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return f"✅ Rows between {start_date} and {end_date}: {df[(df[column] >= start_date) & (df[column] <= end_date)].shape[0]}"

def get_longest_detainment(df):
    df['STARTINGDATE'] = pd.to_datetime(df['STARTINGDATE'], errors='coerce')
    df['ENDINGDATE'] = pd.to_datetime(df['ENDINGDATE'], errors='coerce')
    df['detainment_days'] = (df['ENDINGDATE'] - df['STARTINGDATE']).dt.days
    max_row = df.loc[df['detainment_days'].idxmax()]
    return f"Client {max_row['CLIENTNUMBER']} had the longest detainment of {max_row['detainment_days']} days."


def most_common_value(df, column):
    return f"✅ Most common value in '{column}': {df[column].mode()[0]}"

def count_felonies_in_birth_range(df, birth_col, crime_col, start, end):
    df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")
    filtered = df[(df[birth_col] >= start) & (df[birth_col] <= end)]
    return f"✅ Felony count in range: {filtered[filtered[crime_col].str.lower() == 'felony'].shape[0]}"

def add_age_column(df, birth_col="BIRTHDATE"):
    df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")
    today = pd.Timestamp.today()
    df["AGE"] = (today - df[birth_col]).dt.days // 365
    return df

def calculate_average_age(df, birth_col="BIRTHDATE"):
    df[birth_col] = pd.to_datetime(df[birth_col], errors="coerce")
    today = pd.Timestamp.today()
    df["AGE"] = (today - df[birth_col]).dt.days // 365
    return f"✅ Average age: {df['AGE'].mean():.2f} years"


def sort_by_detainment_length(df, start_col, end_col):
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    df["DURATION_DAYS"] = (df[end_col] - df[start_col]).dt.days
    return df.sort_values(by="DURATION_DAYS", ascending=False)

def sort_by_column(df, col):
    return df.sort_values(by=col)

def count_na_values(df, col):
    return f"✅ Number of missing values in '{col}': {df[col].isna().sum()}"

def extract_signer_names(df, col="LAST SIGNATURE"):
    return df[col].dropna().unique().tolist()

def get_most_common_name(df, column='FIRSTNAME'):
    try:
        if column not in df.columns:
            return f"❌ Column '{column}' not found in the dataset."
        if df[column].isnull().sum() == len(df):
            return f"⚠️ All values in '{column}' are missing."
        return f"The most common name in column '{column}' is: {df[column].mode()[0]}"
    except Exception as e:
        return f"❌ Error: {e}"

def get_dataframe_for_column(column):
    column = column.upper()
    in_fee = column in fee_df.columns
    in_res = column in res_df.columns

    if in_fee and not in_res:
        return "fee", fee_df, fee_agent
    elif in_res and not in_fee:
        return "residential", res_df, res_agent
    elif in_fee and in_res:
        return None, None, None  # exists in both
    else:
        return "none", None, None

# ─────────────────────────────────────────────────────────
# Step 5: LLM + LangChain Agents
# ─────────────────────────────────────────────────────────

llm = Ollama(model="llama3", temperature=0)

fee_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=fee_df,
    verbose=True,
    allow_dangerous_code=True,
    prefix=SYSTEM_PROMPT,
)

res_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=res_df,
    verbose=True,
    allow_dangerous_code=True,
    prefix=SYSTEM_PROMPT,
)

# ─────────────────────────────────────────────────────────
# Step 6: Interactive Prompt with Smart Routing
# ─────────────────────────────────────────────────────────

print("\n🔍 Ask about the Fee or Residential dataset (type 'exit' to quit)...\n")

while True:
    q = input(">>> ").strip()
    if q.lower() in ["exit", "quit"]:
        break

    dataset_input = input("Which dataset? (fee / residential): ").strip().lower()
    columns = list(set(fee_df.columns.tolist() + res_df.columns.tolist()))
    col_in_question = next((col for col in columns if col.lower() in q.lower()), None)

    if col_in_question:
        resolved_data, df, agent = get_dataframe_for_column(col_in_question)
        if resolved_data == "none":
            print(f"❌ Column '{col_in_question}' not found in either dataset.")
            continue
        elif resolved_data is None:
            df = fee_df if dataset_input == "fee" else res_df
            agent = fee_agent if dataset_input == "fee" else res_agent
        else:
            print(f"✅ Using '{resolved_data}' dataset because it contains the column '{col_in_question}'.")
    else:
        df = fee_df if dataset_input == "fee" else res_df
        agent = fee_agent if dataset_input == "fee" else res_agent

    try:
        if "unique" in q and "column" in q:
            col = input("Column name: ").strip().upper()
            print(count_unique_values(df, col), "\n")

        elif "rows" in q and "date range" in q:
            col = input("Date column: ").strip().upper()
            start = input("Start date (YYYY-MM-DD): ")
            end = input("End date (YYYY-MM-DD): ")
            print(count_rows_in_date_range(df, col, start, end), "\n")

        elif "average age" in q:
            print(calculate_average_age(df), "\n")

        elif "longest detainment" in q:
            print(get_longest_detainment(df))

        elif "most common name" in q or "most frequent name" in q:
            col = input("Column name: ").strip()
            print(get_most_common_name(df, col), "\n")

        elif "common" in q:
            col = input("Column name: ").strip().upper()
            print(most_common_value(df, col), "\n")

        elif "felony" in q and "birth" in q:
            start = input("Start birthdate (YYYY-MM-DD): ")
            end = input("End birthdate (YYYY-MM-DD): ")
            print(count_felonies_in_birth_range(df, "BIRTHDATE", "CRIMECLASSDESC", start, end), "\n")

        elif "convert" in q and "age" in q:
            df = add_age_column(df)
            print("✅ Age column added.\n")

        elif "sort" in q and "length" in q:
            df = sort_by_detainment_length(df, "STARTINGDATE", "ENDINGDATE")
            print("✅ Sorted by detainment length.\n")

        elif "sort" in q:
            col = input("Sort by column: ").strip().upper()
            df = sort_by_column(df, col)
            print("✅ Sorted by column.\n")

        elif "na" in q or "missing" in q:
            col = input("Column name: ").strip().upper()
            print(count_na_values(df, col), "\n")

        elif "signer" in q:
            print("✅ Signers:\n", extract_signer_names(df), "\n")

        else:
            print(agent.run(q), "\n")

    except Exception as e:
        print(f"[Error] {e}\n")

