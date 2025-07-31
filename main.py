import google.generativeai as genai
import fitz
import io
import pandas as pd
from typing import List, Any, Dict
import re
from PIL import Image
import os
import json
from dateutil.parser import parse
from datetime import datetime,date, timedelta
import requests
import time
import json
import re
from dotenv import load_dotenv
import concurrent.futures
import glob
from concurrent.futures import ProcessPoolExecutor


# Load environment variables from .env file
load_dotenv()

def get_api_key():
    """Fetches the API key from environment variables and validates it."""
    api_key = "AIzaSyC4zktO8iLL8l68ORkf3D_deUbuNjZ-GNw"
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please check your environment configuration.")
    return api_key


def extract_json_object(response_text: str) -> dict | None:
    """
    Extracts a JSON object from a string enclosed in Markdown ```json fences.

    Args:
        response_text: The raw string response which may contain a JSON object.

    Returns:
        A Python dictionary representing the parsed JSON, or None if not found.
    """
    # Pattern to find a JSON object within ```json ... ```
    # re.DOTALL makes '.' match newlines as well.
    match = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)

    if not match:
        print("Error: No JSON object found in the Markdown code block.")
        return None

    json_str = match.group(1)

    try:
        # Convert the JSON string to a Python dictionary
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def get_accounts_types(pdf_path: str,csv_path: str) -> str | None:
    """Extracts tables with holdings data from a PDF."""
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash',generation_config={'temperature': 0})
        holdings_pdf_file = genai.upload_file(pdf_path)
        account_definitions_csv_file = genai.upload_file(csv_path)
        response = model.generate_content(
            contents=[
                """
                You are an expert financial analyst AI. Your task is to analyze a financial holdings document and classify the investment accounts listed within it, using a provided list of account type definitions.
                **Instructions:**
                1.  **Analyze the Holdings Document (Input 1 - PDF):**
                    *If the document is *not a Portfolio Holdings document* please reply with only *'This is not a Portfolio Holdings Document'*
                2. **If it have portfolio holdings tables**:
                        * Scan the document to identify each unique Account Number.
                        *Crucially, you must use the Account Number listed within each individual table. This is the ID the fund company uses to track specific fund positions.
                        *Do not use the Plan Number.
                        * For each account, classify the type with the provided Definitions CSV file
                3.  **Generate a Formatted Output:**
                    * The final output must be a single, valid JSON object.
                    * Each object inside the "accounts" list must represent one account and contain the following keys:
                        "account_number": The account number identified from the source document. If there's No account Number Please State 'None'.
                        "account_type": The account's type NAME from the provided csv file and not the abbreviation.
                *If an account type identified in the PDF cannot be found in the definitions file, clearly state 'None'*
                """,
                # This is where you would pass the variable containing your PDF file
                holdings_pdf_file,
                # This is where you would pass the variable containing your account definitions file
                account_definitions_csv_file
            ]
        )
        if 'This is not a Portfolio Holdings Document' in response.text:
            return 'This is not a Portfolio Holdings Document'
        else:
            return extract_json_object(response.text)

    except Exception as e:
        print(f"An error occurred during table extraction: {e}")
        return None

    # --- Step 1: Date Extraction Methods ---
def format_date_mm_dd_yy(date_string: str) -> str | None:
    """Parses and formats a date string to MM/DD/YYYY."""
    try:
        date_format = parse(date_string)
        return date_format.strftime("%m/%d/%Y")
    except (ValueError, TypeError):
        return None
def extract_date_from_json(json_string: str) -> str | None:
    """Extracts a date from a JSON string."""
    try:
        json_content = re.search(r'```json\n(.*)\n```', json_string, re.DOTALL).group(1)
        data = json.loads(json_content)
        date = data.get("Date")
        return format_date_mm_dd_yy(date)
    except (AttributeError, json.JSONDecodeError, KeyError):
        return None

def get_portfolio_date(pdf_path: str) -> str | None:
    """Extracts the document date from the first page of a PDF."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash',generation_config={'temperature': 0})

    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)
        pix = page.get_pixmap()
        image = Image.open(io.BytesIO(pix.tobytes("png")))

        prompt_parts = [
            "From the following document identify and extract the Date of the Document and not the Date of The statement.",
            "If the date is a period get extract only the last date of the period"
            "Return the results in a JSON format with the following structure:",
            '{"Date": "Date Value"}',
            image
        ]
        response = model.generate_content(prompt_parts)
        response.resolve()
        return extract_date_from_json(response.text)
    except Exception as e:
        print(f"An error occurred while getting the portfolio date: {e}")
        return None

def extract_tables_from_pdf_multiple_accounts(pdf_path: str) -> str | None:
    """Extracts tables with holdings data from a PDF."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro',generation_config={'temperature': 0})
    try:
        sample_pdf = genai.upload_file(pdf_path)
        response = model.generate_content(
            contents=[
                """
                You are a highly skilled financial data extraction specialist. Your task is to extract information from tables within a portfolio holdings document.
                **Overall Instructions:**
                1. **Table Extraction:** Focus exclusively on extracting tables that contain portfolio holdings data.  You must explicitly ignore and do not extract any tables that show account activity, transaction histories, or statements. For the portfolio holdings tables that you do extract, ensure that all rows and columns are captured completely and accurately. Include the "Total" and "Total Portfolio" rows or any summary values if they are present within the table.
                    - if there's MISALIGNEMENT between the SYMBOL and NAME in the table please correct it.

                2. **Currency Identification and Column Creation:**
                    - For each table, determine the currency or currencies used.
                    - Add a new column labeled "Currency" to each table.
                    - Populate the "Currency" column with the appropriate currency abbreviation (e.g., "USD", "CAD", "EUR"). Do not use currency symbols (e.g., "$", "€").
                    - If a table uses only one currency, apply that abbreviation to all rows in the "Currency" column.
                    - If a table uses multiple currencies:
                    - Identify which currency applies to each row.
                    - Populate the "Currency" column accordingly for each row.
                    - If the currency is not explicitly stated for a row, populate the "Currency" column with "None" for that row.

                3. **Account Number Identification and Column Handling:**
                    - Check for an existing "Account Number" column in the source table.
                    - If an "Account Number" column already exists, use that column. Make sure it is correctly filled for all rows. Do not add a new column.
                    - If an "Account Number" column does NOT exist :
                        - For each table, identify the associated account number as in the portfolio, if present.
                        - Add a new column labeled "Account Number" to each table.
                        - Populate the "Account Number" column with the corresponding account number.
                        - If no account number is found for a table, populate the "Account Number" column with "None" for all rows in that table.

                4. **Markdown Output:** Present the extracted tables in Markdown format. Ensure that the Markdown tables are well-formatted and easy to read. Include all extracted data, including the "Currency" and "Account Number" columns.
                """,
                sample_pdf
            ]
        )
        return response.text
    except Exception as e:
        print(f"An error occurred during table extraction: {e}")
        return None

def extract_tables_from_pdf_one_account(pdf_path: str) -> str | None:
    """Extracts tables with holdings data from a PDF."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro',generation_config={'temperature': 0})
    try:
        sample_pdf = genai.upload_file(pdf_path)
        response = model.generate_content(
            contents=[
                """
                You are a highly skilled financial data extraction specialist. Your task is to extract information from tables within a portfolio holdings document.
                **Overall Instructions:**
                1. **Table Extraction:** Focus exclusively on extracting tables that contain portfolio holdings data.  You must explicitly ignore and do not extract any tables that show account activity, transaction histories, or statements. For the portfolio holdings tables that you do extract, ensure that all rows and columns are captured completely and accurately. Include the "Total" and "Total Portfolio" rows or any summary values if they are present within the table.
                    - if there's MISALIGNEMENT between the SYMBOL and NAME in the table please correct it

                2. **Currency Identification and Column Creation:**
                    - For each table, determine the currency or currencies used.
                    - Add a new column labeled "Currency" to each table.
                    - Populate the "Currency" column with the appropriate currency abbreviation (e.g., "USD", "CAD", "EUR"). Do not use currency symbols (e.g., "$", "€").
                    - If a table uses only one currency, apply that abbreviation to all rows in the "Currency" column.
                    - If a table uses multiple currencies:
                    - Identify which currency applies to each row.
                    - Populate the "Currency" column accordingly for each row.
                    - If the currency is not explicitly stated for a row, populate the "Currency" column with "None" for that row.

                3. **Markdown Output:** Present the extracted tables in Markdown format. Ensure that the Markdown tables are well-formatted and easy to read. Include all extracted data, including the "Currency" column.
                """,
                sample_pdf,
            ]
        )
        return response.text
    except Exception as e:
        print(f"An error occurred during table extraction: {e}")
        return None
def get_asset_classes(response_tables: str) -> str | None:
    """Identifies and adds an "Asset Class" column to the extracted tables."""
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro',generation_config={'temperature': 0})
    try:
        response = model.generate_content(
            contents=[
                """
                You are a highly skilled financial data specialist. Your task is to extract information from tables within a portfolio holdings document.
                1.  **Asset Class Identification and Column Creation:**
                        -   For each row in each table, identify the Asset Class from *ONLY* this following list:
                            -Government Bonds
                            -Corporate Bonds
                            -Municipal Bonds
                            -Agency Bonds
                            -Convertible Bonds
                            -Mortgage-Backed Securities
                            -Asset-Backed Securities
                            -Inflation-Linked Bonds
                            -Floating Rate Notes
                            -Structured Notes
                            -Common Stock
                            -Preferred Shares
                            -Exchange-Traded Funds
                            -American Depository Receipts
                            -Government Depository Receipts
                            -Mutual Funds
                            -Futures
                            -Options
                            -Swaps :
                                -Interest Rate Swaps
                                -Credit Default Swaps
                                -Total Return Swaps
                            -Cross Currency Swaps
                            -Swaptions
                            -Forward Rate Agreements
                            -Cash
                            -Certificates of Deposit
                            -Guaranteed Investment Certificates
                            -Commercial Paper
                            -Treasury Bills
                            -Repurchase Agreements
                            -Spot FX
                            -FX Forwards
                            -Non-Deliverable Forwards
                            -FX Options
                            -Private Equity
                            -Hedge Funds
                            -Real Estate
                            -Infrastructure Investments
                            -Physical Commodities
                            -Commodity Derivatives
                            -Collateralized Debt Obligations
                            -Collateralized Loan Obligations
                            -Structured Credit Instruments
                        -   Add a new column labeled "Asset Class L2" to each table.
                        -   Populate the "Asset Class L2" column with the identified asset class for each row. If no asset class can be determined for a row, populate it with "None".
                2.  **Markdown Output:** Present the extracted tables in Markdown format. Ensure that the Markdown tables are well-formatted and easy to read. Include all extracted data, including the "Currency", "Account Number", and "Asset Class" columns.
                    """,
                response_tables
            ]
        )
        return response.text
    except Exception as e:
        print(f"An error occurred while getting asset classes: {e}")
        return None

def add_asset_class_l1(list_of_dfs: List[pd.DataFrame], asset_class_csv_path: str) -> List[pd.DataFrame]:
    """
    Adds an 'Asset Class L1' column by mapping the 'Asset Class' (L2) using a CSV file.

    Args:
        list_of_dfs: The list of DataFrames to process.
        asset_class_csv_path: Path to the CSV file with 'L1' and 'L2' columns for mapping.

    Returns:
        A list of DataFrames with the new 'Asset Class L1' column added.
    """
    try:
        # Read the mapping file
        mapping_df = pd.read_csv(asset_class_csv_path)

        # Check if required columns exist in the mapping file
        if 'L1' not in mapping_df.columns or 'L2' not in mapping_df.columns:
            print("Error: The asset class mapping CSV must contain 'L1' and 'L2' columns.")
            return list_of_dfs

        # Create a lookup dictionary from L2 (Asset Class) to L1
        l1_lookup = pd.Series(mapping_df.L1.values, index=mapping_df.L2).to_dict()

        processed_dfs = []
        for df in list_of_dfs:
            # Check if the source 'Asset Class' column exists in the DataFrame
            if 'Asset Class L2' in df.columns:
                # Map the 'Asset Class' column to the lookup dictionary to create the new L1 column
                # Use .fillna() to handle any L2 classes that aren't in the mapping file
                #df.rename(columns={'Asset Class': 'Asset Class L2'}, inplace=True)
                df['Asset Class L1'] = df['Asset Class L2'].map(l1_lookup).fillna('None')
            else:
                # If there's no 'Asset Class' column, add the L1 column with a default value
                df['Asset Class L1'] = 'None'
            processed_dfs.append(df)

        return processed_dfs

    except FileNotFoundError:
        print(f"Error: Asset class mapping file not found at {asset_class_csv_path}")
        return list_of_dfs
    except Exception as e:
        print(f"An error occurred while adding Asset Class L1: {e}")
        return list_of_dfs

def extract_markdown_tables(ocr_response) -> List[pd.DataFrame]:
    """Extracts all markdown tables from an OCR response string and converts them to a list of pandas DataFrames.
    Handles inconsistencies in column counts, cleans headers and cells, and normalizes symbol columns."""
    markdown_content = ocr_response # Assuming ocr_response is the markdown string directly
    table_groups = []
    current_table = []
    for line in markdown_content.splitlines():
        if line.strip().startswith("|"):
            current_table.append(line.strip())
        else:
            if current_table:
                table_groups.append(current_table)
                current_table = []
    if current_table: # Add the last table if the content ends with it
        table_groups.append(current_table)

    if not table_groups:
        print("Warning: No markdown tables found in the extracted markdown content.")
        return []

    def parse_table(table_lines: List[str]) -> pd.DataFrame:
        """
        Converts a list of markdown table lines to a pandas DataFrame.
        Handles inconsistent column counts by padding rows.
        Cleans headers and data cells by removing '*' and '**'.
        """
        rows = []
        for line in table_lines:
            line = line.strip('|') # Remove leading/trailing pipe characters from the line
            cells = [cell.strip() for cell in line.split('|')]
            rows.append(cells)

        if len(rows) < 2: # Header + Separator
            # Returning an empty DataFrame or specific error object might be more robust
            # than raising ValueError if this can be a semi-valid state handled later.
            print(f"Warning: Table does not have enough rows for header and separator. Found {len(rows)} rows. Skipping this table. Table lines: {table_lines[:2]}")
            return pd.DataFrame() # Return an empty DataFrame

        # Clean header: remove '**' and '*' then strip whitespace
        header = [col.replace('**', '').replace('*', '').replace('($)','').strip() for col in rows[0]]

        data_rows_raw = rows[2:]

        max_cols = len(header)
        padded_data = []
        for i, row_cells in enumerate(data_rows_raw):
            cleaned_row = [cell.replace('**', '').replace('*', '').strip() for cell in row_cells]

            if len(cleaned_row) < max_cols:
                cleaned_row.extend([''] * (max_cols - len(cleaned_row)))
            elif len(cleaned_row) > max_cols:
                cleaned_row = cleaned_row[:max_cols]
            padded_data.append(cleaned_row)

        # Create DataFrame, ensuring header is used even if padded_data is empty
        if not padded_data and header:
            return pd.DataFrame(columns=header)
        elif not header and not padded_data: # both empty
            return pd.DataFrame()
        return pd.DataFrame(padded_data, columns=header)

    dataframes = []
    for i, table_lines in enumerate(table_groups):
        try:
            df = parse_table(table_lines)
            if df.empty: # If parse_table returned an empty DataFrame, skip further processing for it
                # print(f"Info: Table {i+1} was parsed as empty. Skipping symbol normalization.")
                dataframes.append(df) # Still append the empty DF if that's desired behavior
                continue

            # --- Add Symbol Normalization Step ---
            # Try to find a symbol column (e.g., 'Symbol', 'symbol', 'Ticker')
            # The actual column name will depend on your markdown table headers after cleaning.
            symbol_col_to_normalize = None
            possible_symbol_column_names = ['symbol', 'ticker'] # Add other common names if needed

            for col_name in df.columns:
                if col_name.strip().lower() in possible_symbol_column_names:
                    symbol_col_to_normalize = col_name
                    break # Found a symbol column

            if symbol_col_to_normalize:
                # print(f"Normalizing symbol column '{symbol_col_to_normalize}' in table {i+1}...")
                # Ensure the column is treated as string type before applying lstrip
                df[symbol_col_to_normalize] = df[symbol_col_to_normalize].astype(str).str.lstrip('.')
            # --- End Symbol Normalization Step ---

            dataframes.append(df)
        except ValueError as ve: # Catch specific errors from parse_table if needed
            print(f"Skipping table {i+1} due to parsing error: {ve}. Table content (first 3 lines): {table_lines[:3]}")
        except Exception as e:
            # Provide more context for the skipped table
            print(f"Skipping table {i+1} due to an unexpected error: {e}. Table content (first 3 lines): {table_lines[:3]}")
    return dataframes

def process_tables_individually(ocr_response: str, portfolio_date: Any) -> pd.DataFrame | None:
    """Processes OCR response, creates a combined DataFrame, and adds date."""
    dfs_from_markdown = extract_markdown_tables(ocr_response)
    date_string_for_df = None # This will store the date as 'MM/DD/YYYY' string

    if pd.notna(portfolio_date): # Checks for None, NaN, NaT
        if isinstance(portfolio_date, str):
            # Use the provided function to parse and format the string
            date_string_for_df = format_date_mm_dd_yy(portfolio_date)
        elif isinstance(portfolio_date, datetime): # Handles datetime.datetime
            date_string_for_df = portfolio_date.strftime("%m/%d/%Y")
        elif isinstance(portfolio_date, date):   # Handles datetime.date
            date_string_for_df = portfolio_date.strftime("%m/%d/%Y")
        else:
            # For other types (e.g., pd.Timestamp, np.datetime64, numbers representing dates),
            # try to convert to pandas Timestamp and then format.
            try:
                # pd.to_datetime is robust and can handle various inputs
                dt_object = pd.to_datetime(portfolio_date)
                if pd.notna(dt_object):  # Check if conversion was successful (not NaT)
                    date_string_for_df = dt_object.strftime("%m/%d/%Y")
                # else:
                    # print(f"Warning: date_Prtofolio_input '{date_Prtofolio_input}' (type: {type(date_Prtofolio_input)}) converted to NaT by pd.to_datetime.")
            except Exception:
                # print(f"Warning: Could not convert date_Prtofolio_input '{date_Prtofolio_input}' (type: {type(date_Prtofolio_input)}) to a recognizable date format.")
                pass # date_string_for_df remains None

    processed_dfs = []
    for df_original in dfs_from_markdown:
        df = df_original.copy()
        if not df.empty:
            # Remove existing "Date" column to ensure new one is at the desired position and format
            if "Date" in df.columns:
                del df["Date"]

            if date_string_for_df is not None:
                # Insert the formatted date string at the beginning of the DataFrame
                df.insert(0, "Date", date_string_for_df)
            # else:
                # print("Warning: date_string_for_df is None or invalid, 'Date' column not added reliably.")
        processed_dfs.append(df)
    return processed_dfs

# --- Step 5: Currency Conversion Methods (Integrated) ---
def market_value_str_to_float(market_value_str: Any) -> float | None:
    """Converts a market value string to a float."""
    if pd.isna(market_value_str):
        return None
    if isinstance(market_value_str, (int, float)):
        return float(market_value_str)
    if not isinstance(market_value_str, str):
        return None

    cleaned_str = market_value_str.strip()
    if ',' in cleaned_str and '.' in cleaned_str:
        if cleaned_str.rfind('.') < cleaned_str.rfind(','):
            cleaned_str = cleaned_str.replace('.', '').replace(',', '.')
        else:
            cleaned_str = cleaned_str.replace(',', '')
    elif ',' in cleaned_str:
        cleaned_str = cleaned_str.replace(',', '.') if re.search(r",\d{1,2}$", cleaned_str) else cleaned_str.replace(',', '')

    cleaned_str = re.sub(r"[^\d\.\-]", "", cleaned_str)
    try:
        return float(cleaned_str) if cleaned_str else None
    except ValueError:
        return None

def get_historical_exchange_rate_boc(base_currency, target_currency, date_obj):
    """
    Fetches the historical exchange rate using the Bank of Canada Valet API.
    Includes a fallback to find the latest observation within a 7-day window.
    """
    # 1. Input Validation
    if pd.isna(base_currency) or pd.isna(target_currency) or pd.isna(date_obj):
        return None

    base = str(base_currency).strip().upper()
    target = str(target_currency).strip().upper()

    if not base or not target: return None
    if base == target: return 1.0

    # 3. Format Date
    date_str = date_obj.strftime("%Y-%m-%d")
    requested_date_obj_dt = pd.to_datetime(date_str)

    # 4. Determine Required BoC Series
    series_needed = []
    if base != 'CAD': series_needed.append(f"FX{base}CAD")
    if target != 'CAD': series_needed.append(f"FX{target}CAD")
    if not series_needed:
        # This handles the case where one is CAD and the other isn't, but no series were added.
        # It's an edge case, but good to have.
            if base == 'CAD' and target != 'CAD': series_needed.append(f"FX{target}CAD")
            elif target == 'CAD' and base != 'CAD': series_needed.append(f"FX{base}CAD")
            else: return None # Should not happen if base != target

    series_needed = list(set(series_needed))
    series_string = ",".join(series_needed)

    # 5. Build and Call API
    api_url = f"https://www.bankofcanada.ca/valet/observations/{series_string}/json"
    response = None
    obs_data_to_use = None

    try:
        # --- Primary Attempt: Get rate for the exact date ---
        params = {'start_date': date_str, 'end_date': date_str}
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data_api = response.json()
        obs_list = data_api.get('observations')

        if obs_list:
            obs_data_to_use = obs_list[0]
        else:
            # --- Fallback Attempt: If no data, look back up to 7 days ---
            print(f"Info: No BoC data for {date_str}. Looking back up to 7 days.")
            fallback_start_date = (date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
            params_fallback = {'start_date': fallback_start_date, 'end_date': date_str}
            response = requests.get(api_url, params=params_fallback, timeout=15)
            response.raise_for_status()
            data_fallback = response.json()
            fallback_obs_list = data_fallback.get('observations')
            if fallback_obs_list:
                obs_data_to_use = fallback_obs_list[-1] # Get the most recent one
                print(f"Info: Using BoC data from {obs_data_to_use['d']} for requested date {date_str}.")

        if not obs_data_to_use:
            print(f"Warning: No BoC observations found for {series_string} on or before {date_str}.")
            return None

        # 6. Parse Response and Extract Rates
        rates = {}
        for s_key in series_needed:
            if s_key in obs_data_to_use and 'v' in obs_data_to_use[s_key]:
                rates[s_key] = float(obs_data_to_use[s_key]['v'])
            else:
                return None # A required rate is missing

        # 7. Calculate Final Rate
        rate_base_cad = rates.get(f"FX{base}CAD")
        rate_target_cad = rates.get(f"FX{target}CAD")

        if base == 'CAD':
            if not rate_target_cad or rate_target_cad == 0: return None
            return 1.0 / rate_target_cad
        elif target == 'CAD':
            return rate_base_cad
        else:
            if not rate_base_cad or not rate_target_cad or rate_target_cad == 0: return None
            return rate_base_cad / rate_target_cad

    except requests.exceptions.RequestException as e:
        print(f"ERROR: BoC API request failed for {series_string} on {date_str}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        print(f"ERROR: Failed to parse BoC API response for {series_string} on {date_str}: {e}")
    return None


import pandas as pd
import time
from typing import List, Optional, Any

# --- Helper Functions (re-used from before) ---
def market_value_str_to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace('$', '').replace(',', '').strip())
        except (ValueError, TypeError): return None
    return None

def get_historical_exchange_rate_boc(currency: str, date: pd.Timestamp) -> Optional[float]:
    if currency == 'CAD': return 0.75
    if currency == 'USD': return 1.0
    return None

# --- Main Updated Function ---

def add_usd_market_value(list_of_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Adds USD market value columns with prioritized searching.

    1. It first searches for a column named exactly 'Market Value' (case-insensitive).
    2. If not found, it falls back to searching for any column containing 'market value'.
    """
    processed_dfs = []
    for df_idx, df in enumerate(list_of_dfs):
        print(f"\n[DEBUG] Processing DataFrame {df_idx+1}/{len(list_of_dfs)}")
        df_copy = df.copy()

        # --- New Prioritized Search Logic ---
        market_value_cols_to_process = []

        # 1. First, look for an exact, case-insensitive match like 'Market Value' or 'market value'.
        exact_match = [col for col in df_copy.columns if col.lower() == 'market value']
        print(f"[DEBUG] exact_match columns: {exact_match}")

        if exact_match:
            # If an exact match is found, use only that column/s.
            market_value_cols_to_process = exact_match
        else:
            # 2. If no exact match, fall back to finding any column containing the phrase.
            market_value_cols_to_process = [col for col in df_copy.columns if 'market value' in col.lower()]
        print(f"[DEBUG] market_value_cols_to_process: {market_value_cols_to_process}")

        # Find date and currency columns
        date_col = next((col for col in df_copy.columns if col.lower() == 'date'), None)
        currency_col = next((col for col in df_copy.columns if col.lower() == 'currency'), None)
        print(f"[DEBUG] date_col: {date_col}, currency_col: {currency_col}")

        # If duplicate columns exist, select the first one
        if date_col is not None and isinstance(df_copy[date_col], pd.DataFrame):
            date_col_idx = df_copy.columns.get_loc(date_col)
            date_col = df_copy.columns[date_col_idx]
            print(f"[DEBUG] date_col was duplicate, using first: {date_col}")
        if currency_col is not None and isinstance(df_copy[currency_col], pd.DataFrame):
            currency_col_idx = df_copy.columns.get_loc(currency_col)
            currency_col = df_copy.columns[currency_col_idx]
            print(f"[DEBUG] currency_col was duplicate, using first: {currency_col}")

        print(f"[DEBUG] type(date_col): {type(date_col)}, value: {date_col}")
        print(f"[DEBUG] type(currency_col): {type(currency_col)}, value: {currency_col}")
        print(f"[DEBUG] type(market_value_cols_to_process): {type(market_value_cols_to_process)}, value: {market_value_cols_to_process}")
        if isinstance(date_col, pd.Index):
            print(f"[DEBUG] date_col is Index, contents: {list(date_col)}")
        if isinstance(currency_col, pd.Index):
            print(f"[DEBUG] currency_col is Index, contents: {list(currency_col)}")
        if isinstance(market_value_cols_to_process, pd.Index):
            print(f"[DEBUG] market_value_cols_to_process is Index, contents: {list(market_value_cols_to_process)}")

        if (
            date_col is None or
            currency_col is None or
            not isinstance(market_value_cols_to_process, list) or
            len(market_value_cols_to_process) == 0 or
            isinstance(date_col, pd.Index) or
            isinstance(currency_col, pd.Index)
        ):
            print(f"[DEBUG] Skipping DataFrame {df_idx+1}: missing or ambiguous required columns.")
            processed_dfs.append(df)
            continue

        # Process only the columns identified by the logic above
        for source_col in market_value_cols_to_process:
            new_col_name = 'Market Value USD'
            print(f"[DEBUG] Processing source_col: {source_col}")
            # Show a sample row and the values being passed to convert_row_to_usd
            if not df_copy.empty:
                sample_row = df_copy.iloc[0]
                print(f"[DEBUG] Sample row for apply: {sample_row.to_dict()}")
                print(f"[DEBUG] Sample value for date_col: {sample_row[date_col]}")
                print(f"[DEBUG] Sample value for currency_col: {sample_row[currency_col]}")
                print(f"[DEBUG] Sample value for source_col: {sample_row[source_col]}")
            try:
                df_copy[new_col_name] = df_copy.apply(
                    lambda row: convert_row_to_usd(row, source_col, date_col, currency_col, safe_get=lambda r, c: r[c] if not isinstance(r[c], (pd.Series, pd.Index)) else r[c].iloc[0]),
                    axis=1
                )
            except Exception as e:
                print(f"[ERROR] Exception while applying USD conversion for column '{source_col}': {e}")
                raise
        processed_dfs.append(df_copy)

    return processed_dfs

def convert_row_to_usd(row: pd.Series, value_col: str, date_col: str, currency_col: str, safe_get=None) -> Optional[float]:
    """Helper function for .apply() to convert a single row."""
    if safe_get is None:
        safe_get = lambda r, c: r[c]
    value_float = market_value_str_to_float(safe_get(row, value_col))
    date_obj = pd.to_datetime(safe_get(row, date_col), errors='coerce')
    currency = safe_get(row, currency_col)

    if pd.notna(value_float) and pd.notna(date_obj) and pd.notna(currency):
        time.sleep(0.1) # Simulate API delay
        rate = get_historical_exchange_rate_boc(currency, date_obj)
        if rate is not None:
            return value_float * rate
    return None

import pandas as pd

def add_account_type_for_multiple_accounts(df: pd.DataFrame, account_map_json: dict) -> pd.DataFrame:
    """
    Matches account numbers in a DataFrame with a JSON map, adds 'Account Type'.
    Handles cases where duplicate 'Account Number' columns may exist.

    Args:
        df: The pandas DataFrame containing an 'Account Number' column.
        account_map_json: The JSON object with the account number to type mappings.

    Returns:
        The DataFrame with an added 'Account Type' column.
    """
    if 'Account Number' not in df.columns:
        print(f"Warning: DataFrame does not have an 'Account Number' column. Skipping account type mapping for this table.")
        return df

    # 1. Create a clean lookup dictionary from the JSON object
    account_lookup = {
        item['account_number'].replace('*', ''): item['account_type']
        for item in account_map_json.get('accounts', [])
    }

    # --- START OF FIX ---
    # 2. Safely select the 'Account Number' column to ensure it's a Series

    account_number_data = df['Account Number']
    account_number_series = None

    # Check if the selection returned a DataFrame (due to duplicate columns)
    if isinstance(account_number_data, pd.DataFrame):
        print(f"Warning: Duplicate 'Account Number' columns detected in a table for file '{os.path.basename(df.attrs.get('source_file', 'Unknown'))}'. Using the first instance.")
        # Select the first column to ensure we have a Series
        account_number_series = account_number_data.iloc[:, 0]
    else:
        # It's already a Series, so we can use it directly
        account_number_series = account_number_data

    # Perform the string operations on the guaranteed Series object
    cleaned_numbers = account_number_series.astype(str).str.replace('*', '', regex=False)
    # --- END OF FIX ---

    # 3. Create the new 'Account Type' column by mapping the cleaned numbers
    df['Account Type'] = cleaned_numbers.map(account_lookup).fillna('None') # Added fillna for robustness

    return df

def add_account_type_for_one_account(df: pd.DataFrame, account_map_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds Account Number and Account Type columns to a DataFrame, assuming
    the JSON object contains exactly one account.

    This function overwrites any existing 'Account Number' or 'Account Type' columns.

    Args:
        df: The pandas DataFrame to be updated.
        account_map_json: The JSON object containing a single account's details.

    Returns:
        The DataFrame with added 'Account Number' and 'Account Type' columns.
    """
    accounts_list = account_map_json.get('accounts', [])

    # 1. Validate that there is exactly one account in the JSON
    if not isinstance(accounts_list, list) or len(accounts_list) != 1:
        print("Error: This function is for JSON objects with exactly one account.")
        print(f"Found {len(accounts_list) if isinstance(accounts_list, list) else 'invalid'} accounts.")
        return df

    # 2. Extract the single account's information
    single_account = accounts_list[0]
    account_number_raw = single_account.get('account_number', '')
    account_type = single_account.get('account_type', 'None')

    # 3. Clean the account number
    cleaned_account_number = account_number_raw.replace('*', '')

    # 4. Add the information directly to the DataFrame as new columns
    #    This will apply the same value to every row.
    df['Account Number'] = cleaned_account_number
    df['Account Type'] = account_type
    return df

def check_account_count(json_object: Dict[str, Any]) -> str:
    """
    Verifies account count and checks if a single account is empty.

    An account is considered "empty" if its dictionary is empty or if all its
    values are 'None' (either the None type or the string "None").

    Args:
        json_object: The JSON data (as a Python dictionary) to check.
                     Expected structure: {'accounts': [...]}.

    Returns:
        A string: "Single Account", "Multiple Accounts", "No Accounts Found",
        or "Single Account is Empty".
    """
    # Safely get the list of accounts
    accounts_list: List[Dict] = json_object.get('accounts', [])

    # Return if accounts key is not a list or the list is empty
    if not isinstance(accounts_list, list) or not accounts_list:
        return "No Accounts Found"

    # Check the number of accounts
    if len(accounts_list) > 1:
        return "Multiple Accounts"

    # At this point, there is only one account in the list.
    single_account = accounts_list[0]

    # Check if the account is an empty dictionary (e.g., {}) or None
    if not single_account:
        return "Single Account is Empty"

    # Check if all values in the account's dictionary are None or the string 'None'
    # This considers values like None, 'None'
    is_empty = all(value is None or str(value).lower() == 'none' for value in single_account.values())

    if is_empty:
        return "No Accounts Found"
    else:
        return "Single Account"
def save_dataframes_to_csv(list_of_dfs: List[pd.DataFrame], pdf_path: str, output_folder: str) :
    if not list_of_dfs:
        print("No DataFrames to save.")
        return

    base_name = os.path.basename(pdf_path)
    file_name, _ = os.path.splitext(base_name)

    for i, df in enumerate(list_of_dfs):
        # Create a unique name for each table's CSV file
        output_path = os.path.join(output_folder, f"{file_name}_table_{i+1}.csv")
        try:
            df.to_csv(output_path, index=False)
            print(f"Successfully saved table {i+1} to {output_path}")
        except Exception as e:
            print(f"Error saving table {i+1} to CSV: {e}")

# --- Main Processing Classes ---

STANDARD_COLUMNS = [
    "Date", "Security Description", "Symbol", "Security", "Currency", "Quantity", "Average Cost", "Book Value",
    "Price", "Market Value", "% Of Portfolio", "Accrued Int. & Pending Dividend", "Unrealized Gain/Loss",
    "Unrealized Gain/Loss (%)", "Estimated Annual Income", "Yield (%)", "Currency", "Account Number",
    "Asset Class L2", "Asset Class L1", "Account Type", "Market Value USD"
]

def standardize_markdown_with_gemini(markdown_tables: str) -> str:
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro', generation_config={'temperature': 0})
    prompt = f"""
    Please reformat the following markdown table(s) so that each table contains the following columns in this exact order:
    {', '.join(STANDARD_COLUMNS)}
    If a column does not exist in the original table, add it as an empty column.
    Output the result as markdown tables only.
    """
    response = model.generate_content([prompt, markdown_tables])
    return response.text

def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[STANDARD_COLUMNS]

class SingleAccountProcessor:
    """Processes a portfolio PDF determined to have one or no accounts."""
    def __init__(self, pdf_path: str, csv_path: str, asset_class_csv_path:str, account_details,output_dir: str):
        self.pdf_path = pdf_path
        self.file_name = os.path.basename(pdf_path)
        self.csv_path = csv_path
        self.asset_class_csv_path=asset_class_csv_path
        self.account_details=account_details
        self.output_dir = output_dir

    def run_processing_pipeline(self):
        print(f"[{self.file_name}] --- Running Single Account Pipeline ---")

        print(f"[{self.file_name}] --  Extracting Portfolio Date...")
        portfolio_date = get_portfolio_date(self.pdf_path)

        print(f"[{self.file_name}] -- Extracting Tables...")
        response_tables = extract_tables_from_pdf_one_account(self.pdf_path)
        if not response_tables:
            print("Failed to extract tables. Aborting.")
            return

        print(f"[{self.file_name}] -- Classifying Assets...")
        markdown_tables = get_asset_classes(response_tables)

        print(f"[{self.file_name}] -- Standardizing Markdown Tables with Gemini...")
        standardized_markdown = standardize_markdown_with_gemini(markdown_tables)

        print(f"[{self.file_name}] -- Converting to DataFrames...")
        list_of_dfs_simple = process_tables_individually(standardized_markdown, portfolio_date)
        list_of_dfs = add_asset_class_l1(list_of_dfs_simple,self.asset_class_csv_path)
        if not list_of_dfs:
            print("No data was processed into DataFrames. Aborting.")
            return

        print(f"[{self.file_name}] -- Adding Account Information...")
        final_dfs = [add_account_type_for_one_account(df, self.account_details) for df in list_of_dfs]

        print(f"[{self.file_name}] -- Converting Market Values to USD...")
        final_dfs_with_usd = add_usd_market_value(final_dfs)

        print(f"[{self.file_name}] -- Standardizing DataFrame Columns Locally...")
        standardized_final_dfs = [standardize_dataframe_columns(df) for df in final_dfs_with_usd]

        print(f"[{self.file_name}] -- Saving to CSV...")
        save_dataframes_to_csv(standardized_final_dfs, self.pdf_path, self.output_dir)
        print(f"[{self.file_name}] -- Single Account Processing Complete ---")

class MultipleAccountsProcessor:
    """Processes a portfolio PDF determined to have multiple accounts."""
    def __init__(self, pdf_path: str, csv_path: str, asset_class_csv_path:str, account_details,output_dir: str):
        self.pdf_path = pdf_path
        self.file_name = os.path.basename(pdf_path)
        self.csv_path = csv_path
        self.asset_class_csv_path=asset_class_csv_path
        self.account_details=account_details
        self.output_dir = output_dir

    def run_processing_pipeline(self):
        print(f"[{self.file_name}] --- Running Multiple Accounts Pipeline ---")
        print(f"[{self.file_name}] -- Step 2: Extracting Portfolio Date...")
        portfolio_date = get_portfolio_date(self.pdf_path)

        print(f"[{self.file_name}] -- Extracting Tables...")
        response_tables = extract_tables_from_pdf_multiple_accounts(self.pdf_path)
        if not response_tables:
            print("Failed to extract tables. Aborting.")
            return

        print(f"[{self.file_name}] -- Classifying Assets...")
        markdown_tables = get_asset_classes(response_tables)

        print(f"[{self.file_name}] -- Standardizing Markdown Tables with Gemini...")
        standardized_markdown = standardize_markdown_with_gemini(markdown_tables)

        print(f"[{self.file_name}] -- Converting to DataFrames...")
        list_of_dfs_simple = process_tables_individually(standardized_markdown, portfolio_date)
        list_of_dfs = add_asset_class_l1(list_of_dfs_simple,self.asset_class_csv_path)
        if not list_of_dfs:
            print("No data was processed into DataFrames. Aborting.")
            return

        print(f"[{self.file_name}] -- Adding Account Information...")
        final_dfs = [add_account_type_for_multiple_accounts(df, self.account_details) for df in list_of_dfs]

        print(f"[{self.file_name}] -- Converting Market Values to USD...")
        final_dfs_with_usd = add_usd_market_value(final_dfs)

        print(f"[{self.file_name}] -- Standardizing DataFrame Columns Locally...")
        standardized_final_dfs = [standardize_dataframe_columns(df) for df in final_dfs_with_usd]

        print(f"[{self.file_name}] --  Saving to CSV...")
        save_dataframes_to_csv(standardized_final_dfs, self.pdf_path, self.output_dir)
        print(f"[{self.file_name}] --- Multiple Accounts Processing Complete ---")


# --- Main Execution Block ---

def process_portfolio_document(pdf_path: str, csv_path: str, asset_class_csv_path:str, output_dir: str):
    """
    Main function to process a portfolio document. It determines the account
    structure and runs the appropriate processing pipeline.

    Args:
        pdf_path (str): Path to the input PDF file.
        csv_path (str): Path to the account definitions CSV file.
        output_dir (str): Folder where the final CSV files will be saved.
    """
    file_name = os.path.basename(pdf_path)
    start_time = time.monotonic() # <-- Record start time
    try:
        # 1. Validate paths and create output directory
        if not all([os.path.exists(pdf_path), os.path.exists(csv_path)]):
            print("Error: One or more input paths are invalid. Please check the file paths.")
            return
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # 2. Determine account structure from the document
        print(f"[{file_name}] Step 1: Determining account structure from document...")
        account_details = get_accounts_types(pdf_path, csv_path)

        if account_details == 'This is not a Portfolio Holdings Document':
            print(f"[{file_name}] is not a supported portfolio holdings file.")
            return

        # 3. Check the number of accounts found
        account_status = check_account_count(account_details)
        print(f"Document identified as: {account_status}")

        # 4. Instantiate and run the correct processor
        if account_status == "Multiple Accounts":
            processor = MultipleAccountsProcessor(
                pdf_path=pdf_path,
                csv_path=csv_path,
                asset_class_csv_path=asset_class_csv_path,
                account_details=account_details,
                output_dir=output_dir
            )
            processor.run_processing_pipeline()
        else:  # Handles "Single Account" and "No Accounts Found"
            processor = SingleAccountProcessor(
                pdf_path=pdf_path,
                csv_path=csv_path,
                asset_class_csv_path=asset_class_csv_path,
                account_details=account_details,
                output_dir=output_dir
            )
            processor.run_processing_pipeline()

        duration = time.monotonic() - start_time # <-- Calculate duration on success
        return f"Successfully processed {file_name}", duration

    except Exception as e:
        duration = time.monotonic() - start_time # <-- Calculate duration on failure
        print(f"!!!!!!!! An unexpected error occurred while processing {file_name}: {e} !!!!!!!!")
        return f"Failed to process {file_name}", duration


def main(
    pdf_files_to_process: List[str],
    output_csv_folder: str,
    account_definitions_csv: str,
    asset_class_csv: str,
    max_workers: int = 5
):
    """
    Sets up the environment and processes a list of PDF files using a thread pool.

    Args:
        pdf_files_to_process: A list of full paths to the PDF files.
        output_csv_folder: Path to the folder where output CSVs will be saved.
        account_definitions_csv: Path to the account definitions CSV file.
        asset_class_csv: Path to the asset class mapping CSV file.
        max_workers: The maximum number of threads to use.
    """
    # --- Setup ---
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_csv_folder):
        os.makedirs(output_csv_folder)
        print(f"Created output directory: {output_csv_folder}")

    # Check if the input list is empty
    if not pdf_files_to_process:
        print("The provided list of PDF files is empty. Exiting.")
        return

    print(f"Found {len(pdf_files_to_process)} PDF(s) to process. Starting multithreaded processing with {max_workers} workers.")

    # --- Multithreading Execution ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each PDF processing task to the executor
        future_to_pdf = {
            executor.submit(process_portfolio_document, pdf_path, account_definitions_csv, asset_class_csv, output_csv_folder): pdf_path
            for pdf_path in pdf_files_to_process
        }

        # Process results as they are completed
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                result = future.result()
                print(f"--- RESULT for {os.path.basename(pdf_path)}: {result} ---")
            except Exception as exc:
                print(f"--- ERROR for {os.path.basename(pdf_path)} generated an exception: {exc} ---")


if __name__ == "__main__":

    pdf_file = ["EB Aggressive UBS Template Final 20241001.pdf"]
    output_folder = "output/eb_ubs1"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["Etrade Employer Stock Plan Self-Managed - April.pdf"]
    output_folder = "output/etr_emp1"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["EB Aggressive UBS Template Final 20241001.pdf"]
    output_folder = "output/eb_ubs2"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["Etrade Employer Stock Plan Self-Managed - April.pdf"]
    output_folder = "output/etr_emp2"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["EB Aggressive UBS Template Final 20241001.pdf"]
    output_folder = "output/eb_ubs3"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["Etrade Employer Stock Plan Self-Managed - April.pdf"]
    output_folder = "output/etr_emp3"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["EB Aggressive UBS Template Final 20241001.pdf"]
    output_folder = "output/eb_ubs4"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)

    pdf_file = ["Etrade Employer Stock Plan Self-Managed - April.pdf"]
    output_folder = "output/etr_emp4"
    account_definitions_csv = "Account_Types.csv"
    asset_class_csv = "Shatterpoint - Parser - Asset Classification - v1.csv"

    main(pdf_file, output_folder, account_definitions_csv, asset_class_csv, max_workers=5)
