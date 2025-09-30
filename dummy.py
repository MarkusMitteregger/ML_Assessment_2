import pandas as pd

def diagnose_disease(symptom1: bool, symptom2: bool, symptom3: bool, symptom4: bool, symptom5: bool) -> str:
    """
    This dummy function attempts to differentiate between two fictional diseases
    based on 5 symptoms as input.
    """
    # Disease A: Requires a combination of Fever (symptom1), Cough (symptom2), and Fatigue (symptom4)
    if symptom1 and symptom2 and symptom4 and not symptom5:
        return "Krankheit A (Grippe-ähnlich)"
    # Disease B: Requires a combination of Headache (symptom3), Rash (symptom5), and Fatigue (symptom4)
    elif symptom3 and symptom5 and symptom4 and not symptom1:
        return "Krankheit B (Allergische Reaktion)"
    else:
        return "Keine klare Diagnose"

# --- Main execution block ---

# Define the path to your CSV file
csv_file_path = 'dataset/dataset.csv' # <-- Name of your downloaded file

print("--- Diagnose basierend auf dem CSV-Datensatz ---")

try:
    # Load the dataset using pandas
    df = pd.read_csv(csv_file_path)

    # --- FIX ---
    # Clean up column names: remove leading/trailing whitespace.
    df.columns = df.columns.str.strip()

    # Check if the 'Disease' column exists. If not, print available columns and exit.
    if 'Disease' not in df.columns:
        print("\n[Error] The column 'Disease' was not found in your CSV file.")
        print("Please check the correct column name for the diagnosis.")
        print("Available columns are:", list(df.columns))
    else:
        # We'll just take the first 5 patients as an example
        for index, patient_row in df[:5].iterrows():
            patient_id = f"Patient_{index + 1}"
            
            # Map the symptoms from the CSV to your function's arguments.
            # NOTE: The column names must exactly match those in the CSV.
            # Since the dataset doesn't have 'fever' or 'cough', we'll make some assumptions.
            # This demonstrates an important concept: "Feature Mapping".
            
            # We check if the columns exist before accessing them to avoid errors.
            symptom1_val = bool(patient_row.get('fever', 0)) # The dataset has no 'fever' column, so this will be 0
            symptom2_val = bool(patient_row.get('cough', 0)) # The dataset has no 'cough' column, so this will be 0
            symptom3_val = bool(patient_row.get('headache', 0)) # This column exists
            symptom4_val = bool(patient_row.get('fatigue', 0))  # This column exists
            symptom5_val = bool(patient_row.get('skin_rash', 0)) # We'll use 'skin_rash' for 'rash'
            
            symptoms_for_diagnosis = (symptom1_val, symptom2_val, symptom3_val, symptom4_val, symptom5_val)
            
            diagnosis = diagnose_disease(*symptoms_for_diagnosis)
            
            # Get the actual diagnosis from the dataset for comparison
            actual_disease = patient_row['Disease']
            
            print(f"\n{patient_id}:")
            print(f"  > Echte Diagnose laut Datensatz: {actual_disease}")
            print(f"  > Dummy-Funktion Diagnose: {diagnosis}")


except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    print(f"Please make sure the file is in the same directory as this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")