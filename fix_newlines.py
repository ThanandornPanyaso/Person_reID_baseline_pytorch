import pandas as pd

metadata_csv = "/home/big/Github/dataset_prepare/get_img_cosine_renew800/output/metadata.csv"

# Read the file manually and fix broken lines
fixed_lines = []
with open(metadata_csv, "r", encoding="utf-8") as f:
    current_line = ""

    for line in f:
        # If the line starts with a digit (likely class_id), it's a new row
        if line[0].isdigit():
            if current_line:  # Save the previous complete row
                fixed_lines.append(current_line.strip())
            current_line = line.strip()  # Start a new row
        else:
            current_line += " " + line.strip()  # Append to the previous row

    if current_line:  # Save the last row
        fixed_lines.append(current_line.strip())

# Convert fixed lines into a DataFrame
from io import StringIO
csv_fixed_content = "\n".join(fixed_lines)

df = pd.read_csv(StringIO(csv_fixed_content))

print(df.head())  # Verify the output
csv_fixed_content = StringIO(csv_fixed_content).getvalue()
with open("cleaned_metadata800.csv", "w", encoding="utf-8") as f:
    f.write(csv_fixed_content)