import csv
import unicodedata

def process_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = [row for row in reader]

    processed_rows = [
        [
            remove_accents(cell).upper() if cell.strip() else "ND"
            for cell in row
        ] for row in rows
    ]

    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

# Exemplos de uso:
# Altere 'input.csv' e 'output.csv' para os nomes dos seus arquivos.
input_csv = 'Teste.csv'
output_csv = 'teste_formatado.csv'

def print_column(input_file, column_name):
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)  # Usa DictReader para acessar colunas por nome
        if column_name not in reader.fieldnames:
            print(f"A coluna '{column_name}' não foi encontrada no arquivo.")
            return

        print(f"Valores da coluna '{column_name}':")
        for row in reader:
            print(row[column_name] if row[column_name].strip() else "ND")

#process_csv(input_csv, output_csv)
print_column(input_csv,"Andamento atual")

print(f"Arquivo processado e salvo como {output_csv}")
