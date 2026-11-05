import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_path = "data/raw/datos_abiertos_vigilancia_iras_2000_2025 PUNO.csv"
df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"ubigeo": str})

total_direct = int(df["ira_no_neumonia"].sum())
prov_sums = df.groupby("provincia")["ira_no_neumonia"].sum()
prov_total = int(prov_sums.sum())

print(f"Total directo df['ira_no_neumonia'].sum(): {total_direct:,}")
print(f"Suma de prov_sums.sum(): {prov_total:,}")
print(f"Diferencia directa vs prov_sums: {total_direct - prov_total}")

table_values = {
    "SAN ROMAN": 569203,
    "PUNO": 383892,
    "AZANGARO": 254292,
    "CARABAYA": 180457,
    "MELGAR": 177949,
    "LAMPA": 138722,
    "SANDIA": 132521,
    "CHUCUITO": 116127,
    "HUANCANE": 99744,
    "EL COLLAO": 97619,
    "SAN ANTONIO DE PUTINA": 68673,
    "YUNGUYO": 54161,
    "MOHO": 24018
}

sum_table = sum(table_values.values())
print(f"\nSuma manual de los 13 valores en la tabla del informe anterior: {sum_table:,}")
print(f"Diferencia observada por el usuario: {total_direct - sum_table:,}")

print("\n--- COMPARACIÓN PROVINCIA POR PROVINCIA ---")
for p in sorted(prov_sums.keys()):
    real_val = int(prov_sums[p])
    tab_val = table_values.get(p, None)
    diff = real_val - tab_val if tab_val is not None else "N/A"
    print(f"Provincia: {p:22} | Real CSV: {real_val:8,d} | Tabla Informe: {tab_val:8,d} | Diferencia: {diff}")
