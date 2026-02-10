# data-quality-analyzer
Toolkit d'audit de qualite des donnees oriente data governance : metriques, detection d'anomalies, regles metier configurables et reporting automatique (HTML/CSV).

## Objectif
Analyser un dataset CSV, appliquer des regles YAML, calculer des metriques et un score global, puis generer un rapport HTML et des exports CSV.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Exemple de commande
```bash
dqa run --input examples/demo.csv --rules examples/rules.yml --out out --format html,csv
```

Options utiles:
- `--delimiter` pour changer le separateur CSV
- `--encoding` pour changer l'encodage
- `--id-cols` pour definir une cle d'unicite (ex: `id` ou `colA,colB`)
- `--verbose` pour logs detailes

## Regles YAML
Le fichier `rules.yml` supporte:
```yaml
columns:
	id:
		type: int
		required: true
	age:
		type: int
		required: true
		min: 0
	email:
		type: string
		regex: "^[^@]+@[^@]+\\.[^@]+$"
	end_date:
		type: date
		not_future: true
unique_keys:
	- [id]
row_rules:
	- name: "start_before_end"
		expr: "start_date <= end_date"
```

### Expression `row_rules`
Expressions autorisees:
- Operateurs: `<=`, `<`, `>=`, `>`, `==`, `!=`, `and`, `or`
- Fonctions: `is_null(x)`, `not_null(x)`
- Acces direct aux colonnes (ex: `age`, `start_date`)

## Score
Sous-scores 0-100 par dimension, puis score global par pondération:
- completeness 25
- validity 25
- uniqueness 20
- consistency 20
- outliers 10

Chaque dimension penalise le taux d'erreurs associe (ex: valeurs manquantes, doublons, violations de regles).

## Outputs
La commande genere un dossier de sortie (ex: `out/`) contenant:
- `summary.csv`: metriques par colonne + ligne globale
- `issues.csv`: liste des problemes detectes
- `plots/`: graphiques PNG (missing, distributions, outliers)
- `report.html`: rapport HTML (si format html)

## Limitations
- L'inference de type utilise des heuristiques et peut etre approximative.
- Les expressions `row_rules` supportent un sous-ensemble strict pour la securite.
- Le traitement est en memoire (pas de streaming pour tres gros CSV).
