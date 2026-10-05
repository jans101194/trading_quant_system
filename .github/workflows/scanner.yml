name: Automated Market Scanner

on:
  schedule:
    # Se ejecuta de lunes a viernes a las 22:00 UTC (al cierre de mercados)
    - cron: '0 22 * * 1-5'
  workflow_dispatch: # Permite lanzarlo manualmente desde la pestaña "Actions" en GitHub

jobs:
  run-scanner:
    runs-on: ubuntu-latest
    steps:
      - name: Clonar repositorio
        uses: actions/checkout@v4

      - name: Configurar entorno Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Instalar dependencias del sistema
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Ejecutar escáner cuantitativo
        run: python src/scanner.py

      - name: Sincronizar datos actualizados en el repositorio
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add data/latest_market_scan.json data/signals_log.csv
          git diff --quiet && git diff --staged --quiet || (git commit -m "Actualización automática de señales de mercado [skip ci]" && git push)