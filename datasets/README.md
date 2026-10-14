# Banco DuckDB (ANP)

Arquivos desta pasta:

- `anp_combustiveis/`: dados brutos (`raw/`) e processados (`processed/`).
- `init.sql`: cria as tabelas no DuckDB a partir desses dados.
- `anp.duckdb`: banco gerado (versionado).

## Tabelas

| Tabela | Descrição |
|---|---|
| `precos_semanal_raw` | Os dois XLSX da ANP (2004-2012 e desde 2013) unidos |
| `precos_semanal` | `precos_semanal_raw` sem semanas repetidas entre os arquivos |
| `serie_semanal_nacional` | Série processada (parquet) usada nos modelos |

## Criar o banco com Docker

Execute a partir desta pasta (`datasets/`):

```bash
docker run --rm \
  -v "$PWD/anp_combustiveis:/data:ro" \
  -v "$PWD:/work" -w /work \
  --entrypoint duckdb duckdb/duckdb \
  anp.duckdb -f /work/init.sql
```

O comando recria as tabelas a cada execução (`CREATE OR REPLACE`).

## Consultar via terminal

```bash
docker run --rm -it -v "$PWD:/work" -w /work --entrypoint duckdb duckdb/duckdb anp.duckdb
```

Somente leitura (permite outros processos lendo o arquivo):

```bash
docker run --rm -v "$PWD:/work" -w /work --entrypoint duckdb duckdb/duckdb -readonly anp.duckdb -c "SELECT * FROM precos_semanal LIMIT 5;"
```

## Conectar o DBeaver

1. *Database > New Database Connection* e escolha **DuckDB**.
2. Se solicitado, deixe o DBeaver baixar o driver.
3. No campo *Path*, informe o caminho absoluto de `anp.duckdb` no macOS, por exemplo:
   `/Users/<usuario>/.../projeto-anp/datasets/anp.duckdb`
4. Clique em *Test Connection* e depois em *Finish*.
5. As tabelas ficam em `anp > main > Tables`.

URL JDBC equivalente: `jdbc:duckdb:/caminho/para/anp.duckdb`.

### Cuidados

- O DuckDB permite só um processo com acesso de escrita por vez. Feche a conexão do DBeaver antes de recriar o banco com o Docker, para evitar erro de lock.
- Para apenas consultar, marque *Read-only connection* em *Connection settings > General*.
- O DBeaver abre o arquivo diretamente pelo driver JDBC, então use o caminho do macOS, não o `/work` do container.
