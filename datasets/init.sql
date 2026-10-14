INSTALL excel;
LOAD excel;

-- Planilhas ANP: ~17 linhas de metadados antes do cabeçalho; datas vêm como serial Excel.
CREATE OR REPLACE MACRO anp_xlsx(path) AS TABLE
SELECT
    DATE '1899-12-30' + CAST(A AS INTEGER)                  AS data_inicial,
    DATE '1899-12-30' + CAST(B AS INTEGER)                  AS data_final,
    upper(strip_accents(trim(C)))                           AS produto,
    CAST(D AS INTEGER)                                      AS numero_postos,
    E                                                       AS unidade_medida,
    TRY_CAST(F AS DOUBLE)                                   AS preco_medio_revenda,
    TRY_CAST(G AS DOUBLE)                                   AS desvio_padrao_revenda,
    TRY_CAST(H AS DOUBLE)                                   AS preco_minimo_revenda,
    TRY_CAST(I AS DOUBLE)                                   AS preco_maximo_revenda,
    TRY_CAST(J AS DOUBLE)                                   AS margem_media_revenda,
    TRY_CAST(K AS DOUBLE)                                   AS coef_variacao_revenda,
    TRY_CAST(L AS DOUBLE)                                   AS preco_medio_distribuicao,
    TRY_CAST(M AS DOUBLE)                                   AS desvio_padrao_distribuicao,
    TRY_CAST(N AS DOUBLE)                                   AS preco_minimo_distribuicao,
    TRY_CAST(O AS DOUBLE)                                   AS preco_maximo_distribuicao,
    TRY_CAST(P AS DOUBLE)                                   AS coef_variacao_distribuicao,
    regexp_extract(path, '[^/]+$')                          AS arquivo_origem
FROM read_xlsx(path, header = false, all_varchar = true, range = 'A1:Q50000')
WHERE TRY_CAST(A AS INTEGER) IS NOT NULL AND C IS NOT NULL;

CREATE OR REPLACE TABLE precos_semanal_raw AS
SELECT * FROM anp_xlsx('/data/raw/semanal-brasil-2004-a-2012.xlsx')
UNION ALL
SELECT * FROM anp_xlsx('/data/raw/semanal-brasil-desde-2013.xlsx');

-- Remove semanas repetidas na fronteira entre os dois arquivos.
CREATE OR REPLACE TABLE precos_semanal AS
SELECT * EXCLUDE (rn)
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY data_inicial, data_final, produto ORDER BY arquivo_origem
    ) AS rn
    FROM precos_semanal_raw
)
WHERE rn = 1
ORDER BY data_inicial, produto;

CREATE OR REPLACE TABLE serie_semanal_nacional AS
SELECT
    CAST(data_inicial AS DATE) AS data_inicial,
    * EXCLUDE (data_inicial)
FROM read_parquet('/data/processed/serie_semanal_nacional.parquet')
ORDER BY data_inicial;

SELECT 'precos_semanal_raw' AS tabela, count(*) AS linhas FROM precos_semanal_raw
UNION ALL SELECT 'precos_semanal', count(*) FROM precos_semanal
UNION ALL SELECT 'serie_semanal_nacional', count(*) FROM serie_semanal_nacional;
