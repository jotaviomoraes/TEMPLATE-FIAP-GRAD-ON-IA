-- =====================================================================
-- Sompo AgroPredict - Queries de Consulta e Analise
-- Sprint 2 - Challenge Sompo x FIAP
-- =====================================================================
-- Conjunto de queries uteis para alimentar dashboards, relatorios da
-- Sompo e auditoria das decisoes do modelo.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Distribuicao geral de risco
-- ---------------------------------------------------------------------
SELECT
    classe_risco,
    COUNT(*) AS quantidade,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM predicoes), 2) AS percentual
FROM predicoes
GROUP BY classe_risco
ORDER BY DECODE(classe_risco, 'BAIXO', 1, 'MEDIO', 2, 'ALTO', 3);


-- ---------------------------------------------------------------------
-- 2. Top 10 predicoes mais criticas
-- ---------------------------------------------------------------------
SELECT
    p.id_predicao,
    TO_CHAR(p.data_hora, 'DD/MM/YYYY HH24:MI') AS quando,
    e.modelo AS equipamento,
    o.nome AS operador,
    p.classe_risco,
    p.score_risco,
    p.recomendacao
FROM predicoes p
LEFT JOIN equipamentos e ON e.id_equipamento = p.id_equipamento
LEFT JOIN operadores   o ON o.id_operador    = p.id_operador
ORDER BY p.score_risco DESC
FETCH FIRST 10 ROWS ONLY;


-- ---------------------------------------------------------------------
-- 3. Score medio por equipamento (para gestor de frota)
-- ---------------------------------------------------------------------
SELECT
    e.id_equipamento,
    e.modelo,
    e.fabricante,
    COUNT(p.id_predicao) AS total_operacoes,
    ROUND(AVG(p.score_risco), 2)   AS score_medio,
    MAX(p.score_risco)             AS score_maximo,
    SUM(CASE WHEN p.classe_risco = 'ALTO' THEN 1 ELSE 0 END) AS qtd_alto
FROM predicoes p
JOIN equipamentos e ON e.id_equipamento = p.id_equipamento
GROUP BY e.id_equipamento, e.modelo, e.fabricante
ORDER BY score_medio DESC;


-- ---------------------------------------------------------------------
-- 4. Taxa de aderencia dos operadores aos alertas (relatorio Sompo)
-- ---------------------------------------------------------------------
SELECT
    o.id_operador,
    o.nome,
    COUNT(a.id_alerta) AS total_alertas,
    SUM(CASE WHEN a.acao_tomada = 'RESPEITOU' THEN 1 ELSE 0 END) AS respeitou,
    SUM(CASE WHEN a.acao_tomada = 'IGNOROU'   THEN 1 ELSE 0 END) AS ignorou,
    SUM(CASE WHEN a.acao_tomada = 'PENDENTE'  THEN 1 ELSE 0 END) AS pendente,
    ROUND(
        SUM(CASE WHEN a.acao_tomada = 'RESPEITOU' THEN 1 ELSE 0 END) * 100.0
        / NULLIF(COUNT(a.id_alerta), 0), 2
    ) AS taxa_aderencia_pct
FROM operadores o
LEFT JOIN predicoes p ON p.id_operador = o.id_operador
LEFT JOIN alertas a   ON a.id_predicao = p.id_predicao
GROUP BY o.id_operador, o.nome
ORDER BY taxa_aderencia_pct DESC NULLS LAST;


-- ---------------------------------------------------------------------
-- 5. Identificacao de fatores de risco predominantes
-- (auxilia analise da Sompo sobre o que mais causa sinistros)
-- ---------------------------------------------------------------------
SELECT
    'Solo saturado + chuva recente' AS cenario,
    COUNT(*) AS qtd
FROM predicoes
WHERE classe_risco IN ('MEDIO','ALTO')
  AND umidade_solo > 70 AND chuva_24h > 20
UNION ALL
SELECT 'Velocidade alta em terreno inclinado', COUNT(*)
FROM predicoes
WHERE classe_risco IN ('MEDIO','ALTO')
  AND velocidade > 15 AND declividade > 15
UNION ALL
SELECT 'Maquina antiga em uso intenso', COUNT(*)
FROM predicoes
WHERE classe_risco IN ('MEDIO','ALTO')
  AND idade_maquina > 12 AND horas_operacao > 9
UNION ALL
SELECT 'Calor extremo + jornada longa', COUNT(*)
FROM predicoes
WHERE classe_risco IN ('MEDIO','ALTO')
  AND temperatura > 35 AND horas_operacao > 10
ORDER BY qtd DESC;


-- ---------------------------------------------------------------------
-- 6. Metricas diarias (view consolidada)
-- ---------------------------------------------------------------------
SELECT * FROM vw_metricas_diarias ORDER BY dia DESC;
