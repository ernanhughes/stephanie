SELECT DISTINCT s.score, ds.section_text
FROM scores s
JOIN evaluations e ON s.evaluation_id = e.id
JOIN document_sections ds ON e.document_id = ds.document_id
    AND ds.section_name = 'title'
WHERE s.dimension = 'relevance'
  AND s.score > 74
  AND e.document_id > 0
ORDER BY s.score DESC, ds.section_text;