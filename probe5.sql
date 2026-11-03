select '--- create ---' as mark;
call DBMS_AI_SERVICE.CREATE_AI_MODEL('om4','{"type":"completion","model_name":"qwen2.5"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('ome4','{"ai_model_name":"om4","scope":"all","url":"http://0.0.0.0:8000/api/generate","access_key":"","provider":"ollama"}');
select * from oceanbase.DBA_OB_AI_MODEL_ENDPOINTS where endpoint_name='ome4'\G
select '--- alter provider ---' as mark;
call DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('ome4','{"provider":"Ollama"}');
select * from oceanbase.DBA_OB_AI_MODEL_ENDPOINTS where endpoint_name='ome4'\G
call DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ome4');
call DBMS_AI_SERVICE.DROP_AI_MODEL('om4');
