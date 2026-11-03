call DBMS_AI_SERVICE.CREATE_AI_MODEL('om','{"type":"completion","model_name":"qwen2.5"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('ome','{"ai_model_name":"om","scope":"all","url":"http://0.0.0.0:8000/api/generate","access_key":"","provider":"ollama"}');
select '--- after create ---' as mark;
select endpoint_name, url, access_key, provider, request_model_name, parameters,
       request_transform_fn, response_transform_fn, scope
  from oceanbase.DBA_OB_AI_MODEL_ENDPOINTS where endpoint_name='ome'\G
call DBMS_AI_SERVICE.ALTER_AI_MODEL_ENDPOINT('ome','{"provider":"Ollama"}');
select '--- after alter ---' as mark;
select endpoint_name, url, access_key, provider, request_model_name, parameters,
       request_transform_fn, response_transform_fn, scope
  from oceanbase.DBA_OB_AI_MODEL_ENDPOINTS where endpoint_name='ome'\G
select length(ai_complete('om','hello')) as ollama_ok;
call DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ome');
call DBMS_AI_SERVICE.DROP_AI_MODEL('om');
