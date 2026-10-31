call DBMS_AI_SERVICE.CREATE_AI_MODEL('m2', '{"type":"dense_embedding","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e2', '{"ai_model_name":"m2","url":"http://127.0.0.1:8000","access_key":"sk-mock","provider":"openai"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL('m3', '{"type":"rerank","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e3', '{"ai_model_name":"m3","url":"http://127.0.0.1:8000","access_key":"sk-mock","provider":"openai"}');
