call DBMS_AI_SERVICE.CREATE_AI_MODEL('m4', '{"type":"completion","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e4', '{"ai_model_name":"m4","url":"http://127.0.0.1:8000","access_key":"","provider":"ollama"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL('m5', '{"type":"dense_embedding","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e5', '{"ai_model_name":"m5","url":"http://127.0.0.1:8000","access_key":"","provider":"ollama"}');
