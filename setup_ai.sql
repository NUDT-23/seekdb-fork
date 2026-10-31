call DBMS_AI_SERVICE.CREATE_AI_MODEL('m1', '{"type":"completion","model_name":"mock"}');
call DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('e1', '{"ai_model_name":"m1","url":"http://127.0.0.1:8000","access_key":"sk-mock","provider":"openai"}');
